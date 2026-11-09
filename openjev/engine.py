"""The QwenJev-lite engine: shared state, isolated branches, parallel readouts."""

from __future__ import annotations

import copy
import time
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .config import JevLimits, QwenJevConfig
from .confidence import billing_output_tokens, summarise
from .prompt import render_branch, render_state
from .readout import Readout, build_readout
from .schema import (
    MAX_LETTER_LABELS,
    Question,
    QuestionType,
    bool_true_key,
    build_question,
    question_option_ids,
)
from .tokenize import TokenizedBranch, tokenize_branch


def load_backbone(
    model_path: str,
    *,
    dtype,
    device: str,
    text_only: bool = False,
):
    """Load any Qwen backbone, multimodal or text-only.

    The engine only ever asks a backbone for hidden states and an output embedding, so
    it does not care whether the checkpoint has a vision tower. Qwen3.5 ships as
    ``Qwen3_5ForConditionalGeneration`` (image-text-to-text); a plain causal LM - a
    text-only Qwen3, a bigger dense model, an MoE - resolves under
    ``AutoModelForCausalLM`` instead, and that class is the fallback rather than a
    second code path. The multimodal class is tried first only because it is what this
    project's numbers were measured on.
    """

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    kwargs = {
        "dtype": dtype,
        "device_map": {"": device},
        "local_files_only": True,
    }
    if text_only:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            print(
                f"note: {model_path!r} did not load as an image-text-to-text model ({exc}); "
                f"trying a text-only causal LM",
                file=sys.stderr,
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model


def backbone_hidden_size(model) -> int:
    """Width of a backbone's residual stream, wherever that model reports it.

    The decision head is sized from this, so nothing here may assume a particular
    checkpoint: Qwen3.5 puts it under ``config.text_config``, a text-only causal LM
    keeps it at the top level, and a wrapper may nest it one level deeper. Reading it
    from the model instead of a constant is what lets the same code follow the
    backbone's actual residual width.
    """

    config = getattr(model, "config", None)
    for attribute in ("text_config", "language_config", "llm_config", "language_model_config"):
        nested = getattr(config, attribute, None)
        size = getattr(nested, "hidden_size", None)
        if size:
            return int(size)
    size = getattr(config, "hidden_size", None)
    if size:
        return int(size)
    raise ValueError(
        f"cannot read hidden_size from {type(config).__name__}; the decision head needs "
        f"the width of the backbone's residual stream"
    )


@dataclass
class QuestionDecision:
    """The parallel numerical answer to one question."""

    question_id: str
    type: str
    probabilities: "OrderedDict[str, float]"
    answer: str
    confidence: float
    score_value: float | None = None
    raw_logits: list[float] | None = None
    #: The letter the model actually reads for each answer, in order: ``{"A": <key>, …}``.
    #: Carried in the response so the caller can see the mapping back to its own criteria.
    slots: dict[str, str] = field(default_factory=dict)
    #: For a boolean question, the key that means "yes" - resolved from the caller's own
    #: criteria names, so a custom pair such as ``true``/``false`` is not inverted.
    bool_true_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "answer": self.answer,
        }
        if self.type == "bool":
            payload["probability"] = round(self.probabilities[self.answer], 6)
            payload["answer"] = self.answer == (self.bool_true_key or "yes")
        if self.score_value is not None:
            payload["score"] = round(self.score_value, 6)
        if self.slots:
            payload["slots"] = dict(self.slots)
        payload["confidence"] = round(self.confidence, 6)
        return payload


@dataclass
class JevResponse:
    decisions: "OrderedDict[str, QuestionDecision]"
    usage: dict[str, int]
    timing: dict[str, float | bool]
    hidden: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": {qid: d.to_dict() for qid, d in self.decisions.items()},
            "usage": self.usage,
            "timing": self.timing,
        }


class _LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._store: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: Any, value: Any) -> None:
        if self.maxsize <= 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


class QwenJevLite:
    """Encode the state once, answer every question in parallel, read out numbers."""

    def __init__(self, model, tokenizer, config: QwenJevConfig | None = None, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        #: The multimodal processor (image resize + patchify + ``mm_token_type_ids``).
        #: ``None`` for a text-only backbone, in which case image states raise.
        self.processor = processor
        self.config = config or QwenJevConfig()
        self.device = model.get_input_embeddings().weight.device

        lm_head = model.get_output_embeddings()
        if lm_head is None:  # pragma: no cover - defensive
            raise ValueError("backbone has no output embedding / lm_head")
        hidden_size = backbone_hidden_size(model)
        # A checkpoint fixes the width of its own head (older runs used 256 rows), so
        # read it before building the module rather than after. When the configuration
        # allows more rows than the checkpoint has, the head is *grown*: the trained rows
        # are copied in and the free rows start at zero, which is what lets a full head
        # learn the answers of one more task instead of quietly falling back to the
        # reserved letter rows.
        state = None
        if self.config.readout_checkpoint:
            state = torch.load(self.config.readout_checkpoint, map_location=self.device)
        self._auto_readouts: dict[str, Readout] = {}
        if self.config.readout == "auto":
            # ``auto`` owns no weights: it is two zero-shot readouts and a rule, and the
            # option count of each request picks between them (see :meth:`readout_for`).
            if state is not None:
                raise ValueError(
                    "readout='auto' cannot be loaded from a checkpoint; name the readout "
                    "the checkpoint was trained with (e.g. 'slot_head')"
                )
            self._auto_readouts = {
                name: build_readout(
                    name,
                    tokenizer,
                    lm_head,
                    hidden_size,
                    max_slots=self.limits.max_slots,
                    max_options=self.limits.max_options,
                ).to(self.device)
                for name in ("reserved_label", "verbalizer")
            }
            for readout in self._auto_readouts.values():
                readout.eval()
            self.readout = self._auto_readouts["reserved_label"]
            self.readout_temperature = 1.0
            self._state_cache = _LRUCache(self.config.state_cache_size)
            return
        max_slots = self.limits.max_slots
        if state is not None and isinstance(state.get("readout"), dict):
            rows = state["readout"].get("proj.weight")
            if rows is not None:
                max_slots = max(int(rows.shape[0]), self.limits.max_slots)
        self.readout: Readout = build_readout(
            self.config.readout,
            tokenizer,
            lm_head,
            hidden_size,
            max_slots=max_slots,
            max_options=self.limits.max_options,
        ).to(self.device)
        self.readout.eval()
        self.readout_temperature = 1.0
        if state is not None:
            trained_with = state.get("readout_name")
            if trained_with and trained_with != self.readout.name:
                raise ValueError(
                    f"checkpoint {self.config.readout_checkpoint!r} was trained with the "
                    f"{trained_with!r} readout but the engine is using {self.readout.name!r}; "
                    f"pass readout={trained_with!r} as well"
                )
            weights = dict(state["readout"])
            tensor = weights.get("proj.weight")
            discarded = False
            if tensor is not None and int(tensor.shape[1]) != hidden_size:
                # The head was trained against a different backbone. Its rows are
                # directions in *that* model's residual stream, so a wider or narrower
                # backbone cannot reuse them - unlike the row *count*, which is grown
                # below. Keep the row assignment (the slot table) so a run that moves a
                # recipe from one backbone to another can train from here.
                print(
                    f"note: checkpoint {self.config.readout_checkpoint!r} holds a "
                    f"{int(tensor.shape[1])}-wide head but this backbone is {hidden_size}-wide; "
                    f"trained rows do not transfer across backbones, so they are discarded "
                    f"and the head is re-initialised from this model's own label rows",
                    file=sys.stderr,
                )
                weights = {}
                tensor = None
                discarded = True
            if tensor is not None and int(tensor.shape[0]) != int(self.readout.max_slots):
                grown = torch.zeros(
                    int(self.readout.max_slots), int(tensor.shape[1]), dtype=tensor.dtype
                )
                grown[: int(tensor.shape[0])] = tensor
                weights["proj.weight"] = grown
            if weights:
                self.readout.load_state_dict(weights)
            if hasattr(self.readout, "set_slot_table"):
                self.readout.set_slot_table(state.get("slot_table"))
            if not discarded:
                self.readout_temperature = float(state.get("temperature", 1.0))
        self._state_cache = _LRUCache(self.config.state_cache_size)

    @property
    def limits(self) -> JevLimits:
        return self.config.limits

    def clear_state_cache(self) -> None:
        """Drop every encoded state."""

        self._state_cache._store.clear()

    def set_state_cache_size(self, size: int) -> int:
        """Resize the state cache; returns the previous size."""

        previous = self._state_cache.maxsize
        self._state_cache.maxsize = size
        self.config.state_cache_size = size
        if size <= 0:
            self._state_cache._store.clear()
        return previous

    def readout_for(self, question: Question) -> Readout:
        """The readout that answers *this* question.

        With ``readout="auto"`` the choice follows the question's own option count: the
        reserved letters reach :data:`MAX_LETTER_LABELS`, anything wider is scored by the
        option text through :class:`qwenjev.readout.VerbalizerReadout`. Any other
        ``readout`` setting is a single fixed readout and is returned as is.

        It is deliberately *per question*, not per request. A request that mixes a 3-way
        and a 200-way question used to run both through the wide path, because the choice
        was made once from the widest question - the narrow one silently paid for the
        wide one's protocol. Questions are independent, so the answer is per question.
        """

        if not self._auto_readouts:
            return self.readout
        return self._auto_readouts[
            "reserved_label"
            if len(question.options) <= MAX_LETTER_LABELS
            else "verbalizer"
        ]

    def readouts_for(self, questions: Sequence[Question]) -> list[Readout]:
        """One readout per question, in the same order."""

        return [self.readout_for(question) for question in questions]

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_path: str | None = None, config: QwenJevConfig | None = None, **kwargs):
        from transformers import AutoProcessor, AutoTokenizer

        config = config or QwenJevConfig(**kwargs)
        if model_path is not None:
            config.model_path = model_path
        if not Path(config.model_path).exists() and not Path(config.model_path).is_absolute():
            # A relative path that is not there is the usual first-run mistake; say what
            # to do instead of letting the Hub error out.
            print(
                f"note: backbone {config.model_path!r} is not on disk; pass --model <path>, "
                f"put the checkpoint there, or set QWENJEV_MODEL. Anything transformers can "
                f"load works (a local path or a Hub repo id).",
                file=sys.stderr,
            )
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(config.model_path)
        except Exception as exc:  # a text-only checkpoint simply has no processor
            print(
                f"note: no multimodal processor for {config.model_path!r} ({exc}); "
                f"records with images will be rejected",
                file=sys.stderr,
            )
        model = load_backbone(
            config.model_path,
            dtype=getattr(torch, config.dtype),
            device=config.device,
            text_only=config.text_only,
        )
        return cls(model, tokenizer, config, processor=processor)

    # -- state encoding --------------------------------------------------------
    def _state_payload(self, state: str, images: Sequence[str] = ()) -> dict:
        """Backbone inputs for one state: text ids, or the processor's multimodal set.

        The image tokens live *inside the state*, so every question about that state
        shares them - they are encoded once and live in the KV cache afterwards.
        """

        if not images:
            ids = list(self.tokenizer(render_state(state), add_special_tokens=False)["input_ids"])
            return {
                "input_ids": torch.tensor([ids], device=self.device),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
            }
        if self.processor is None:
            raise RuntimeError(
                "this backbone has no multimodal processor; run with a vision-language "
                "checkpoint (the state carries images)"
            )
        if len(images) > self.config.max_images:
            raise ValueError(
                f"state has {len(images)} images but max_images is {self.config.max_images}"
            )
        from PIL import Image

        opened = []
        # transformers 5 moved the pixel budget onto the image processor's ``size``
        # (``longest_edge`` is the old ``max_pixels``); passing ``max_pixels`` as a
        # keyword is silently ignored, which leaves every picture at full resolution.
        default_size = getattr(self.processor.image_processor, "size", None)
        shortest_edge = int(getattr(default_size, "shortest_edge", 0) or 65536)
        try:
            for path in images:
                with Image.open(path) as handle:
                    opened.append(handle.convert("RGB"))
            payload = self.processor(
                text=[render_state(state, len(opened))],
                images=[opened],
                return_tensors="pt",
                size={
                    "longest_edge": int(self.config.image_max_pixels),
                    "shortest_edge": shortest_edge,
                },
            )
        finally:
            for image in opened:
                image.close()
        return {
            key: (
                value.to(self.device, dtype=self.model.dtype)
                if torch.is_floating_point(value)
                else value.to(self.device)
            )
            for key, value in payload.items()
        }

    @torch.no_grad()
    def encode_state(self, state: str, images: Sequence[str] = ()):
        """Encode the shared state once; returns ``(cache, token_count, seconds)``."""

        cache, n_tokens, _delta, elapsed = self._encode_state_full(state, images)
        return cache, n_tokens, elapsed

    @staticmethod
    def _state_key(state: str, images: Sequence[str] = ()) -> tuple:
        """Identity of a state in the LRU, so a caller can ask whether it is cached."""

        return (state.strip(), tuple(str(path) for path in images))

    @torch.no_grad()
    def _encode_state_full(self, state: str, images: Sequence[str] = ()):
        """As :meth:`encode_state`, but also hands back the state's mRoPE delta."""

        key = self._state_key(state, images)
        cached = self._state_cache.get(key)
        if cached is not None:
            cache, n_tokens, _delta = cached
            return cache, n_tokens, _delta, 0.0

        payload = self._state_payload(state, images)
        n_tokens = int(payload["input_ids"].shape[1])
        # The backbone keeps ``rope_deltas`` as global state: after an image prefill the
        # mRoPE positions of everything that follows are shifted by the number of vision
        # positions. Reset it so a text state never inherits an image state's delta, and
        # read back the delta this state really produced.
        backbone = self.model.model
        backbone.rope_deltas = None
        t0 = time.perf_counter()
        out = backbone(**payload, use_cache=True, return_dict=True)
        cache = out.past_key_values
        elapsed = (time.perf_counter() - t0) * 1000
        delta = getattr(out, "rope_deltas", None)
        if delta is not None:
            delta = delta.detach().reshape(-1)[:1].clone()
            backbone.rope_deltas = None
        self._state_cache.put(key, (cache, n_tokens, delta))
        return cache, n_tokens, delta, elapsed

    # -- branch execution ------------------------------------------------------
    def _chunk_branches(self, branches: Sequence[TokenizedBranch]):
        """Pack the independent branches into batches.

        Two limits: the configured batch size, and a token budget so that long
        branches cannot blow up the padding of a large batch.
        """

        chunks: list[list[TokenizedBranch]] = []
        current: list[TokenizedBranch] = []
        longest = 0
        for branch in branches:
            longest = max(longest, len(branch))
            too_many = len(current) + 1 > self.config.branch_batch_size
            too_long = bool(current) and (len(current) + 1) * longest > self.config.max_batch_tokens
            if current and (too_many or too_long):
                chunks.append(current)
                current, longest = [], len(branch)
            current.append(branch)
        if current:
            chunks.append(current)
        return chunks

    def _expand_prefix(self, cache, n: int):
        """Give every branch its own view of the shared prefix ("prefix KV cache
        with separate causal suffixes"), without disturbing the cached state."""

        branch_cache = copy.deepcopy(cache)
        branch_cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.device))
        return branch_cache

    @torch.no_grad()
    def _run_branches(
        self,
        prefix_cache,
        state_ids: Sequence[int],
        branches: Sequence[TokenizedBranch],
        rope_delta=None,
    ):
        n = len(branches)
        state_len = len(state_ids)
        full = prefix_cache is None
        length = (state_len if full else 0) + max(len(b) for b in branches)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((n, length), pad_id, dtype=torch.long, device=self.device)
        for i, branch in enumerate(branches):
            if full:
                input_ids[i, :state_len] = torch.tensor(state_ids, device=self.device)
            start = state_len if full else 0
            input_ids[i, start : start + len(branch)] = torch.tensor(branch.ids, device=self.device)

        if full:  # reference path: every branch carries its own copy of the state
            # ``length`` already contains the state, so the mask must be that wide too.
            mask = torch.zeros((n, length), dtype=torch.long, device=self.device)
            mask[:, :state_len] = 1
            for i, branch in enumerate(branches):
                mask[i, state_len : state_len + len(branch)] = 1
            past = None
            position_ids = None
        else:
            mask = torch.ones((n, state_len + length), dtype=torch.long, device=self.device)
            for i, branch in enumerate(branches):
                mask[i, state_len + len(branch) :] = 0
            past = self._expand_prefix(prefix_cache, n)
            # With a cache the backbone would rebuild positions from the (padded)
            # attention mask, which is one token too wide and breaks mRoPE. Give it the
            # incremental positions itself: each branch token sits at
            # ``cache_length + i``, shifted by the state's mRoPE delta.
            positions = torch.arange(
                state_len, state_len + length, dtype=torch.long, device=self.device
            )
            position_ids = positions.view(1, 1, -1).expand(3, n, -1)
            if rope_delta is not None:
                position_ids = position_ids + rope_delta.to(self.device).view(1, 1, 1)

        out = self.model.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        offset = state_len if full else 0
        decision = torch.stack([hidden[i, offset + b.decision_index] for i, b in enumerate(branches)])
        option_hidden = [
            hidden[i, [offset + j for j in b.option_end_indices], :] for i, b in enumerate(branches)
        ]
        return decision, option_hidden

    def _row_batch_inputs(self, prepared_rows):
        """One padded batch of ``[state + branch]`` rows, images included.

        Rows without images fall back to the tokenizer path, so text-only records keep
        the exact behaviour they had before multimodal support existed.
        """

        device = self.device
        pad_id = self.tokenizer.pad_token_id or 0
        per_row = []
        for payload, branch in prepared_rows:
            per_row.append((payload["input_ids"][0].tolist(), payload, branch))
        length = max(len(ids) + len(branch) for ids, _payload, branch in per_row)
        input_ids = torch.full((len(per_row), length), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(per_row), length), dtype=torch.long, device=device)
        mm_types = torch.zeros((len(per_row), length), dtype=torch.long, device=device)
        pixels, grids = [], []
        for i, (state_ids, payload, branch) in enumerate(per_row):
            start = len(state_ids)
            input_ids[i, :start] = torch.tensor(state_ids, device=device)
            input_ids[i, start : start + len(branch)] = torch.tensor(branch.ids, device=device)
            mask[i, : start + len(branch)] = 1
            if "mm_token_type_ids" in payload:
                mm_types[i, :start] = payload["mm_token_type_ids"][0]
            if "pixel_values" in payload:
                pixels.append(payload["pixel_values"])
                grids.append(payload["image_grid_thw"])
        inputs = {"input_ids": input_ids, "attention_mask": mask}
        if pixels:
            inputs["pixel_values"] = torch.cat(pixels, dim=0)
            inputs["image_grid_thw"] = torch.cat(grids, dim=0)
            inputs["mm_token_type_ids"] = mm_types
        out = self.model.model(**inputs, return_dict=True, use_cache=False)
        hidden = out.last_hidden_state
        decision = torch.stack(
            [hidden[i, len(state_ids) + branch.decision_index] for i, (state_ids, _p, branch) in enumerate(per_row)]
        )
        option_hidden = [
            hidden[i, [len(state_ids) + j for j in branch.option_end_indices], :]
            for i, (state_ids, _p, branch) in enumerate(per_row)
        ]
        return decision, option_hidden, per_row

    def _score(
        self,
        readouts: Sequence[Readout],
        decision,
        option_hidden,
        branches: Sequence[TokenizedBranch],
        questions,
        continuations=None,
    ):
        """Score a chunk whose rows may not share a readout.

        Under ``readout="auto"`` the widest question in a request is not the only thing
        that decides how rows are scored any more: each row is scored by the readout that
        fits *its own* option count. Rows are therefore scored in groups and scattered
        back into one padded ``(B, K_max)`` pair, which is what every readout already
        returns (``-inf`` in the columns past a row's own option count), so callers see
        the same shape they always did.
        """

        groups: "OrderedDict[str, list[int]]" = OrderedDict()
        for position, readout in enumerate(readouts):
            groups.setdefault(readout.name, []).append(position)
        width = max(len(branch.labels) for branch in branches)
        device = decision.device
        log_probs = torch.full((len(branches), width), float("-inf"), device=device)
        raw = torch.full((len(branches), width), float("-inf"), device=device)
        for positions in groups.values():
            readout = readouts[positions[0]]
            index = torch.tensor(positions, dtype=torch.long, device=device)
            scored = readout.score(
                decision[index],
                labels=[list(branches[p].labels) for p in positions],
                option_hidden=(
                    [option_hidden[p] for p in positions]
                    if readout.uses_option_states
                    else None
                ),
                option_ids=[question_option_ids(questions[p]) for p in positions],
                option_texts=[[o.description for o in questions[p].options] for p in positions],
                continuation_logprobs=(
                    [continuations[p] for p in positions]
                    if continuations is not None
                    else None
                ),
            )
            log_probs[index, : scored.log_probs.shape[1]] = scored.log_probs
            raw[index, : scored.raw_logits.shape[1]] = scored.raw_logits
        from .readout import ReadoutLogits

        return ReadoutLogits(log_probs=log_probs, raw_logits=raw)

    #: How many option continuations share one forward pass. Each row keeps a
    #: ``width x vocab`` logit block alive, so this bounds transient memory rather than
    #: throughput: 16 rows of a 4-token name is well under a hundred megabytes.
    continuation_options_per_forward = 16

    def _continuation_ids(self, branch: TokenizedBranch, text: str, prefix: str = " ") -> list[int]:
        """The token sequence the model would emit to continue the branch with ``text``.

        ``prefix`` is what the readout's cue leaves before the option - a space for a cue
        that ends a sentence, nothing after an opening quote - so the target is the
        tokenisation of ``prefix + text``: a self-contained sequence, scored token by
        token against the branch. Re-tokenising ``branch.text + text`` and slicing off
        the branch instead looks more direct and is wrong: the tokeniser is free to merge
        across that join (Qwen merges ``": "`` into one token), which re-segments the
        branch's last token and shifts every score by one position. Measured 0.00 that way against 0.38 with this.
        """

        ids = [
            int(token)
            for token in self.tokenizer.encode(prefix + text, add_special_tokens=False)
        ]
        return ids or [int(self.tokenizer.pad_token_id or 0)]

    @torch.no_grad()
    def _option_continuation_logprobs(
        self,
        state: str,
        images: Sequence[str],
        question: Question,
        branch: TokenizedBranch,
        text_prefix: str = " ",
    ) -> torch.Tensor:
        """Mean log-probability of every option's own text, teacher-forced after the cue.

        The verbalizer reads an option's *name*, and a name is usually several tokens
        ("what are your hobbies") while one decision position offers one distribution
        over the vocabulary - which cannot tell two options apart beyond their first
        word. So instead each option is appended to the branch and run through the model,
        its tokens scored in the context of the ones before them.

        The branch is prefilled once with a KV cache and every continuation rides on it,
        which is what keeps this affordable: the extra forwards carry a few tokens each
        instead of a whole 151-option branch each.
        """

        device = self.device
        payload = self._state_payload(state, tuple(images))
        state_ids = [int(token) for token in payload["input_ids"][0].tolist()]
        prompt_ids = state_ids + list(branch.ids)
        rows = [
            self._continuation_ids(branch, option.description, text_prefix)
            for option in question.options
        ]

        backbone = self.model.model
        head = self.model.get_output_embeddings()
        pad_id = int(self.tokenizer.pad_token_id or 0)
        prompt = torch.tensor([prompt_ids], device=device)
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in payload:
                inputs[key] = payload[key]
        backbone.rope_deltas = None
        prefill = backbone(**inputs, use_cache=True, return_dict=True)
        cache = prefill.past_key_values
        rope_delta = getattr(prefill, "rope_deltas", None)
        if rope_delta is not None:
            rope_delta = rope_delta.detach().reshape(-1)[:1].clone()
            backbone.rope_deltas = None

        length = len(prompt_ids)
        n = len(rows)
        width = max(len(ids) for ids in rows)
        input_ids = torch.full((n, width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((n, width), dtype=torch.long, device=device)
        for i, ids in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
            mask[i, : len(ids)] = 1
        positions = torch.arange(length, length + width, device=device)
        positions = positions.view(1, 1, -1).expand(3, n, -1)
        if rope_delta is not None:
            positions = positions + rope_delta.to(device).view(1, 1, 1)
        attention = torch.cat(
            [torch.ones((n, length), dtype=torch.long, device=device), mask], dim=1
        )
        # One expanded view of the cached branch serves every option, and the continuations
        # are a few tokens each, so one forward covers them all. Only the head is chunked:
        # its output is ``rows x width x vocabulary``, which is the one block big enough to
        # matter (151 options x 4 tokens x 248k vocabulary would be ~600 MB at once).
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=positions,
            past_key_values=self._expand_prefix(cache, n),
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state.to(head.weight.dtype)
        scores = torch.zeros(n, dtype=torch.float32, device=device)
        for start in range(0, n, self.continuation_options_per_forward):
            stop = start + self.continuation_options_per_forward
            log_probs = torch.log_softmax(head(hidden[start:stop]).float(), dim=-1)
            gathered = log_probs.gather(-1, input_ids[start:stop].unsqueeze(-1)).squeeze(-1)
            block = mask[start:stop]
            scores[start:stop] = (gathered * block).sum(-1) / block.sum(-1).clamp(min=1)
        return scores.cpu()

    def _probabilities(self, logits):
        """Softmax over the row's slots, after the fitted readout temperature."""

        return torch.softmax(logits.raw_logits / self.readout_temperature, dim=-1)

    # -- public API ------------------------------------------------------------
    @torch.no_grad()
    def decide_batch(
        self,
        requests: Sequence[tuple],
        *,
        return_hidden: bool = False,
    ) -> list[JevResponse]:
        """Answer many independent requests in shared forward passes.

        A request is ``(state, questions)`` or ``(state, questions, images)``. Each keeps
        its own state and its own isolated branches; the rows of one forward pass simply
        belong to different requests. Requests whose state carries several questions are
        answered through the cached per-state path, so one image encode serves them all;
        the rest share row-batched passes. Both paths give the same numbers as calling
        :meth:`decide` once per request.
        """

        normalised: list[tuple[str, Any, tuple]] = []
        for request in requests:
            if len(request) == 3:
                state, questions, images = request
            else:
                state, questions = request
                images = ()
            normalised.append((state, questions, tuple(images or ())))
        if not normalised:
            return []

        # A state with several questions (or a long text state) is cheaper through the
        # cached per-state path: the state, images included, is encoded once.
        cached_indices = {
            index
            for index, (_state, questions, images) in enumerate(normalised)
            if images and len(questions) > 1
        }
        for index, (state, questions, images) in enumerate(normalised):
            if index in cached_indices or images:
                continue
            if self.prefers_shared_prefill([(state, questions)]):
                cached_indices.add(index)
        if cached_indices:
            responses: list[JevResponse | None] = [None] * len(normalised)
            for index in sorted(cached_indices):
                state, questions, images = normalised[index]
                responses[index] = self.decide(
                    state,
                    questions,
                    images=images or None,
                    share_state=True,
                    return_hidden=return_hidden,
                )
            row_indices = [index for index in range(len(normalised)) if index not in cached_indices]
            if row_indices:
                row_responses = self._decide_rows(
                    [normalised[index] for index in row_indices], return_hidden=return_hidden
                )
                for index, response in zip(row_indices, row_responses):
                    responses[index] = response
            return [response for response in responses if response is not None]
        return self._decide_rows(normalised, return_hidden=return_hidden)

    @torch.no_grad()
    def _decide_rows(self, requests: Sequence[tuple], *, return_hidden: bool = False) -> list[JevResponse]:
        """Row-batched path: several requests share one padded forward pass."""

        t_start = time.perf_counter()
        prepared = []
        for state, questions, images in requests:
            question_list = [build_question(qid, spec) for qid, spec in questions.items()]
            readouts = self.readouts_for(question_list)
            for question, readout in zip(question_list, readouts):
                self._validate(question, readout)
            payload = self._state_payload(state, images)
            state_len = int(payload["input_ids"].shape[1])
            branches = [
                tokenize_branch(
                    self.tokenizer, render_branch(q, readout.reply_style, readout.decision_cue)
                )
                for q, readout in zip(question_list, readouts)
            ]
            self.check_limits(state_len, branches)
            prepared.append((state, images, payload, branches, question_list, readouts))
        if not prepared:
            return []

        rows: list[tuple[int, int]] = [
            (request_index, branch_index)
            for request_index, (_state, _images, _payload, branches, _questions, _readout) in enumerate(prepared)
            for branch_index in range(len(branches))
        ]
        # Length bucketing keeps the padding inside a chunk small.
        rows.sort(
            key=lambda pair: int(prepared[pair[0]][2]["input_ids"].shape[1]) + len(prepared[pair[0]][3][pair[1]])
        )

        scored: dict[tuple[int, int], Any] = {}
        hidden_dump: dict[str, Any] = {}
        for chunk in self._chunk_rows(rows, prepared):
            decision, option_hidden, _rows = self._row_batch_inputs(
                [(prepared[ri][2], prepared[ri][3][bi]) for ri, bi in chunk]
            )
            # A readout that reads option *names* needs the branch continued through the
            # model once per option; do that here, where the state and the branch are
            # both still in hand.
            continuations: dict[tuple[int, int], torch.Tensor] = {}
            for ri, bi in chunk:
                if prepared[ri][5][bi].uses_continuations:
                    continuations[(ri, bi)] = self._option_continuation_logprobs(
                        prepared[ri][0],
                        prepared[ri][1],
                        prepared[ri][4][bi],
                        prepared[ri][3][bi],
                        prepared[ri][5][bi].continuation_prefix,
                    )
            # "auto" can put two readouts in one chunk, so score each group of rows with
            # its own readout over the same hidden states.
            groups: "OrderedDict[str, list[int]]" = OrderedDict()
            for position, (ri, bi) in enumerate(chunk):
                groups.setdefault(prepared[ri][5][bi].name, []).append(position)
            for positions in groups.values():
                readout = prepared[chunk[positions[0]][0]][5][chunk[positions[0]][1]]
                selected = torch.tensor(positions, dtype=torch.long, device=decision.device)
                logits = readout.score(
                    decision[selected],
                    labels=[list(prepared[chunk[p][0]][3][chunk[p][1]].labels) for p in positions],
                    option_hidden=(
                        [option_hidden[p] for p in positions]
                        if readout.uses_option_states
                        else None
                    ),
                    option_ids=[
                        question_option_ids(prepared[chunk[p][0]][4][chunk[p][1]])
                        for p in positions
                    ],
                    option_texts=[
                        [o.description for o in prepared[chunk[p][0]][4][chunk[p][1]].options]
                        for p in positions
                    ],
                    continuation_logprobs=(
                        [continuations[(chunk[p][0], chunk[p][1])] for p in positions]
                        if readout.uses_continuations
                        else None
                    ),
                )
                probabilities = torch.softmax(
                    logits.raw_logits / self.readout_temperature, dim=-1
                )
                for offset, position in enumerate(positions):
                    ri, bi = chunk[position]
                    scored[(ri, bi)] = (
                        probabilities[offset],
                        logits.raw_logits[offset],
                        decision[position],
                        option_hidden[position],
                    )

        branch_ms = (time.perf_counter() - t_start) * 1000
        responses: list[JevResponse] = []
        for ri, (state, images, payload, branches, question_list, readout) in enumerate(prepared):
            state_tokens = int(payload["input_ids"].shape[1])
            visual_tokens = int((payload.get("mm_token_type_ids") == 1).sum()) if "mm_token_type_ids" in payload else 0
            decisions: "OrderedDict[str, QuestionDecision]" = OrderedDict()
            for bi, question in enumerate(question_list):
                row_probabilities, raw_logits, _decision, _options = scored[(ri, bi)]
                k = len(branches[bi].labels)
                row = row_probabilities[:k]
                probs = OrderedDict(
                    (option.key, float(p)) for option, p in zip(question.options, row.tolist())
                )
                answer_key = max(probs, key=probs.get)
                decisions[question.id] = QuestionDecision(
                    question_id=question.id,
                    type=question.type.value,
                    probabilities=probs,
                    answer=answer_key,
                    confidence=summarise(question.type.value, probs),
                    score_value=(
                        question.expected_value(probs)
                        if question.type is QuestionType.SCORE
                        else None
                    ),
                    raw_logits=[float(x) for x in raw_logits[:k].tolist()],
                    slots={
                        label: option.key
                        for label, option in zip(branches[bi].labels, question.options)
                    },
                    bool_true_key=(
                        bool_true_key(question.options)
                        if question.type is QuestionType.BOOL
                        else None
                    ),
                )
                if return_hidden:
                    hidden_dump[question.id] = {
                        "decision": _decision.float().cpu(),
                        "options": _options.float().cpu(),
                        "labels": list(branches[bi].labels),
                        "option_keys": [o.key for o in question.options],
                    }
            question_tokens = sum(len(branch) for branch in branches)
            usage = {
                "state_tokens": state_tokens,
                "visual_tokens": visual_tokens,
                "text_state_tokens": state_tokens - visual_tokens,
                "question_tokens": question_tokens,
                "input_tokens": state_tokens + question_tokens,
                "output_tokens": billing_output_tokens(
                    [question.id for question in question_list], self.tokenizer
                ),
                "questions": len(question_list),
                "branches": len(question_list),
                "requests": 1,
            }
            timing = {
                "prefill_ms": 0.0,
                "branch_ms": round(branch_ms / max(len(prepared), 1), 2),
                "total_ms": round(branch_ms / max(len(prepared), 1), 2),
                "state_cache_hit": False,
                "shared_state": False,
                "readout": self._readout_label(prepared[ri][5]),
                "batched": True,
            }
            responses.append(
                JevResponse(
                    decisions=decisions,
                    usage=usage,
                    timing=timing,
                    hidden=hidden_dump if return_hidden else None,
                )
            )
        return responses

    def prefers_shared_prefill(self, requests: Sequence[tuple[str, Any]]) -> bool:
        """Whether the cached per-state path beats row-batching for these requests.

        Row-batching pays off when the states are short: one forward pass covers every
        branch of every request, and the per-call overhead is paid once. It loses when a
        state is long, because each branch carries its own copy of it - so the decision
        is made on ``state tokens x branches`` per request.
        """

        budget = self.config.state_reuse_tokens
        if not budget:
            return False
        for state, questions in requests:
            branches = len(questions)
            if branches < 2:
                continue
            state_tokens = len(
                self.tokenizer(render_state(state), add_special_tokens=False)["input_ids"]
            )
            if state_tokens * branches >= budget:
                return True
        return False

    def _chunk_rows(
        self,
        rows: Sequence[tuple[int, int]],
        prepared: Sequence[tuple],
    ) -> list[list[tuple[int, int]]]:
        """Group rows into forward passes under a token budget."""

        chunks: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        longest = 0
        for pair in rows:
            ri, bi = pair
            length = int(prepared[ri][2]["input_ids"].shape[1]) + len(prepared[ri][3][bi])
            longest = max(longest, length)
            too_many = len(current) + 1 > self.config.branch_batch_size
            too_long = bool(current) and (len(current) + 1) * longest > self.config.max_batch_tokens
            if current and (too_many or too_long):
                chunks.append(current)
                current, longest = [], length
            current.append(pair)
        if current:
            chunks.append(current)
        return chunks

    def decide(
        self,
        state: str,
        questions: Mapping[str, Mapping[str, Any] | Question] | Iterable[Question],
        *,
        images: Sequence[str] | None = None,
        share_state: bool | None = None,
        return_hidden: bool = False,
    ) -> JevResponse:
        t_start = time.perf_counter()
        if isinstance(questions, Mapping):
            question_list = [build_question(qid, spec) for qid, spec in questions.items()]
        else:
            question_list = [build_question(q.id, q) for q in questions]

        readouts = self.readouts_for(question_list)
        for question, readout in zip(question_list, readouts):
            self._validate(question, readout)

        rendered = [
            render_branch(q, readout.reply_style, readout.decision_cue)
            for q, readout in zip(question_list, readouts)
        ]
        branches = [tokenize_branch(self.tokenizer, r) for r in rendered]
        images = tuple(images or ())
        continuations = (
            [
                self._option_continuation_logprobs(
                    state, images, question, branch, readout.continuation_prefix
                )
                for question, branch, readout in zip(question_list, branches, readouts)
            ]
            if any(readout.uses_continuations for readout in readouts)
            else None
        )
        payload = self._state_payload(state, images)
        state_ids = payload["input_ids"][0].tolist()
        state_len = len(state_ids)
        self.check_limits(state_len, branches)

        if images:
            # The image lives in the state prefix, so it has to be prefilled once: the
            # reference path (each branch carries its own copy of the state) would pay
            # for the vision tower again for every question.
            share_state = True
        if share_state is None:
            share_state = self.config.share_state
            if share_state and len(question_list) < 2 and not images:
                # Sharing means "prefill the state once, then run each branch against the
                # cache". A request with a single branch has nothing to share *within*,
                # so a state that is not cached yet costs two forwards (state, then
                # branch) where one combined pass would do - measured 700 ms against
                # cheaper on short states. A state that *is* cached is the
                # opposite: the branch alone is one forward and the state is free, so
                # the shared path stays.
                share_state = self._state_cache.get(self._state_key(state, images)) is not None
        if share_state:
            prefix_cache, _, rope_delta, prefill_ms = self._encode_state_full(state, images)
        else:
            prefix_cache = None
            rope_delta = None
            prefill_ms = 0.0
        cache_hit = share_state and prefill_ms == 0.0

        t_branch = time.perf_counter()
        decisions: "OrderedDict[str, QuestionDecision]" = OrderedDict()
        hidden_dump: dict[str, Any] = {}
        index = 0
        requests_made = 0
        for chunk in self._chunk_branches(branches):
            requests_made += 1
            decision, option_hidden = self._run_branches(
                prefix_cache, state_ids, chunk, rope_delta=rope_delta
            )
            logits = self._score(
                readouts[index : index + len(chunk)],
                decision,
                option_hidden,
                chunk,
                question_list[index : index + len(chunk)],
                continuations[index : index + len(chunk)] if continuations is not None else None,
            )
            probabilities = self._probabilities(logits)
            for i, (question, branch) in enumerate(
                zip(question_list[index : index + len(chunk)], chunk)
            ):
                row = probabilities[i, : len(branch.labels)]
                probs = OrderedDict(
                    (option.key, float(p)) for option, p in zip(question.options, row.tolist())
                )
                answer_key = max(probs, key=probs.get)
                confidence = summarise(question.type.value, probs)
                score_value = (
                    question.expected_value(probs) if question.type is QuestionType.SCORE else None
                )
                decisions[question.id] = QuestionDecision(
                    question_id=question.id,
                    type=question.type.value,
                    probabilities=probs,
                    answer=answer_key,
                    confidence=confidence,
                    score_value=score_value,
                    raw_logits=[float(x) for x in logits.raw_logits[i, : len(branch.labels)].tolist()],
                    slots={
                        label: option.key
                        for label, option in zip(branch.labels, question.options)
                    },
                    bool_true_key=(
                        bool_true_key(question.options)
                        if question.type is QuestionType.BOOL
                        else None
                    ),
                )
                if return_hidden:
                    hidden_dump[question.id] = {
                        "decision": decision[i].float().cpu(),
                        "options": option_hidden[i].float().cpu(),
                        "labels": list(branch.labels),
                        "option_keys": [o.key for o in question.options],
                    }
            index += len(chunk)
        branch_ms = (time.perf_counter() - t_branch) * 1000

        question_tokens = sum(len(b) for b in branches)
        visual_tokens = int((payload["mm_token_type_ids"] == 1).sum()) if "mm_token_type_ids" in payload else 0
        usage = {
            "state_tokens": state_len,
            "visual_tokens": visual_tokens,
            "text_state_tokens": state_len - visual_tokens,
            "question_tokens": question_tokens,
            "input_tokens": state_len + question_tokens,
            "output_tokens": billing_output_tokens([q.id for q in question_list], self.tokenizer),
            "questions": len(question_list),
            "branches": len(question_list),
            "requests": requests_made,
        }
        timing = {
            "prefill_ms": round(prefill_ms, 2),
            "branch_ms": round(branch_ms, 2),
            "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
            "state_cache_hit": bool(cache_hit),
            "shared_state": bool(share_state),
            "readout": self._readout_label(readouts),
        }
        return JevResponse(
            decisions=decisions, usage=usage, timing=timing, hidden=hidden_dump or None
        )

    @staticmethod
    def _readout_label(readouts: Sequence[Readout]) -> str:
        """What to report as ``timing["readout"]`` for a request or a row.

        With ``readout="auto"`` a request can legitimately use more than one readout, so
        the label names all of them instead of quietly reporting the first.
        """

        names = sorted({readout.name for readout in readouts})
        return names[0] if len(names) == 1 else "auto:" + "/".join(names)

    # -- validation ------------------------------------------------------------
    def _validate(self, question: Question, readout: Readout | None = None) -> None:
        """One question against the readout that will answer it.

        ``max_options`` means a different thing per readout (the reserved letters are a
        tokenisation fact, the API cap is a contract, a slot head's is a row budget), so
        the check is always made against the readout that is actually going to score this
        question rather than against the engine's default.
        """

        limits = self.limits
        readout = readout or self.readout
        count = len(question.options)
        if count > limits.max_options:
            raise ValueError(
                f"question {question.id!r} has {count} options; the API accepts at most "
                f"{limits.max_options}"
            )
        if count > readout.max_options:
            raise ValueError(
                f"question {question.id!r} has {count} options but the {readout.name!r} "
                f"readout supports at most {readout.max_options} "
                f"({readout.limit_reason}); use readout='auto' (which picks a readout per "
                f"question) or readout='slot_head' for a trained head"
            )
        if count < 1:
            raise ValueError(f"question {question.id!r} has no options")

    def check_limits(self, state_len: int, branches: Sequence[TokenizedBranch]) -> None:
        branch_lens = [len(b) for b in branches]
        if state_len + max(branch_lens) > self.limits.max_branch_tokens:
            raise ValueError(
                f"branch needs {state_len + max(branch_lens)} tokens > "
                f"{self.limits.max_branch_tokens}"
            )
        total = state_len + sum(branch_lens)
        if total > self.limits.max_request_tokens:
            raise ValueError(
                f"request needs {total} tokens > {self.limits.max_request_tokens}"
            )

    def account(self, state: str, questions: Mapping[str, Any], images: Sequence[str] | None = None) -> dict[str, int]:
        """Token accounting before running anything (the request limits of section 2)."""

        question_list = [build_question(qid, spec) for qid, spec in questions.items()]
        readouts = self.readouts_for(question_list)
        payload = self._state_payload(state, tuple(images or ()))
        state_len = int(payload["input_ids"].shape[1])
        visual_tokens = int((payload["mm_token_type_ids"] == 1).sum()) if "mm_token_type_ids" in payload else 0
        branch_lens = [
            len(
                tokenize_branch(
                    self.tokenizer, render_branch(q, readout.reply_style, readout.decision_cue)
                ).ids
            )
            for q, readout in zip(question_list, readouts)
        ]
        per_branch = [state_len + x for x in branch_lens]
        return {
            "state_tokens": state_len,
            "visual_tokens": visual_tokens,
            "text_state_tokens": state_len - visual_tokens,
            "question_tokens": sum(branch_lens),
            "request_tokens": state_len + sum(branch_lens),
            "max_branch_tokens": max(per_branch) if per_branch else state_len,
            "output_tokens": billing_output_tokens([q.id for q in question_list], self.tokenizer),
        }

    @staticmethod
    def assert_within_limits(accounting: Mapping[str, int], limits: JevLimits | None = None) -> None:
        """Check a pre-computed accounting dict against the API limits."""

        limits = limits or JevLimits()
        if accounting["max_branch_tokens"] > limits.max_branch_tokens:
            raise ValueError(
                f"branch needs {accounting['max_branch_tokens']} tokens > {limits.max_branch_tokens}"
            )
        if accounting["request_tokens"] > limits.max_request_tokens:
            raise ValueError(
                f"request needs {accounting['request_tokens']} tokens > {limits.max_request_tokens}"
            )
