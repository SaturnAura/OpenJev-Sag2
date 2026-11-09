"""Request limits and defaults.

Validation, token accounting and cache sizing all use these numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: Backbone checkpoint. Nothing in this project depends on which one it is: the engine
#: only asks for hidden states, so any causal (or multimodal) transformer works. The
#: default is a *relative* directory next to the checkout - put the model there, pass
#: ``--model``, or set ``QWENJEV_MODEL`` to an absolute path or a Hub repo id.
DEFAULT_MODEL_PATH = "qwen3.5-9B"

#: The Laya baseline (an external model used for comparison) resolves the same way.
DEFAULT_LAYA_PATH = "laya"


def default_model_path() -> str:
    """``$QWENJEV_MODEL`` if set, else the sibling directory ``qwen3.5-9B``."""

    return os.environ.get("QWENJEV_MODEL") or DEFAULT_MODEL_PATH


def default_laya_path() -> str:
    """``$QWENJEV_LAYA`` (or ``$LAYA_PATH``) if set, else the sibling directory ``laya``."""

    return os.environ.get("QWENJEV_LAYA") or os.environ.get("LAYA_PATH") or DEFAULT_LAYA_PATH


@dataclass(frozen=True)
class JevLimits:
    """Hard limits the API enforces."""

    #: "The API accepts at most 255 options (2**8 - 1)".
    max_options: int = 255
    #: Maximum input: each branch (the state plus one question) is capped here, so the
    #: state may be 64k long. Override it per run with ``--max-input-tokens`` (on
    #: ``train.py`` / ``test.py`` / ``demo.py`` / ``serve.py``), or build the limits with
    #: :func:`limits_for`. The backbone itself allows 262,144 positions.
    max_branch_tokens: int = 64_000
    #: The whole request is capped at twice that: the state is counted once, so a 64k
    #: state still carries a fair number of questions before the sum hits the cap.
    max_request_tokens: int = 128_000
    #: Billing figure: "4 shared tokens, plus 15 per answer, plus the token length of
    #: each question's identifier".
    billing_base_tokens: int = 4
    billing_tokens_per_answer: int = 15
    #: Number of option slots the dedicated readout head is sized for. Every trained
    #: label space gets a private block, so this has to hold the 27 reserved rows (26
    #: letters + one neutral) plus one row per option of every label space in the
    #: training mix. It is a row budget, *not* a cap on how many options a question may
    #: have: a question's options are scored by its own block, which grows with the
    #: label space itself.
    max_slots: int = 1024


def limits_for(max_input_tokens: int | None = None) -> JevLimits:
    """The request limits, with the maximum input overridable in one number.

    ``max_input_tokens`` is the branch cap - the state plus its longest question - which
    is what "how long can the input be" means here. The request cap follows at twice that,
    because the state is counted once no matter how many questions ride on it.
    """

    if not max_input_tokens:
        return JevLimits()
    branch = int(max_input_tokens)
    return JevLimits(max_branch_tokens=branch, max_request_tokens=branch * 2)


@dataclass
class QwenJevConfig:
    """Runtime configuration for :class:`qwenjev.engine.QwenJevLite`."""

    model_path: str = field(default_factory=default_model_path)
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    limits: JevLimits = field(default_factory=JevLimits)
    #: Load the transformer's text decoder directly. The OpenJev decision path is
    #: text-only, so this avoids allocating the multimodal vision tower for retrieval.
    text_only: bool = True

    #: Readout used to turn hidden states into probabilities.
    #: ``reserved_label`` = probability mass the LM head puts on reserved label tokens
    #: (works out of the box, K <= 26). ``slot_head`` = a dedicated K-slot linear head
    #: (a dedicated final-position head, trained by RLCD). ``pointer`` = listwise
    #: pointer scorer over per-option hidden states. ``verbalizer`` = the LM head's
    #: probability of each option's own text (zero-shot, any K). ``auto`` (the default)
    #: picks ``reserved_label`` when a request's widest question fits in the 26 reserved
    #: letters and ``verbalizer`` when it does not, so the option count a caller sends
    #: is the only thing that decides which readout runs.
    readout: str = "auto"
    #: Optional checkpoint produced by :mod:`qwenjev.rlcd`.
    readout_checkpoint: str | None = None

    #: Share one state encoding across all branches of a request.
    share_state: bool = True
    #: Keep encoded states in an LRU so identical states are not re-encoded.
    state_cache_size: int = 4
    #: Branches are evaluated in batches of this size.
    branch_batch_size: int = 64
    #: A batch also stops growing once ``branches x longest_branch`` exceeds this.
    max_batch_tokens: int = 8192
    #: The row-batched path (:meth:`QwenJevLite.decide_batch`) re-encodes every state
    #: once per branch. When ``state_tokens x branches`` reaches this many tokens, it is
    #: cheaper to prefill each state once and run its branches from the cache instead -
    #: measured 12x on a 6,000-character state with 64 questions (0.4 s against 4.7 s
    #: per request). Set to 0 to always row-batch.
    state_reuse_tokens: int = 800

    #: How many images one state may carry. 0 turns the vision path off entirely.
    max_images: int = 4
    #: Images are downscaled by the processor so ``width * height`` never exceeds this
    #: before patching (512x512 -> 256 visual tokens, 768x768 -> 576, 1024x1024 -> 1024).
    #: It maps onto the processor's ``size.longest_edge``; ``max_pixels`` is ignored by
    #: the transformers 5 image processors, so do not pass that keyword instead.
    image_max_pixels: int = 768 * 768

    def __post_init__(self) -> None:
        if self.readout not in {"auto", "reserved_label", "verbalizer", "slot_head", "pointer"}:
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.branch_batch_size < 1:
            raise ValueError("branch_batch_size must be >= 1")
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")
        if self.state_reuse_tokens < 0:
            raise ValueError("state_reuse_tokens must be >= 0")
        if self.max_images < 0:
            raise ValueError("max_images must be >= 0")
        if self.image_max_pixels <= 0:
            raise ValueError("image_max_pixels must be > 0")
        if self.state_cache_size < 0:
            raise ValueError("state_cache_size must be >= 0")
