"""Typed decision questions, mirroring the documented Jev request shape.

The wire format for a request::

    {
        "state": "My payouts have failed three times. ...",
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {"payments": "...", "account": "...", "other": "..."}
            },
            "escalate": {
                "type": "bool",
                "instructions": "Does this message require urgent human attention?"
            }
        }
    }
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class QuestionType(str, Enum):
    CHOICE = "choice"
    BOOL = "bool"
    SCORE = "score"


#: Label tokens reserved for option slots: the 26 capitals, which this tokenizer holds
#: as single tokens in both bare and space-prefixed form, and which the model has read
#: as answer labels all its life.
#
#: These 26 are what the *zero-shot* ``reserved_label`` readout can score - it reads
#: the LM head's mass on one token per option. Nothing else in the project is capped
#: at 26: :func:`slot_label` names any number of options (``A..Z, AA, AB, ...``), and
#: a question with more options than this is answered by the ``slot_head`` readout
#: (trained rows, no token bottleneck) or by the ``verbalizer`` readout (zero-shot,
#: per-option text probability). See ``qwenjev.readout``.
SLOT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Number of options the reserved single-token letters cover. A question with more
#: options than this cannot be scored by ``ReservedLabelReadout``; the engine's
#: ``readout="auto"`` is what picks a readout that can.
MAX_LETTER_LABELS = len(SLOT_LETTERS)

#: Rows ``0 .. 25`` of a trained slot head stay the pretrained label rows, and row
#: ``SLOT_UNKNOWN_ROW`` is a neutral row for an option past the 26 reserved ones whose
#: label space the head never saw. Every trained label space gets its own block
#: *after* them, so training one label space can never overwrite the rows another
#: label space falls back on.
SLOT_UNKNOWN_ROW = len(SLOT_LETTERS)
SLOT_RESERVE = SLOT_UNKNOWN_ROW + 1

#: Separator used when an answer is turned into a dictionary key.
FAMILY_SEP = "\x1f"

#: How a choice question's options are laid out in the branch before it is rendered.
#:
#: ``"sorted"`` (the default) renders them in a canonical order - by answer key, and for a
#: boolean question with the "true" key first - so the prompt, the option labels and the
#: decision are a function of the option *set*, never of the order the caller happened to
#: list it in. This is what makes a joint-encoding engine permutation-invariant: this
#: architecture puts the options inside the prompt, so the caller's order would otherwise
#: be part of the input, and permuting the criteria would permute the tokens, move the
#: decision position's state, and flip answers (measured: see ``tools/permutation_check.py``).
#: Sibling architectures that encode every candidate on its own get the same property by
#: construction; sorting is the equivalent move for a single joint sequence.
#:
#: ``"declared"`` keeps the caller's order (set ``QWENJEV_OPTION_ORDER=declared``). Answers,
#: probabilities and metrics are always reported under the caller's own keys either way, and
#: each response's ``slots`` says which key each rendered letter stands for.
#: ``score`` questions are never reordered: their declaration order is the rubric's scale.
OPTION_ORDER = os.environ.get("QWENJEV_OPTION_ORDER", "sorted")

#: Key names that mean "the claim holds" on a boolean question. A caller may name the two
#: criteria anything, so the polarity is read from the names rather than assumed to be the
#: literal key ``yes`` - doing the latter inverted the boolean answer for anyone who wrote
#: ``{"true": ..., "false": ...}``.
TRUE_KEYS = {"true", "yes", "y", "1", "true_label", "positive", "entailment"}
FALSE_KEYS = {"false", "no", "n", "0", "false_label", "negative"}


def bool_true_key(options: "Sequence[Option] | Sequence[str]") -> str:
    """Which of a boolean question's two keys means "yes".

    Matched by name first (``yes`` / ``true`` / ``entailment`` / …), then by elimination
    against the negative names, then - for a caller who invented two arbitrary keys - by
    taking the first option, which is the order the options are offered in.
    """

    keys = [str(getattr(option, "key", option)) for option in options]
    if not keys:
        return "yes"
    for key in keys:
        if key.lower() in TRUE_KEYS:
            return key
    for key in keys:
        if key.lower() not in FALSE_KEYS:
            return key
    return keys[0]


def canonical_options(options: list["Option"], *, true_first: bool = False) -> list["Option"]:
    """The order a question's options are rendered in (see :data:`OPTION_ORDER`)."""

    if OPTION_ORDER != "sorted":
        return options
    if true_first:
        wanted = bool_true_key(options)
        return sorted(options, key=lambda option: (str(option.key) != wanted, str(option.key)))
    return sorted(options, key=lambda option: str(option.key))


def label_space(option_keys: "Sequence[str]") -> str:
    """Canonical name of a label space: its answer keys, sorted.

    Sorting is what makes the name independent of the order a dataset happens to list
    its options in - a dataset's train split and its test split can ship the same
    labels in different orders, and they must still be the same label space.
    """

    return FAMILY_SEP.join(sorted(str(key) for key in option_keys))


def option_ids(
    question_type: "QuestionType | str",
    option_keys: "Sequence[str]",
    *,
    question_id: str = "",
) -> list[str]:
    """One stable id per allowed answer: its question, its label space, then its key.

    The readout is ``z = W h + b`` over the *K allowed answers* of one question.
    Keying the rows by the question and the answer, rather than by their position in the
    row, is what lets one head serve every question without overwriting itself: every
    question owns a private block of rows, and *within* a block the row is chosen by the
    answer's key rather than by its position, so two files that list the same options in
    a different order still read the same row. The question id is part of the identity
    because two different questions can offer the same answers - six Jigsaw labels, 28
    GoEmotions emotions and five retrieval collections all ask ``yes``/``no`` - and
    pooling them into one pair of rows measured *worse than chance* on the collection
    with the fewest samples (scifact relevance 0.882 -> 0.212).
    """

    qtype = question_type.value if isinstance(question_type, QuestionType) else str(question_type)
    space = label_space(option_keys)
    return [
        f"{qtype}{FAMILY_SEP}{question_id}{FAMILY_SEP}{space}{FAMILY_SEP}{key}"
        for key in option_keys
    ]


def question_option_ids(question: "Question") -> list[str]:
    """Answer ids of a built :class:`Question`, in the order the options are shown."""

    return option_ids(
        question.type, [option.key for option in question.options], question_id=question.id
    )


def family_signature(question_type: "QuestionType | str", option_keys: "Sequence[str]") -> str:
    """Name of a whole label space (used for reporting and for dataset checks)."""

    qtype = question_type.value if isinstance(question_type, QuestionType) else str(question_type)
    return f"{qtype}{FAMILY_SEP}{label_space(option_keys)}"

def slot_label(index: int) -> str:
    """``A..Z, AA, AB, ...`` for arbitrarily many options.

    The reserved-label readout still only understands the 26 single-token letters; a
    larger label space is answered by the trained slot head, which indexes slots rather
    than tokens.
    """

    if index < 0:
        raise ValueError("slot index must be >= 0")
    name = ""
    position = index + 1
    while position:
        position, remainder = divmod(position - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def slot_index(label: str) -> int:
    """Inverse of :func:`slot_label`: ``"A"``/``"aa"``/``"EW"`` -> ``0``/``0``/``"150"``.

    A bare number is taken as an index as well (``"0"`` -> ``0``), which is what the
    dataset adapters need: a source file may write a correct answer as a position, as
    ``A``-style letters, or - past 26 options - as ``AA``-style letters.
    """

    text = str(label).strip()
    if not text:
        raise ValueError("empty slot label")
    if text.isdigit():
        return int(text)
    letters = text.upper()
    if not letters.isalpha() or not letters.isascii():
        raise ValueError(f"not a slot label: {label!r}")
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


@dataclass
class DecisionSample:
    """One supervised decision: (state, question, observed outcome).

    Used by the RLCD trainer and by the dataset adapters. ``spec`` carries the
    question definition when it varies per item (MMLU choices, Jigsaw label sets);
    otherwise the caller passes a shared ``question_specs`` mapping.
    """

    state: str
    question_id: str
    target: str
    spec: "dict | None" = None
    #: Image paths that belong to the shared state (empty for text-only records). The
    #: engine feeds them to the backbone's vision tower once per state, so every
    #: question about that state sees the same picture without re-encoding it.
    images: "list[str]" = field(default_factory=list)


@dataclass
class Question:
    """Base class for a single decision."""

    id: str
    instructions: str
    options: "list[Option]" = field(default_factory=list)
    type: QuestionType = QuestionType.CHOICE
    #: Per-branch text that belongs to this question rather than to the shared state
    #: (a candidate passage in a retrieval task). Our engine renders it into the branch;
    #: Laya's ``noul`` template wants it directly after the claim.
    context: str = ""

    def labels(self) -> list[str]:
        return [slot_label(i) for i in range(len(self.options))]


@dataclass
class Option:
    """One allowed answer: the caller's key plus the text shown to the model."""

    key: str
    description: str
    #: For ``score`` questions: the numeric value of this level.
    value: float | None = None


@dataclass
class ChoiceQuestion(Question):
    type: QuestionType = QuestionType.CHOICE


@dataclass
class BoolQuestion(Question):
    type: QuestionType = QuestionType.BOOL

    def __post_init__(self) -> None:
        if not self.options:
            self.options = [Option("yes", "Yes"), Option("no", "No")]


@dataclass
class ScoreQuestion(Question):
    """Ordered levels; the response carries a distribution and its expected value."""

    type: QuestionType = QuestionType.SCORE

    def expected_value(self, probabilities: "Mapping[str, float]") -> float:
        total = 0.0
        for i, opt in enumerate(self.options):
            value = opt.value if opt.value is not None else float(i)
            total += probabilities.get(opt.key, 0.0) * value
        return total


def _coerce_options(raw: Any, *, require_values: bool) -> list[Option]:
    if isinstance(raw, Mapping):
        items: Sequence[tuple[str, Any]] = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = []
        for i, entry in enumerate(raw):
            if isinstance(entry, Mapping):
                key = str(entry.get("key", entry.get("id", i)))
                desc = entry.get("description", entry.get("text", ""))
                value = entry.get("value")
            else:
                key, desc, value = str(entry), str(entry), None
            items.append((key, {"description": desc, "value": value}))
    else:
        raise TypeError("options/criteria must be a mapping or a sequence")

    options: list[Option] = []
    for key, payload in items:
        if isinstance(payload, Mapping):
            desc = str(payload.get("description", payload.get("text", "")))
            value = payload.get("value")
        else:
            desc, value = str(payload), None
        options.append(
            Option(key=str(key), description=desc, value=float(value) if value is not None else None)
        )
    if require_values and any(o.value is None for o in options):
        # Fall back to declaration order when the caller does not supply values.
        for i, opt in enumerate(options):
            opt.value = float(i)
    return options


def build_question(question_id: str, spec: Mapping[str, Any] | Question) -> Question:
    """Build a typed question from the wire format (or pass a Question through)."""

    if isinstance(spec, Question):
        if not spec.id:
            spec.id = question_id
        return spec

    qtype = str(spec.get("type", "choice")).lower()
    instructions = str(spec.get("instructions", "")).strip()
    context = str(spec.get("context", "") or "")
    if not instructions:
        raise ValueError(f"question {question_id!r} is missing 'instructions'")

    if qtype in {"bool", "boolean", "yesno", "yes_no", "noul"}:
        raw = spec.get("criteria")
        options = _coerce_options(raw, require_values=False) if raw else [
            Option("yes", "Yes"),
            Option("no", "No"),
        ]
        if len(options) != 2:
            raise ValueError("bool questions take exactly two criteria (yes/no)")
        options = canonical_options(options, true_first=True)
        return BoolQuestion(
            id=question_id, instructions=instructions, options=options, context=context
        )

    if qtype in {"score", "ordered", "ordinal"}:
        raw = spec.get("criteria") or spec.get("levels")
        if not raw:
            raise ValueError(f"score question {question_id!r} needs ordered 'criteria'")
        options = _coerce_options(raw, require_values=True)
        return ScoreQuestion(
            id=question_id, instructions=instructions, options=options, context=context
        )

    if qtype in {"choice", "classify", "classification"}:
        raw = spec.get("criteria") or spec.get("options")
        if not raw:
            raise ValueError(f"choice question {question_id!r} needs 'criteria'")
        options = _coerce_options(raw, require_values=False)
        return ChoiceQuestion(
            id=question_id,
            instructions=instructions,
            options=canonical_options(options),
            context=context,
        )

    raise ValueError(f"unknown question type {qtype!r} for question {question_id!r}")
