"""Serialisation of the shared state and the isolated question branches."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Question, QuestionType

STATE_TEMPLATE = "<state>\n{state}\n</state>\n"

#: How a vision-language backbone is told "an image goes here". The middle token is the
#: one the processor expands into as many visual tokens as the image needs; the engine
#: writes one group of these per image, in order, at the head of the state.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

CHOICE_INSTRUCTION = "Reply with a single letter naming the best option."
SCORE_INSTRUCTION = "Reply with a single letter naming the best level."
#: Used by the ``verbalizer`` readout, which scores the *text* of each option instead
#: of a letter: with more options than there are single-token letters the model cannot
#: name an option by letter, so the branch asks for the option itself.
NAME_INSTRUCTION = "Reply with the name of the best option."
#: Text the decision position sits behind. The original had ``"Answer: ("``, which reads
#: as a call for a single letter - but measured over seven tasks (AG News, Rotten
#: Tomatoes, SST-2, Emotion, MMLU, C-Eval) the open form below scored higher on all
#: of them, by up to 0.13 (C-Eval 0.57 -> 0.70, MMLU 0.62 -> 0.71). The reserved-letter
#: readout sums the probability of a label token and its space-prefixed form, so both
#: cues are readable to it; the model is simply more informative when it is not steered
#: towards a bracket.
DECISION_CUE = "Answer: "

#: The two ways a branch can ask for its answer. ``letter`` is the protocol every
#: shipped number was measured with; ``name`` is the many-option escape hatch.
REPLY_STYLES = ("letter", "name")


@dataclass
class RenderedBranch:
    """A rendered question branch plus the token index of every option it offers."""

    question_id: str
    text: str
    labels: list[str]
    #: character span of each option description inside ``text``
    option_char_spans: list[tuple[int, int]] = field(default_factory=list)
    #: filled in by :func:`tokenize_branch`
    option_token_spans: list[tuple[int, int]] = field(default_factory=list)
    decision_index: int = -1

    @property
    def num_options(self) -> int:
        return len(self.labels)


def render_branch(
    question: Question, reply_style: str = "letter", cue: str | None = None
) -> RenderedBranch:
    """Render one question as a self-contained branch using the documented templates.

    ``reply_style`` picks the sentence that tells the model what to answer: a letter
    (the default, and what every readout that indexes option slots expects) or the
    option's own name (the ``verbalizer`` readout, for questions with more options than
    there are reserved single-token letters).

    ``cue`` is the text the decision position sits behind - see :data:`DECISION_CUE` for
    the default and why it is not a bracket. A readout can override it: one that reads an
    option's *name* has to leave the sentence open on an opening quote, or the model
    continues the cue itself ("Answer", "Select", "Option") and every option scores the
    same. See :attr:`qwenjev.readout.Readout.decision_cue`.
    """

    if reply_style not in REPLY_STYLES:
        raise ValueError(f"unknown reply style {reply_style!r}; expected one of {REPLY_STYLES}")
    cue = DECISION_CUE if cue is None else cue
    header = "Options" if question.type is QuestionType.CHOICE else "Levels"
    if question.type is QuestionType.SCORE:
        header = "Levels"
        instruction = SCORE_INSTRUCTION
    elif reply_style == "name":
        instruction = NAME_INSTRUCTION
    else:
        instruction = CHOICE_INSTRUCTION

    parts = [f"Question: {question.instructions}\n"]
    if question.context:
        parts.append(f"Passage:\n{question.context}\n\n")
    parts.append(f"{header}:\n")
    for label, option in zip(question.labels(), question.options):
        parts.append(f"{label}. {option.description}\n")
    parts.append(f"{instruction}\n")
    parts.append(cue)
    text = "".join(parts)

    spans: list[tuple[int, int]] = []
    cursor = (
        len(f"Question: {question.instructions}\n")
        + (len(f"Passage:\n{question.context}\n\n") if question.context else 0)
        + len(f"{header}:\n")
    )
    for label, option in zip(question.labels(), question.options):
        start = cursor + len(f"{label}. ")
        end = start + len(option.description)
        spans.append((start, end))
        cursor = end + 1  # newline

    return RenderedBranch(
        question_id=question.id,
        text=text,
        labels=question.labels(),
        option_char_spans=spans,
    )


def render_state(state: str, images: int = 0) -> str:
    """The shared state as text, with one image placeholder group per image."""

    prefix = IMAGE_PLACEHOLDER * max(0, int(images))
    return STATE_TEMPLATE.format(state=prefix + state.strip())
