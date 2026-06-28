"""Generation schemas — the per-format shapes the LLM must return.

These describe what a freshly *generated* question looks like coming back from
the model. After validation, questions are stored as plain dicts (YAML-friendly,
template-friendly) with an added ``id`` and ``type``; see ``generate`` and the
format-field reference in the README.
"""

from typing import List

from pydantic import BaseModel, Field

# The five question formats and the difficulty levels the kernel understands.
FORMAT_KEYS = ("mc", "fill_blank", "match", "short", "freetext")
DIFFICULTIES = ("easy", "medium", "hard")


class GenMC(BaseModel):
    """A generated multiple-choice question."""

    prompt: str
    choices: List[str] = Field(description="3-4 options; exactly one correct, others plausible-but-wrong")
    answer_idx: int = Field(description="0-based index of the correct choice")
    explanation: str = Field(description="Why the answer is right — a teaching sentence")
    difficulty: str = Field(description="easy | medium | hard")


class GenFill(BaseModel):
    """A generated fill-in-the-blank question."""

    prompt: str = Field(description="A sentence with a ______ blank where the answer belongs")
    accepted_answers: List[str] = Field(description="Every acceptable spelling/synonym/abbreviation of the answer")
    explanation: str
    difficulty: str


class GenPair(BaseModel):
    left: str
    right: str


class GenMatch(BaseModel):
    """A generated match-the-following question."""

    prompt: str
    pairs: List[GenPair] = Field(description="3-4 unambiguous one-to-one pairs; distinct right-side labels")
    explanation: str
    difficulty: str


class GenOpen(BaseModel):
    """A generated open-ended (short / free-response) question."""

    prompt: str
    model_answer: str = Field(description="A strong, concise reference answer")
    rubric: List[str] = Field(description="3-5 key points a correct answer must cover")
    difficulty: str


class MCBatch(BaseModel):
    questions: List[GenMC]


class FillBatch(BaseModel):
    questions: List[GenFill]


class MatchBatch(BaseModel):
    questions: List[GenMatch]


class OpenBatch(BaseModel):
    questions: List[GenOpen]


# Per-format: the batch schema the model fills, and a human-readable description
# spliced into the generation prompt. ``short`` and ``freetext`` share the open
# schema but differ in framing (one-liners vs. walk-me-through scenarios).
FORMATS = {
    "mc": (MCBatch, "multiple-choice questions with one best answer and 3 plausible distractors"),
    "fill_blank": (FillBatch, "fill-in-the-blank questions (a sentence with a ______ blank)"),
    "match": (MatchBatch, "match-the-following questions with 3-4 unambiguous pairs"),
    "short": (OpenBatch, "short-answer questions answerable in 1-2 sentences"),
    "freetext": (OpenBatch, "scenario / free-response questions ('walk me through how you'd…')"),
}
