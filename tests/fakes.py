"""Test doubles — a LangChain-shaped chat model that returns canned structured
output, so the kernel can be tested with no real provider."""

import itertools

from quizforge.schemas import (
    FillBatch,
    GenFill,
    GenMatch,
    GenMC,
    GenOpen,
    GenPair,
    MatchBatch,
    MCBatch,
    OpenBatch,
)

_DIFFS = itertools.cycle(["easy", "medium", "hard"])


class _Bound:
    def __init__(self, schema, responder, calls):
        self._schema = schema
        self._responder = responder
        self._calls = calls

    def invoke(self, messages):
        self._calls.append((self._schema, messages))
        return self._responder(self._schema, messages)


class FakeLLM:
    """Quacks like a LangChain chat model: ``with_structured_output(schema)``
    then ``.invoke(messages)`` -> a validated ``schema`` instance."""

    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def with_structured_output(self, schema):
        return _Bound(schema, self._responder, self.calls)


def bank_responder(per_call: int = 8):
    """A responder that emits ``per_call`` valid questions of the requested
    batch type, with globally-unique prompts so dedup doesn't collapse them."""
    counter = itertools.count(1)

    def _make_one(batch_schema, k):
        diff = next(_DIFFS)
        if batch_schema is MCBatch:
            return GenMC(prompt=f"MC question number {k}?", choices=["alpha", "beta", "gamma", "delta"],
                         answer_idx=0, explanation="Alpha is correct.", difficulty=diff)
        if batch_schema is FillBatch:
            return GenFill(prompt=f"Fill in the blank number {k}: the ______ matters.",
                           accepted_answers=[f"answer{k}", f"ans{k}"], explanation="e", difficulty=diff)
        if batch_schema is MatchBatch:
            return GenMatch(prompt=f"Match the following set {k}.",
                            pairs=[GenPair(left=f"L{k}a", right=f"R{k}a"),
                                   GenPair(left=f"L{k}b", right=f"R{k}b")],
                            explanation="e", difficulty=diff)
        # OpenBatch (short / freetext)
        return GenOpen(prompt=f"Explain concept number {k}.", model_answer="A solid answer.",
                       rubric=[f"point-{k}-1", f"point-{k}-2"], difficulty=diff)

    def _respond(schema, _messages):
        items = [_make_one(schema, next(counter)) for _ in range(per_call)]
        return schema(questions=items)

    return _respond
