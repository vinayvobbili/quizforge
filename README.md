# quizforge

Generate a deep, **mixed-format** question bank from any source material, then grade it — **deterministic where it can, LLM where it must**. Bring your own chat model.

quizforge is the engine behind a training/readiness feature: it drafts far more questions than any single test shows (multiple choice, fill-in-the-blank, match-the-following, short answer, and free-response scenarios), samples a fresh shuffled test on each attempt — so two learners rarely see the same one — and grades every format. MC/fill/match are graded instantly with no model call; open-ended answers are scored 0–1 with coaching feedback by an LLM you provide.

- **Model-agnostic** — pass any LangChain-style chat model (`with_structured_output`). No SDK is bundled.
- **Deep bank, anti-sharing sampling** — unseen-first, difficulty-spread draws per a configurable blueprint.
- **Cheap grading** — only open-ended answers cost a model call; everything else is local and free.
- **Plain dicts in, plain dicts out** — YAML/JSON-friendly, easy to store and template.

## Install

```bash
pip install quizforge
```

Bring a chat model from whichever provider you use, e.g.:

```bash
pip install langchain-openai   # or langchain-anthropic, etc.
```

## Quickstart

### Generate a bank

```python
from quizforge import generate_bank
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1", temperature=0.4)

material = open("citrix_lesson.md").read()
new_questions = generate_bank(
    material, llm,
    targets={"mc": 40, "fill_blank": 20, "match": 12, "short": 16, "freetext": 12},
    existing=[],                       # pass your current bank to top it up
    coverage="At least half should be applied incident-response scenarios.",
)
# -> list of dicts with id/type/difficulty + per-format fields. Store as you like.
```

`generate_bank` only produces the *shortfall* to reach `targets`, validates each
question, and never duplicates an existing prompt — safe to re-run to grow a bank.

### Sample a test

```python
from quizforge import sample_test, DEFAULT_BLUEPRINT

test = sample_test(bank, blueprint=DEFAULT_BLUEPRINT, seen_ids=already_seen)
# DEFAULT_BLUEPRINT draws mc8 / fill4 / match2 / short4 / freetext2 = 20, shuffled.
```

### Grade

```python
from quizforge import grade_fill_blank, grade_match, grade_open_answer

grade_fill_blank(q, "ICA")                 # {"score": 1.0, "correct": True, ...}
grade_match(q, {"0": "RDP", "1": "ICA"})   # per-pair partial credit
grade_open_answer(q, learner_text, llm)    # QuizGrade(score, verdict, feedback, ...) or None
```

`grade_open_answer` returns `None` if the model was unavailable — exclude that
question from the attempt's max score rather than penalizing the learner.

## Question shapes

Each question is a dict with `id`, `type`, `difficulty`, `prompt`, plus:

- `mc` — `choices: [str]`, `answer_idx: int`, `explanation: str`
- `fill_blank` — `accepted_answers: [str]`, `explanation: str`
- `match` — `pairs: [{left, right}]`, `explanation: str`
- `short` / `freetext` — `model_answer: str`, `rubric: [str]`

## License

MIT
