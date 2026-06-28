"""Bank — a YAML-backed question bank with grow/sample/save convenience.

A bank file is YAML with (at least) a ``questions:`` list; any other top-level
keys are metadata, carried through untouched on save. An optional ``material:``
key holds the source text used to ground generation (or pass it explicitly).

Comments are NOT preserved on save (clean round-trip via PyYAML). If you keep a
heavily-commented bank file, generate into memory and write where you control
the formatting instead of calling ``save``.
"""

from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

import yaml

from .generate import generate_bank
from .sample import sample_test

PathLike = Union[str, Path]


class Bank:
    """Load, grow, sample, and save a question bank."""

    def __init__(self, questions: Optional[List[dict]] = None, meta: Optional[dict] = None):
        self.questions: List[dict] = list(questions or [])
        self.meta: dict = dict(meta or {})

    # ----- construction -----------------------------------------------------
    @classmethod
    def load(cls, path: PathLike) -> "Bank":
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
        questions = data.pop("questions", []) or []
        return cls(questions=questions, meta=data)

    @classmethod
    def from_dict(cls, data: dict) -> "Bank":
        data = dict(data or {})
        return cls(questions=data.pop("questions", []) or [], meta=data)

    # ----- inspection -------------------------------------------------------
    @property
    def material(self) -> str:
        return self.meta.get("material", "") or ""

    def counts_by_type(self) -> dict:
        return dict(Counter(q.get("type", "mc") for q in self.questions))

    def counts_by_difficulty(self) -> dict:
        return dict(Counter(q.get("difficulty", "medium") for q in self.questions))

    def __len__(self) -> int:
        return len(self.questions)

    # ----- mutation ---------------------------------------------------------
    def add(self, new_questions: List[dict]) -> "Bank":
        self.questions.extend(new_questions)
        return self

    def grow(self, llm, *, targets: Optional[dict] = None,
             material: Optional[str] = None, **kwargs) -> List[dict]:
        """Generate the shortfall to reach ``targets`` and append it in place.

        Grounds on ``material`` if given, else the bank's ``material`` metadata.
        Returns just the newly added questions.
        """
        mat = material if material is not None else self.material
        if not (mat or "").strip():
            raise ValueError("no material to ground generation — pass material=... "
                             "or set a 'material:' key in the bank file")
        new = generate_bank(mat, llm, targets=targets, existing=self.questions, **kwargs)
        self.add(new)
        return new

    def sample(self, blueprint: Optional[dict] = None, seen_ids=(), rng=None) -> List[dict]:
        return sample_test(self.questions, blueprint=blueprint, seen_ids=seen_ids, rng=rng)

    # ----- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {**self.meta, "questions": self.questions}

    def save(self, path: PathLike) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True,
                           width=100, default_flow_style=False)
