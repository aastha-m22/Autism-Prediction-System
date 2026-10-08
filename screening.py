"""AQ-10 (Adult) scoring logic used by the web app.

Kept free of any UI code so it can be unit-tested on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ANSWER_OPTIONS = [
    "Definitely agree",
    "Slightly agree",
    "Slightly disagree",
    "Definitely disagree",
]

# Items (1-based) where agreeing scores a point. On every other item,
# disagreeing scores a point. This is the official AQ-10 Adult scoring key.
AGREE_SCORED_ITEMS = {1, 7, 8, 10}
NUM_ITEMS = 10
CUTOFF = 6

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
EXAMPLE_QUESTIONS_FILE = Path(__file__).parent / "questions.example.json"


@dataclass
class QuestionSet:
    questions: list[str]
    official: bool  # False while placeholder wording is in use


def load_questions() -> QuestionSet:
    """Load questions.json if present, otherwise the placeholder example file."""
    path = QUESTIONS_FILE if QUESTIONS_FILE.exists() else EXAMPLE_QUESTIONS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    if len(questions) != NUM_ITEMS:
        raise ValueError(f"{path.name} must contain exactly {NUM_ITEMS} questions, found {len(questions)}")
    return QuestionSet(questions=questions, official=bool(data.get("official", False)))


def item_score(item_number: int, answer: str) -> int:
    """Score one item: 1 if the answer leans towards autistic traits, else 0."""
    if answer not in ANSWER_OPTIONS:
        raise ValueError(f"Unknown answer: {answer!r}")
    agreed = answer in ANSWER_OPTIONS[:2]
    return int(agreed) if item_number in AGREE_SCORED_ITEMS else int(not agreed)


def total_score(answers: list[str]) -> int:
    if len(answers) != NUM_ITEMS:
        raise ValueError(f"Expected {NUM_ITEMS} answers, got {len(answers)}")
    return sum(item_score(i, a) for i, a in enumerate(answers, start=1))


def above_cutoff(score: int) -> bool:
    return score >= CUTOFF
