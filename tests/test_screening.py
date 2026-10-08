import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening import (  # noqa: E402
    AGREE_SCORED_ITEMS,
    ANSWER_OPTIONS,
    NUM_ITEMS,
    above_cutoff,
    item_score,
    load_questions,
    total_score,
)

DA, SA, SD, DD = ANSWER_OPTIONS


@pytest.mark.parametrize("answer", [DA, SA])
def test_agreeing_scores_only_on_agree_items(answer):
    for item in range(1, NUM_ITEMS + 1):
        assert item_score(item, answer) == (1 if item in AGREE_SCORED_ITEMS else 0)


@pytest.mark.parametrize("answer", [SD, DD])
def test_disagreeing_scores_only_on_disagree_items(answer):
    for item in range(1, NUM_ITEMS + 1):
        assert item_score(item, answer) == (0 if item in AGREE_SCORED_ITEMS else 1)


def test_agree_items_are_official_key():
    assert AGREE_SCORED_ITEMS == {1, 7, 8, 10}


def test_all_agree_gives_four():
    assert total_score([DA] * NUM_ITEMS) == 4


def test_all_disagree_gives_six():
    assert total_score([DD] * NUM_ITEMS) == 6


def test_max_and_min_scores():
    trait_answers = [DA if i in AGREE_SCORED_ITEMS else DD for i in range(1, NUM_ITEMS + 1)]
    non_trait_answers = [DD if i in AGREE_SCORED_ITEMS else DA for i in range(1, NUM_ITEMS + 1)]
    assert total_score(trait_answers) == 10
    assert total_score(non_trait_answers) == 0


def test_cutoff():
    assert not above_cutoff(5)
    assert above_cutoff(6)


def test_rejects_wrong_input():
    with pytest.raises(ValueError):
        item_score(1, "Maybe")
    with pytest.raises(ValueError):
        total_score([DA] * 9)


def test_questions_file_loads():
    qs = load_questions()
    assert len(qs.questions) == NUM_ITEMS
