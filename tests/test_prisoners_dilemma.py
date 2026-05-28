import pytest

from agent_arena.games.base import GameState, RoundRecord
from agent_arena.games.prisoners_dilemma import (
    COOPERATE,
    DEFECT,
    P,
    R,
    S,
    T,
    IteratedPrisonersDilemma,
)


@pytest.fixture
def game() -> IteratedPrisonersDilemma:
    return IteratedPrisonersDilemma(max_rounds=10)


def _state(round_num: int = 0, max_rounds: int = 10) -> GameState:
    return GameState(round_num=round_num, history=(), max_rounds=max_rounds)


def test_legal_moves(game: IteratedPrisonersDilemma) -> None:
    assert set(game.legal_moves()) == {COOPERATE, DEFECT}


def test_payoff_both_cooperate(game: IteratedPrisonersDilemma) -> None:
    assert game.payoff(COOPERATE, COOPERATE) == (R, R)


def test_payoff_defect_vs_cooperate(game: IteratedPrisonersDilemma) -> None:
    my, opp = game.payoff(DEFECT, COOPERATE)
    assert my == T and opp == S


def test_payoff_cooperate_vs_defect(game: IteratedPrisonersDilemma) -> None:
    my, opp = game.payoff(COOPERATE, DEFECT)
    assert my == S and opp == T


def test_payoff_both_defect(game: IteratedPrisonersDilemma) -> None:
    assert game.payoff(DEFECT, DEFECT) == (P, P)


def test_payoff_ordering(game: IteratedPrisonersDilemma) -> None:
    assert T > R > P > S
    assert 2 * R > T + S  # mutual cooperation > alternating exploitation


def test_not_terminal_at_start(game: IteratedPrisonersDilemma) -> None:
    assert not game.is_terminal(_state(0))


def test_not_terminal_mid_game(game: IteratedPrisonersDilemma) -> None:
    assert not game.is_terminal(_state(5))


def test_terminal_at_max_rounds(game: IteratedPrisonersDilemma) -> None:
    assert game.is_terminal(_state(10))


def test_terminal_past_max_rounds(game: IteratedPrisonersDilemma) -> None:
    assert game.is_terminal(_state(11))


def test_describe_mentions_cooperate_and_defect(game: IteratedPrisonersDilemma) -> None:
    desc = game.describe().lower()
    assert "cooperate" in desc
    assert "defect" in desc


def test_describe_mentions_round_count(game: IteratedPrisonersDilemma) -> None:
    assert "10" in game.describe()


def test_name(game: IteratedPrisonersDilemma) -> None:
    assert game.name == "iterated_prisoners_dilemma"


def test_max_rounds_respected() -> None:
    g = IteratedPrisonersDilemma(max_rounds=5)
    assert not g.is_terminal(_state(4, 5))
    assert g.is_terminal(_state(5, 5))
