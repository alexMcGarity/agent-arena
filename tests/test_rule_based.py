import pytest

from agent_arena.agents.rule_based import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Pavlov,
    TitForTat,
    TitForTwoTats,
)
from agent_arena.games.base import GameState, RoundRecord

LEGAL = ["cooperate", "defect"]


def empty_state() -> GameState:
    return GameState(round_num=0, history=(), max_rounds=10)


def state_after(*moves: tuple[str, str]) -> GameState:
    records = tuple(RoundRecord(m, o, 0.0, 0.0) for m, o in moves)
    return GameState(round_num=len(records), history=records, max_rounds=10)


# ── AlwaysCooperate ────────────────────────────────────────────────────────────

def test_always_cooperate_empty() -> None:
    assert AlwaysCooperate().choose_move(empty_state(), LEGAL).move == "cooperate"


def test_always_cooperate_with_history() -> None:
    state = state_after(("defect", "defect"))
    assert AlwaysCooperate().choose_move(state, LEGAL).move == "cooperate"


def test_always_cooperate_name() -> None:
    assert AlwaysCooperate().name == "always_cooperate"


# ── AlwaysDefect ───────────────────────────────────────────────────────────────

def test_always_defect_empty() -> None:
    assert AlwaysDefect().choose_move(empty_state(), LEGAL).move == "defect"


def test_always_defect_with_history() -> None:
    state = state_after(("cooperate", "cooperate"))
    assert AlwaysDefect().choose_move(state, LEGAL).move == "defect"


def test_always_defect_name() -> None:
    assert AlwaysDefect().name == "always_defect"


# ── TitForTat ──────────────────────────────────────────────────────────────────

def test_tft_opens_cooperate() -> None:
    assert TitForTat().choose_move(empty_state(), LEGAL).move == "cooperate"


def test_tft_mirrors_cooperation() -> None:
    state = state_after(("cooperate", "cooperate"))
    assert TitForTat().choose_move(state, LEGAL).move == "cooperate"


def test_tft_mirrors_defection() -> None:
    state = state_after(("cooperate", "defect"))
    assert TitForTat().choose_move(state, LEGAL).move == "defect"


def test_tft_recovers_after_one_defection() -> None:
    state = state_after(("cooperate", "defect"), ("defect", "cooperate"))
    assert TitForTat().choose_move(state, LEGAL).move == "cooperate"


def test_tft_name() -> None:
    assert TitForTat().name == "tit_for_tat"


# ── TitForTwoTats ──────────────────────────────────────────────────────────────

def test_tftt_does_not_retaliate_first_defection() -> None:
    state = state_after(("cooperate", "defect"))
    assert TitForTwoTats().choose_move(state, LEGAL).move == "cooperate"


def test_tftt_retaliates_two_consecutive_defections() -> None:
    state = state_after(("cooperate", "defect"), ("cooperate", "defect"))
    assert TitForTwoTats().choose_move(state, LEGAL).move == "defect"


def test_tftt_no_retaliation_non_consecutive() -> None:
    state = state_after(("cooperate", "defect"), ("cooperate", "cooperate"), ("cooperate", "defect"))
    assert TitForTwoTats().choose_move(state, LEGAL).move == "cooperate"


# ── GrimTrigger ────────────────────────────────────────────────────────────────

def test_grim_opens_cooperate() -> None:
    assert GrimTrigger().choose_move(empty_state(), LEGAL).move == "cooperate"


def test_grim_triggers_on_defection() -> None:
    bot = GrimTrigger()
    state = state_after(("cooperate", "defect"))
    assert bot.choose_move(state, LEGAL).move == "defect"


def test_grim_stays_triggered() -> None:
    bot = GrimTrigger()
    state1 = state_after(("cooperate", "defect"))
    bot.choose_move(state1, LEGAL)
    state2 = state_after(("cooperate", "defect"), ("defect", "cooperate"))
    assert bot.choose_move(state2, LEGAL).move == "defect"


def test_grim_reset_clears_trigger() -> None:
    bot = GrimTrigger()
    state = state_after(("cooperate", "defect"))
    bot.choose_move(state, LEGAL)
    bot.reset()
    assert bot.choose_move(empty_state(), LEGAL).move == "cooperate"


# ── Pavlov ─────────────────────────────────────────────────────────────────────

def test_pavlov_opens_cooperate() -> None:
    assert Pavlov().choose_move(empty_state(), LEGAL).move == "cooperate"


def test_pavlov_win_stay_cooperate() -> None:
    # Scored 3 last round (R) → repeat cooperate
    records = (RoundRecord("cooperate", "cooperate", 3.0, 3.0),)
    state = GameState(1, records, 10)
    assert Pavlov().choose_move(state, LEGAL).move == "cooperate"


def test_pavlov_win_stay_defect() -> None:
    # Scored 5 last round (T) → repeat defect
    records = (RoundRecord("defect", "cooperate", 5.0, 0.0),)
    state = GameState(1, records, 10)
    assert Pavlov().choose_move(state, LEGAL).move == "defect"


def test_pavlov_lose_shift_cooperate_to_defect() -> None:
    # Scored 0 last round (S) → switch to defect
    records = (RoundRecord("cooperate", "defect", 0.0, 5.0),)
    state = GameState(1, records, 10)
    assert Pavlov().choose_move(state, LEGAL).move == "defect"


def test_pavlov_lose_shift_defect_to_cooperate() -> None:
    # Scored 1 last round (P) → switch to cooperate
    records = (RoundRecord("defect", "defect", 1.0, 1.0),)
    state = GameState(1, records, 10)
    assert Pavlov().choose_move(state, LEGAL).move == "cooperate"
