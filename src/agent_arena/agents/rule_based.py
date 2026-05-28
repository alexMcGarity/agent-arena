from dataclasses import dataclass

from agent_arena.agents.base import MoveResult
from agent_arena.games.base import GameState


@dataclass
class AlwaysCooperate:
    @property
    def name(self) -> str:
        return "always_cooperate"

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        return MoveResult(move="cooperate")

    def reset(self, game_description: str = "") -> None:
        pass


@dataclass
class AlwaysDefect:
    @property
    def name(self) -> str:
        return "always_defect"

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        return MoveResult(move="defect")

    def reset(self, game_description: str = "") -> None:
        pass


@dataclass
class TitForTat:
    """Cooperates on round 1; thereafter mirrors the opponent's previous move."""

    @property
    def name(self) -> str:
        return "tit_for_tat"

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        if not state.history:
            return MoveResult(move="cooperate", reasoning="opening move")
        last_opp = state.history[-1].p2_move
        return MoveResult(move=last_opp, reasoning=f"mirroring: {last_opp}")

    def reset(self, game_description: str = "") -> None:
        pass


@dataclass
class TitForTwoTats:
    """Defects only after the opponent defects twice in a row."""

    @property
    def name(self) -> str:
        return "tit_for_two_tats"

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        if len(state.history) < 2:
            return MoveResult(move="cooperate")
        last_two = state.history[-2:]
        if all(r.p2_move == "defect" for r in last_two):
            return MoveResult(move="defect", reasoning="opponent defected twice")
        return MoveResult(move="cooperate")

    def reset(self, game_description: str = "") -> None:
        pass


@dataclass
class GrimTrigger:
    """Cooperates until the opponent defects once, then defects forever."""

    _triggered: bool = False

    @property
    def name(self) -> str:
        return "grim_trigger"

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        if not self._triggered and state.history and state.history[-1].p2_move == "defect":
            self._triggered = True
        if self._triggered:
            return MoveResult(move="defect", reasoning="triggered")
        return MoveResult(move="cooperate")

    def reset(self, game_description: str = "") -> None:
        self._triggered = False


@dataclass
class Pavlov:
    """Win-stay, lose-shift: repeats last move if it scored >= 3, otherwise switches."""

    @property
    def name(self) -> str:
        return "pavlov"

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        if not state.history:
            return MoveResult(move="cooperate")
        last = state.history[-1]
        won = last.p1_score >= 3.0
        return MoveResult(
            move=last.p1_move if won else ("defect" if last.p1_move == "cooperate" else "cooperate"),
            reasoning="win-stay" if won else "lose-shift",
        )

    def reset(self, game_description: str = "") -> None:
        pass


@dataclass
class RandomAgent:
    """Cooperates with probability p (default 0.5)."""

    p: float = 0.5

    @property
    def name(self) -> str:
        return f"random_{int(self.p * 100)}"

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        import random

        move = "cooperate" if random.random() < self.p else "defect"
        return MoveResult(move=move)

    def reset(self, game_description: str = "") -> None:
        pass
