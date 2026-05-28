"""
Iterated Prisoner's Dilemma

Payoff matrix (row = my move, col = opponent's move):

              Cooperate      Defect
Cooperate   (R=3, R=3)   (S=0, T=5)
Defect      (T=5, S=0)   (P=1, P=1)

T > R > P > S: 5 > 3 > 1 > 0
2R > T + S:    6 > 5        (mutual cooperation beats alternating exploitation)
"""

from dataclasses import dataclass

from agent_arena.games.base import GameState

COOPERATE = "cooperate"
DEFECT = "defect"

T, R, P, S = 5.0, 3.0, 1.0, 0.0

_PAYOFFS: dict[tuple[str, str], tuple[float, float]] = {
    (COOPERATE, COOPERATE): (R, R),
    (COOPERATE, DEFECT): (S, T),
    (DEFECT, COOPERATE): (T, S),
    (DEFECT, DEFECT): (P, P),
}


@dataclass
class IteratedPrisonersDilemma:
    max_rounds: int = 200

    @property
    def name(self) -> str:
        return "iterated_prisoners_dilemma"

    def legal_moves(self) -> list[str]:
        return [COOPERATE, DEFECT]

    def payoff(self, my_move: str, opp_move: str) -> tuple[float, float]:
        return _PAYOFFS[(my_move, opp_move)]

    def is_terminal(self, state: GameState) -> bool:
        return state.round_num >= state.max_rounds

    def describe(self) -> str:
        return (
            "You are playing the Iterated Prisoner's Dilemma.\n\n"
            "Each round, you and your opponent simultaneously choose to cooperate or defect.\n\n"
            "Payoffs per round:\n"
            "  Both cooperate   → you +3, opponent +3\n"
            "  You defect, they cooperate → you +5, opponent +0\n"
            "  You cooperate, they defect → you +0, opponent +5\n"
            "  Both defect      → you +1, opponent +1\n\n"
            f"The game lasts {self.max_rounds} rounds. "
            "Your goal is to maximise your cumulative score."
        )
