from typing import Protocol
from typing import NamedTuple


class RoundRecord(NamedTuple):
    p1_move: str
    p2_move: str
    p1_score: float
    p2_score: float


class GameState(NamedTuple):
    round_num: int  # 0-indexed; the round currently being played
    history: tuple[RoundRecord, ...]  # completed rounds
    max_rounds: int


class Game(Protocol):
    max_rounds: int

    @property
    def name(self) -> str: ...

    def legal_moves(self) -> list[str]: ...

    def payoff(self, my_move: str, opp_move: str) -> tuple[float, float]:
        """Returns (my_score, opp_score)."""
        ...

    def is_terminal(self, state: GameState) -> bool: ...

    def describe(self) -> str:
        """Natural-language rules fed to the LLM as the system prompt."""
        ...
