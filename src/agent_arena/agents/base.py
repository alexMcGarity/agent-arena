from typing import Protocol
from typing import NamedTuple


class MoveResult(NamedTuple):
    move: str
    reasoning: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    latency_ms: float | None = None


class Agent(Protocol):
    @property
    def name(self) -> str: ...

    def choose_move(self, state: "GameState", legal_moves: list[str]) -> MoveResult: ...  # noqa: F821

    def reset(self, game_description: str = "") -> None:
        """Called by the runner before each match. Pass game description for LLM agents."""
        ...


# Avoid circular import by using string annotation above
from agent_arena.games.base import GameState as GameState  # noqa: E402, F401
