from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class MoveLog(BaseModel):
    round_num: int
    p1_move: str
    p2_move: str
    p1_score: float
    p2_score: float
    p1_reasoning: str | None = None
    p2_reasoning: str | None = None
    p1_input_tokens: int | None = None
    p1_output_tokens: int | None = None
    p1_cache_read_tokens: int | None = None
    p1_cache_write_tokens: int | None = None
    p2_input_tokens: int | None = None
    p2_output_tokens: int | None = None
    p2_cache_read_tokens: int | None = None
    p2_cache_write_tokens: int | None = None
    p1_latency_ms: float | None = None
    p2_latency_ms: float | None = None


class MatchResult(BaseModel):
    match_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    game: str
    p1_name: str
    p2_name: str
    p1_total_score: float
    p2_total_score: float
    rounds: list[MoveLog]
    total_rounds: int
    p1_total_input_tokens: int = 0
    p1_total_output_tokens: int = 0
    p2_total_input_tokens: int = 0
    p2_total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    started_at: datetime
    ended_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)
