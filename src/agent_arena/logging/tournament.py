from datetime import datetime

from pydantic import BaseModel, Field


class AgentScore(BaseModel):
    agent: str
    total_score: float
    matches_as_p1: int
    matches_as_p2: int
    rounds_played: int
    cooperation_rate: float
    mean_score_per_round: float


class TournamentSummary(BaseModel):
    tournament_id: str
    game: str
    agents: list[str]
    rounds_per_match: int
    repeats: int
    total_matches: int
    match_ids: list[str] = Field(default_factory=list)
    scoreboard: list[AgentScore] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    started_at: datetime
    ended_at: datetime
