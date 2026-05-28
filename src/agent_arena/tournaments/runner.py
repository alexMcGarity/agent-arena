from dataclasses import dataclass
from datetime import datetime, timezone

from agent_arena.agents.base import Agent, MoveResult
from agent_arena.games.base import Game, GameState, RoundRecord
from agent_arena.logging.result import MatchResult, MoveLog


def _flip(state: GameState) -> GameState:
    """Return state from the second player's perspective (swap p1/p2)."""
    flipped = tuple(
        RoundRecord(r.p2_move, r.p1_move, r.p2_score, r.p1_score) for r in state.history
    )
    return GameState(state.round_num, flipped, state.max_rounds)


@dataclass
class MatchRunner:
    game: Game
    p1: Agent
    p2: Agent

    def run(self) -> MatchResult:
        desc = self.game.describe()
        self.p1.reset(desc)
        self.p2.reset(desc)

        started_at = datetime.now(timezone.utc)
        state = GameState(round_num=0, history=(), max_rounds=self.game.max_rounds)
        legal = self.game.legal_moves()
        logs: list[MoveLog] = []

        while not self.game.is_terminal(state):
            r1: MoveResult = self.p1.choose_move(state, legal)
            r2: MoveResult = self.p2.choose_move(_flip(state), legal)

            p1_score, p2_score = self.game.payoff(r1.move, r2.move)

            logs.append(
                MoveLog(
                    round_num=state.round_num + 1,
                    p1_move=r1.move,
                    p2_move=r2.move,
                    p1_score=p1_score,
                    p2_score=p2_score,
                    p1_reasoning=r1.reasoning,
                    p2_reasoning=r2.reasoning,
                    p1_input_tokens=r1.input_tokens,
                    p1_output_tokens=r1.output_tokens,
                    p1_cache_read_tokens=r1.cache_read_tokens,
                    p1_cache_write_tokens=r1.cache_write_tokens,
                    p2_input_tokens=r2.input_tokens,
                    p2_output_tokens=r2.output_tokens,
                    p2_cache_read_tokens=r2.cache_read_tokens,
                    p2_cache_write_tokens=r2.cache_write_tokens,
                    p1_latency_ms=r1.latency_ms,
                    p2_latency_ms=r2.latency_ms,
                )
            )

            record = RoundRecord(r1.move, r2.move, p1_score, p2_score)
            state = GameState(
                round_num=state.round_num + 1,
                history=state.history + (record,),
                max_rounds=state.max_rounds,
            )

        ended_at = datetime.now(timezone.utc)

        return MatchResult(
            game=self.game.name,
            p1_name=self.p1.name,
            p2_name=self.p2.name,
            p1_total_score=sum(lg.p1_score for lg in logs),
            p2_total_score=sum(lg.p2_score for lg in logs),
            rounds=logs,
            total_rounds=len(logs),
            p1_total_input_tokens=sum(lg.p1_input_tokens or 0 for lg in logs),
            p1_total_output_tokens=sum(lg.p1_output_tokens or 0 for lg in logs),
            p2_total_input_tokens=sum(lg.p2_input_tokens or 0 for lg in logs),
            p2_total_output_tokens=sum(lg.p2_output_tokens or 0 for lg in logs),
            started_at=started_at,
            ended_at=ended_at,
        )
