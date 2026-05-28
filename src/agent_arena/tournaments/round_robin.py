from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from rich.console import Console

from agent_arena.agents.base import Agent
from agent_arena.games.base import Game
from agent_arena.logging.result import MatchResult
from agent_arena.logging.tournament import AgentScore, TournamentSummary
from agent_arena.tournaments.runner import MatchRunner


@dataclass
class RoundRobinConfig:
    game: Game
    agents: list[Agent]
    repeats: int = 1
    output_dir: Path = Path("data/runs")


@dataclass
class _Accum:
    score: float = 0.0
    as_p1: int = 0
    as_p2: int = 0
    rounds: int = 0
    cooperate: int = 0


class RoundRobinRunner:
    def __init__(self, config: RoundRobinConfig) -> None:
        self.config = config
        self.console = Console()

    def estimate_matches(self) -> int:
        n = len(self.config.agents)
        return n * (n - 1) * self.config.repeats  # C(n,2) pairs × 2 orderings × repeats

    def run(self) -> TournamentSummary:
        tournament_id = str(uuid.uuid4())[:8]
        out_dir = self.config.output_dir / tournament_id
        out_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc)
        agents = self.config.agents

        # Build ordered matchup list: each unordered pair plays as (A,B) and (B,A), × repeats
        matchups: list[tuple[Agent, Agent]] = []
        for a, b in combinations(agents, 2):
            for _ in range(self.config.repeats):
                matchups.append((a, b))
                matchups.append((b, a))

        total = len(matchups)
        self.console.print(
            f"[bold]Tournament {tournament_id}[/bold]  |  "
            f"{len(agents)} agents  |  {total} matches"
        )

        match_ids: list[str] = []
        results: list[MatchResult] = []

        for i, (p1, p2) in enumerate(matchups, 1):
            self.console.print(f"  [{i}/{total}] {p1.name} vs {p2.name}")
            runner = MatchRunner(game=self.config.game, p1=p1, p2=p2)
            result = runner.run()

            (out_dir / f"{result.match_id}.json").write_text(result.model_dump_json(indent=2))
            match_ids.append(result.match_id)
            results.append(result)

        ended_at = datetime.now(timezone.utc)
        scoreboard = self._scoreboard(results)

        summary = TournamentSummary(
            tournament_id=tournament_id,
            game=self.config.game.name,
            agents=[a.name for a in agents],
            rounds_per_match=self.config.game.max_rounds,
            repeats=self.config.repeats,
            total_matches=total,
            match_ids=match_ids,
            scoreboard=scoreboard,
            total_input_tokens=sum(
                r.p1_total_input_tokens + r.p2_total_input_tokens for r in results
            ),
            total_output_tokens=sum(
                r.p1_total_output_tokens + r.p2_total_output_tokens for r in results
            ),
            started_at=started_at,
            ended_at=ended_at,
        )

        (out_dir / "tournament_summary.json").write_text(summary.model_dump_json(indent=2))
        return summary

    def _scoreboard(self, results: list[MatchResult]) -> list[AgentScore]:
        accums: dict[str, _Accum] = {}

        def _get(name: str) -> _Accum:
            if name not in accums:
                accums[name] = _Accum()
            return accums[name]

        for r in results:
            p1 = _get(r.p1_name)
            p1.score += r.p1_total_score
            p1.as_p1 += 1
            p1.rounds += r.total_rounds
            p1.cooperate += sum(1 for lg in r.rounds if lg.p1_move == "cooperate")

            p2 = _get(r.p2_name)
            p2.score += r.p2_total_score
            p2.as_p2 += 1
            p2.rounds += r.total_rounds
            p2.cooperate += sum(1 for lg in r.rounds if lg.p2_move == "cooperate")

        scores = [
            AgentScore(
                agent=name,
                total_score=a.score,
                matches_as_p1=a.as_p1,
                matches_as_p2=a.as_p2,
                rounds_played=a.rounds,
                cooperation_rate=a.cooperate / a.rounds if a.rounds else 0.0,
                mean_score_per_round=a.score / a.rounds if a.rounds else 0.0,
            )
            for name, a in accums.items()
        ]
        return sorted(scores, key=lambda s: s.total_score, reverse=True)
