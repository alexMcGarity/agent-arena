"""Aggregate statistics computed from tournament match logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agent_arena.logging.result import MatchResult
from agent_arena.logging.tournament import TournamentSummary


def load_tournament(tournament_dir: Path) -> tuple[TournamentSummary, list[MatchResult]]:
    summary = TournamentSummary.model_validate(
        json.loads((tournament_dir / "tournament_summary.json").read_text())
    )
    matches: list[MatchResult] = []
    for mid in summary.match_ids:
        p = tournament_dir / f"{mid}.json"
        if p.exists():
            matches.append(MatchResult.model_validate(json.loads(p.read_text())))
    return summary, matches


def cooperation_by_round(
    matches: list[MatchResult], agent: str, n_rounds: int
) -> list[float]:
    """Per-round cooperation rate averaged across all matches where agent played."""
    coop = [0] * n_rounds
    total = [0] * n_rounds
    for m in matches:
        if m.p1_name == agent:
            for lg in m.rounds:
                idx = lg.round_num - 1
                coop[idx] += 1 if lg.p1_move == "cooperate" else 0
                total[idx] += 1
        elif m.p2_name == agent:
            for lg in m.rounds:
                idx = lg.round_num - 1
                coop[idx] += 1 if lg.p2_move == "cooperate" else 0
                total[idx] += 1
    return [c / t if t > 0 else 0.0 for c, t in zip(coop, total)]


def forgiveness_rate(matches: list[MatchResult], agent: str) -> float:
    """P(cooperate at t | opponent defected at t-1)."""
    num = den = 0
    for m in matches:
        rounds = m.rounds
        if m.p1_name == agent:
            for i in range(1, len(rounds)):
                if rounds[i - 1].p2_move == "defect":
                    den += 1
                    if rounds[i].p1_move == "cooperate":
                        num += 1
        elif m.p2_name == agent:
            for i in range(1, len(rounds)):
                if rounds[i - 1].p1_move == "defect":
                    den += 1
                    if rounds[i].p2_move == "cooperate":
                        num += 1
    return num / den if den > 0 else 0.0


def retaliation_rate(matches: list[MatchResult], agent: str) -> float:
    """P(defect at t | opponent defected at t-1)."""
    num = den = 0
    for m in matches:
        rounds = m.rounds
        if m.p1_name == agent:
            for i in range(1, len(rounds)):
                if rounds[i - 1].p2_move == "defect":
                    den += 1
                    if rounds[i].p1_move == "defect":
                        num += 1
        elif m.p2_name == agent:
            for i in range(1, len(rounds)):
                if rounds[i - 1].p1_move == "defect":
                    den += 1
                    if rounds[i].p2_move == "defect":
                        num += 1
    return num / den if den > 0 else 0.0


def head_to_head(
    matches: list[MatchResult],
) -> dict[tuple[str, str], tuple[float, float]]:
    """Mean (p1_score, p2_score) for each ordered pair across all matching matches."""
    sums: dict[tuple[str, str], list[float]] = {}
    counts: dict[tuple[str, str], int] = {}
    for m in matches:
        key = (m.p1_name, m.p2_name)
        if key not in sums:
            sums[key] = [0.0, 0.0]
            counts[key] = 0
        sums[key][0] += m.p1_total_score
        sums[key][1] += m.p2_total_score
        counts[key] += 1
    return {k: (sums[k][0] / counts[k], sums[k][1] / counts[k]) for k in sums}


@dataclass
class AgentStats:
    agent: str
    total_score: float
    rounds_played: int
    cooperation_rate: float
    mean_score_per_round: float
    forgiveness: float
    retaliation: float
    coop_by_round: list[float] = field(default_factory=list)


def compute_all(
    matches: list[MatchResult], agents: list[str], n_rounds: int
) -> list[AgentStats]:
    results = []
    for agent in agents:
        agent_matches = [m for m in matches if m.p1_name == agent or m.p2_name == agent]
        total_score = sum(
            m.p1_total_score if m.p1_name == agent else m.p2_total_score
            for m in agent_matches
        )
        rounds_played = sum(m.total_rounds for m in agent_matches)
        coop_moves = sum(
            sum(1 for lg in m.rounds if (m.p1_name == agent and lg.p1_move == "cooperate")
                or (m.p2_name == agent and lg.p2_move == "cooperate"))
            for m in agent_matches
        )
        results.append(
            AgentStats(
                agent=agent,
                total_score=total_score,
                rounds_played=rounds_played,
                cooperation_rate=coop_moves / rounds_played if rounds_played else 0.0,
                mean_score_per_round=total_score / rounds_played if rounds_played else 0.0,
                forgiveness=forgiveness_rate(matches, agent),
                retaliation=retaliation_rate(matches, agent),
                coop_by_round=cooperation_by_round(matches, agent, n_rounds),
            )
        )
    return sorted(results, key=lambda s: s.total_score, reverse=True)
