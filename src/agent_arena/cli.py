from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from agent_arena.agents.claude import ClaudeAgent  # noqa: E402
from agent_arena.agents.models import PRICING  # noqa: E402
from agent_arena.analysis import plots as _plots  # noqa: E402
from agent_arena.analysis.stats import compute_all, head_to_head, load_tournament  # noqa: E402
from agent_arena.agents.prompts import Persona  # noqa: E402
from agent_arena.agents.rule_based import (  # noqa: E402
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Pavlov,
    RandomAgent,
    TitForTat,
    TitForTwoTats,
)
from agent_arena.games.prisoners_dilemma import IteratedPrisonersDilemma  # noqa: E402
from agent_arena.logging.tournament import TournamentSummary  # noqa: E402
from agent_arena.tournaments.presets import DEFAULT_RULE_BOTS, MODEL_LABELS, TOURNAMENT_PRESETS  # noqa: E402
from agent_arena.tournaments.round_robin import RoundRobinConfig, RoundRobinRunner  # noqa: E402
from agent_arena.tournaments.runner import MatchRunner  # noqa: E402

app = typer.Typer(name="arena", help="AgentArena: LLM game theory tournament lab.")
console = Console()

GAME_REGISTRY = {
    "ipd": IteratedPrisonersDilemma,
}

# Single-agent factories (name -> callable returning fresh instance)
AGENT_FACTORIES: dict[str, object] = {
    "sonnet-neutral": lambda: ClaudeAgent(model="claude-sonnet-4-6", persona=Persona.NEUTRAL),
    "sonnet-cooperative": lambda: ClaudeAgent(model="claude-sonnet-4-6", persona=Persona.COOPERATIVE),
    "sonnet-selfish": lambda: ClaudeAgent(model="claude-sonnet-4-6", persona=Persona.SELFISH),
    "sonnet-academic": lambda: ClaudeAgent(model="claude-sonnet-4-6", persona=Persona.ACADEMIC),
    "sonnet-cot": lambda: ClaudeAgent(model="claude-sonnet-4-6", persona=Persona.COT),
    "haiku-neutral": lambda: ClaudeAgent(model="claude-haiku-4-5-20251001", persona=Persona.NEUTRAL),
    "haiku-cooperative": lambda: ClaudeAgent(model="claude-haiku-4-5-20251001", persona=Persona.COOPERATIVE),
    "haiku-selfish": lambda: ClaudeAgent(model="claude-haiku-4-5-20251001", persona=Persona.SELFISH),
    "haiku-academic": lambda: ClaudeAgent(model="claude-haiku-4-5-20251001", persona=Persona.ACADEMIC),
    "haiku-cot": lambda: ClaudeAgent(model="claude-haiku-4-5-20251001", persona=Persona.COT),
    "opus-neutral": lambda: ClaudeAgent(model="claude-opus-4-7", persona=Persona.NEUTRAL),
    "opus-cooperative": lambda: ClaudeAgent(model="claude-opus-4-7", persona=Persona.COOPERATIVE),
    "opus-selfish": lambda: ClaudeAgent(model="claude-opus-4-7", persona=Persona.SELFISH),
    "opus-academic": lambda: ClaudeAgent(model="claude-opus-4-7", persona=Persona.ACADEMIC),
    "opus-cot": lambda: ClaudeAgent(model="claude-opus-4-7", persona=Persona.COT),
    "tit-for-tat": TitForTat,
    "tit-for-two-tats": TitForTwoTats,
    "always-cooperate": AlwaysCooperate,
    "always-defect": AlwaysDefect,
    "grim-trigger": GrimTrigger,
    "pavlov": Pavlov,
    "random": RandomAgent,
}

_SOFT_CAP_USD = 5.0


def _make_agent(name: str) -> object:
    factory = AGENT_FACTORIES.get(name)
    if factory is None:
        valid = ", ".join(sorted(AGENT_FACTORIES))
        typer.echo(f"Unknown agent '{name}'. Available:\n  {valid}", err=True)
        raise typer.Exit(1)
    return factory() if callable(factory) else factory  # type: ignore[operator]


def _resolve_tournament_agents(preset: str, variants: str, include_bots: bool) -> list[object]:
    if preset not in TOURNAMENT_PRESETS:
        typer.echo(
            f"Unknown tournament preset '{preset}'. Available: {', '.join(TOURNAMENT_PRESETS)}",
            err=True,
        )
        raise typer.Exit(1)

    model_ids = TOURNAMENT_PRESETS[preset]
    persona_names = [v.strip() for v in variants.split(",")]
    agents: list[object] = []

    for model_id in model_ids:
        label = MODEL_LABELS[model_id]
        for pname in persona_names:
            try:
                persona = Persona(pname)
            except ValueError:
                typer.echo(
                    f"Unknown variant '{pname}'. Choose from: {', '.join(p.value for p in Persona)}",
                    err=True,
                )
                raise typer.Exit(1)
            agents.append(ClaudeAgent(model=model_id, persona=persona))

    if include_bots:
        for bot_name in DEFAULT_RULE_BOTS:
            agents.append(_make_agent(bot_name))

    return agents


def _estimate_claude_match_cost(model: str, rounds: int) -> float:
    pricing = PRICING[model]
    total_history = sum(i * 15 for i in range(rounds))
    sys_tokens = 600
    cache_write = sys_tokens
    cache_reads = sys_tokens * (rounds - 1)
    uncached_in = total_history
    total_out = 80 * rounds
    return (
        (uncached_in / 1_000_000) * pricing["input"]
        + (cache_write / 1_000_000) * pricing["cache_write"]
        + (cache_reads / 1_000_000) * pricing["cache_read"]
        + (total_out / 1_000_000) * pricing["output"]
    )


def _print_scoreboard(summary: TournamentSummary) -> None:
    table = Table(title=f"Tournament {summary.tournament_id} — Scoreboard")
    table.add_column("Rank", justify="right")
    table.add_column("Agent")
    table.add_column("Total Score", justify="right")
    table.add_column("Score/Round", justify="right")
    table.add_column("Coop Rate", justify="right")
    table.add_column("Matches", justify="right")
    for rank, s in enumerate(summary.scoreboard, 1):
        table.add_row(
            str(rank),
            s.agent,
            f"{s.total_score:.0f}",
            f"{s.mean_score_per_round:.3f}",
            f"{s.cooperation_rate:.1%}",
            str(s.matches_as_p1 + s.matches_as_p2),
        )
    console.print(table)


# ── arena run ─────────────────────────────────────────────────────────────────

@app.command()
def run(
    game: Annotated[str, typer.Argument(help="Game name (ipd)")] = "ipd",
    players: Annotated[Optional[str], typer.Option(help="Two comma-separated agents for a single match")] = None,
    tournament: Annotated[Optional[str], typer.Option(help="Tournament preset name (e.g. claude-trio)")] = None,
    variants: Annotated[str, typer.Option(help="Comma-separated persona variants")] = "neutral",
    rounds: Annotated[int, typer.Option(help="Rounds per match")] = 200,
    repeats: Annotated[int, typer.Option(help="Repeat count for each pairing (tournament mode)")] = 1,
    include_bots: Annotated[bool, typer.Option(help="Add rule-based bots to tournament")] = False,
    output: Annotated[Path, typer.Option(help="Output directory")] = Path("data/runs"),
) -> None:
    """Run a single match (--players) or a round-robin tournament (--tournament)."""
    if game not in GAME_REGISTRY:
        typer.echo(f"Unknown game '{game}'. Available: {', '.join(GAME_REGISTRY)}", err=True)
        raise typer.Exit(1)

    g = GAME_REGISTRY[game](max_rounds=rounds)

    if tournament:
        _run_tournament(g, tournament, variants, repeats, include_bots, output)
    elif players:
        _run_single(g, players, output)
    else:
        typer.echo("Provide --players <a,b> for a single match or --tournament <preset>.", err=True)
        raise typer.Exit(1)


def _run_single(g: object, players: str, output: Path) -> None:  # type: ignore[type-arg]
    p_names = [n.strip() for n in players.split(",")]
    if len(p_names) != 2:
        typer.echo("--players requires exactly two names.", err=True)
        raise typer.Exit(1)
    p1, p2 = _make_agent(p_names[0]), _make_agent(p_names[1])

    console.print(f"[bold]{p1.name}[/bold] vs [bold]{p2.name}[/bold]  |  {g.name}  |  {g.max_rounds} rounds")  # type: ignore[attr-defined]

    runner = MatchRunner(game=g, p1=p1, p2=p2)  # type: ignore[arg-type]
    result = runner.run()

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"{result.match_id}.json"
    out_path.write_text(result.model_dump_json(indent=2))

    table = Table(title="Match Result")
    table.add_column("Player")
    table.add_column("Score", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_row(result.p1_name, f"{result.p1_total_score:.0f}", f"{result.p1_total_input_tokens:,}")
    table.add_row(result.p2_name, f"{result.p2_total_score:.0f}", f"{result.p2_total_input_tokens:,}")
    console.print(table)
    console.print(f"[dim]Saved: {out_path}[/dim]")


def _run_tournament(
    g: object,  # type: ignore[type-arg]
    preset: str,
    variants: str,
    repeats: int,
    include_bots: bool,
    output: Path,
) -> None:
    agents = _resolve_tournament_agents(preset, variants, include_bots)

    config = RoundRobinConfig(
        game=g,  # type: ignore[arg-type]
        agents=agents,  # type: ignore[arg-type]
        repeats=repeats,
        output_dir=output,
    )
    runner = RoundRobinRunner(config)
    n_matches = runner.estimate_matches()
    console.print(
        f"[bold]Tournament:[/bold] {preset} | variants={variants} | "
        f"{len(agents)} agents | {n_matches} matches | {g.max_rounds} rounds each"  # type: ignore[attr-defined]
    )

    summary = runner.run()
    _print_scoreboard(summary)
    console.print(f"[dim]Results: data/runs/{summary.tournament_id}/[/dim]")


# ── arena cost ────────────────────────────────────────────────────────────────

@app.command()
def cost(
    game: Annotated[str, typer.Argument()] = "ipd",
    players: Annotated[Optional[str], typer.Option()] = None,
    tournament: Annotated[Optional[str], typer.Option()] = None,
    variants: Annotated[str, typer.Option()] = "neutral",
    rounds: Annotated[int, typer.Option()] = 200,
    repeats: Annotated[int, typer.Option()] = 1,
    include_bots: Annotated[bool, typer.Option()] = False,
) -> None:
    """Estimate API cost before running."""
    if tournament:
        agents = _resolve_tournament_agents(tournament, variants, include_bots)
        n = len(agents)
        n_matches = n * (n - 1) * repeats
        claude_agents = [a for a in agents if isinstance(a, ClaudeAgent)]
        console.print(
            f"\nCost estimate: {tournament} | {n} agents | {n_matches} matches | {rounds} rounds\n"
        )
        total = 0.0
        for agent in claude_agents:
            per_match = _estimate_claude_match_cost(agent.model, rounds)
            matches_played = (n - 1) * 2 * repeats
            est = per_match * matches_played
            total += est
            console.print(f"  [cyan]{agent.name}[/cyan]: ~${est:.4f} ({matches_played} matches)")
        console.print(f"\n  [bold]Estimated total: ~${total:.4f}[/bold]\n")
    elif players:
        p_names = [n.strip() for n in players.split(",")]
        total = 0.0
        console.print(f"\nCost estimate: single match | {rounds} rounds\n")
        for name in p_names:
            agent = _make_agent(name)
            if isinstance(agent, ClaudeAgent):
                est = _estimate_claude_match_cost(agent.model, rounds)
                total += est
                console.print(f"  [cyan]{name}[/cyan]: ~${est:.4f}")
            else:
                console.print(f"  [green]{name}[/green]: free (rule-based)")
        console.print(f"\n  [bold]Estimated total: ~${total:.4f}[/bold]\n")
    else:
        typer.echo("Provide --players or --tournament.", err=True)
        raise typer.Exit(1)

    if total > _SOFT_CAP_USD:
        typer.confirm(
            f"Projected spend ${total:.2f} exceeds the ${_SOFT_CAP_USD:.0f} soft cap. Proceed?",
            abort=True,
        )


# ── arena summary ─────────────────────────────────────────────────────────────

@app.command()
def summary(
    tournament_id: Annotated[str, typer.Argument(help="Tournament ID or path to summary JSON")],
    runs_dir: Annotated[Path, typer.Option()] = Path("data/runs"),
) -> None:
    """Print the scoreboard for a completed tournament."""
    # Accept either a short ID or a full path
    summary_path = Path(tournament_id) if Path(tournament_id).exists() else runs_dir / tournament_id / "tournament_summary.json"

    if not summary_path.exists():
        typer.echo(f"No summary found at {summary_path}", err=True)
        raise typer.Exit(1)

    data = TournamentSummary.model_validate(json.loads(summary_path.read_text()))
    _print_scoreboard(data)
    elapsed = (data.ended_at - data.started_at).total_seconds()
    console.print(
        f"[dim]{data.total_matches} matches | "
        f"{data.total_input_tokens:,} input tokens | "
        f"elapsed {elapsed:.0f}s[/dim]"
    )


# ── arena analyze ─────────────────────────────────────────────────────────────

@app.command()
def analyze(
    tournament_id: Annotated[str, typer.Argument(help="Tournament ID (short 8-char or full path)")],
    runs_dir: Annotated[Path, typer.Option()] = Path("data/runs"),
    assets_dir: Annotated[Path, typer.Option()] = Path("articles/assets"),
) -> None:
    """Generate analysis charts for a completed tournament."""
    t_dir = Path(tournament_id) if Path(tournament_id).is_dir() else runs_dir / tournament_id
    if not t_dir.exists():
        typer.echo(f"Tournament directory not found: {t_dir}", err=True)
        raise typer.Exit(1)

    console.print(f"Loading tournament [bold]{tournament_id}[/bold]...")
    summary, matches = load_tournament(t_dir)

    if not matches:
        typer.echo("No match files found in tournament directory.", err=True)
        raise typer.Exit(1)

    agents = summary.agents
    n_rounds = summary.rounds_per_match
    out = assets_dir / tournament_id
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"  {len(matches)} matches | {len(agents)} agents | {n_rounds} rounds each")
    console.print(f"  Charts -> {out}/")

    stats = compute_all(matches, agents, n_rounds)
    h2h = head_to_head(matches)

    # 1. Score bar
    _plots.score_bar(
        stats, out / "score_bar.png",
        title="Tournament Scoreboard",
        subtitle=f"{summary.game}  ·  {summary.rounds_per_match} rounds  ·  {summary.total_matches} matches  ·  {tournament_id}",
    )
    console.print("  [green]score_bar.png[/green]")

    # 2. Cooperation over time
    _plots.coop_over_time(stats, out / "coop_over_time.png", title=f"Cooperation Rate Over Time — {tournament_id}")
    console.print("  [green]coop_over_time.png[/green]")

    # 3. Head-to-head heatmap
    _plots.heatmap(h2h, agents, out / "heatmap.png", rounds_per_match=n_rounds,
                   title=f"Head-to-Head Avg Score/Round — {tournament_id}")
    console.print("  [green]heatmap.png[/green]")

    # 4. Forgiveness / Retaliation
    _plots.behavioral(stats, out / "behavioral.png", title=f"Forgiveness vs Retaliation — {tournament_id}")
    console.print("  [green]behavioral.png[/green]")

    # 5. Animated cooperation over time
    anim_path = _plots.coop_over_time_animated(
        stats, out / "coop_animated.mp4",
        title=f"Cooperation Rate Over Time — {tournament_id}",
    )
    console.print(f"  [green]{anim_path.name}[/green]")

    # Print stats table
    table = Table(title="Agent Stats")
    table.add_column("Agent")
    table.add_column("Score", justify="right")
    table.add_column("Score/Rd", justify="right")
    table.add_column("Coop%", justify="right")
    table.add_column("Forgive%", justify="right")
    table.add_column("Retaliate%", justify="right")
    for s in stats:
        table.add_row(
            _short_name(s.agent),
            f"{s.total_score:.0f}",
            f"{s.mean_score_per_round:.3f}",
            f"{s.cooperation_rate:.0%}",
            f"{s.forgiveness:.0%}",
            f"{s.retaliation:.0%}",
        )
    console.print(table)


def _short_name(name: str) -> str:
    return name.split(":")[-1] if ":" in name else name


if __name__ == "__main__":
    app()
