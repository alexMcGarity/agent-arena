"""Publication-quality chart generation for tournament results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from agent_arena.analysis.stats import AgentStats

# LinkedIn-friendly dimensions (px at 150 dpi)
HERO_SIZE = (12, 6.27)   # 1800×940 → 1200×627 after scaling
SQUARE_SIZE = (7.2, 7.2)  # 1080×1080


_BG = "#111827"       # dark navy
_FG = "#F9FAFB"       # near-white text
_GRID = "#1F2937"     # subtle gridlines
_MEDAL = ["#F59E0B", "#94A3B8", "#B45309"]  # gold, silver, bronze
_DEFAULT = "#3B82F6"  # blue for mid-ranks
_LAST = "#EF4444"     # red for last place

_DISPLAY: dict[str, str] = {
    "selfish": "Claude  ·  selfish",
    "neutral": "Claude  ·  neutral",
    "cooperative": "Claude  ·  cooperative",
    "academic": "Claude  ·  academic",
    "tit_for_tat": "Tit-for-Tat",
    "tit_for_two_tats": "Tit-for-Two-Tats",
    "always_cooperate": "Always Cooperate",
    "always_defect": "Always Defect",
    "grim_trigger": "Grim Trigger",
    "pavlov": "Pavlov",
    "random": "Random (50/50)",
}

# Haiku / Opus model variants share the same persona suffixes
_MODEL_PREFIX: dict[str, str] = {
    "haiku": "Haiku",
    "opus": "Opus",
    "sonnet": "Claude",
}


def _short(name: str) -> str:
    """claude-sonnet-4-6:selfish → selfish  |  tit_for_tat → tit_for_tat"""
    return name.split(":")[-1] if ":" in name else name


def _display(name: str) -> str:
    """Return a human-readable label for an agent name."""
    short = _short(name)
    if short in _DISPLAY:
        return _DISPLAY[short]
    # Handle model-prefixed variants like "haiku:selfish"
    parts = name.split(":")
    if len(parts) == 2:
        for key, prefix in _MODEL_PREFIX.items():
            if key in parts[0]:
                persona = parts[1]
                base = _DISPLAY.get(f"__{persona}", persona)
                return f"{prefix}  ·  {persona}"
    return short.replace("_", " ").title()


def score_bar(
    stats: list[AgentStats],
    out_path: Path,
    title: str = "Tournament Scoreboard",
    subtitle: str = "",
) -> None:
    """Dark-background horizontal scoreboard chart styled for LinkedIn."""
    n = len(stats)
    # Reverse so highest rank is at the top
    ordered = list(reversed(stats))

    labels = [_display(s.agent) for s in ordered]
    scores_per_round = [s.mean_score_per_round for s in ordered]
    coop_rates = [s.cooperation_rate for s in ordered]

    # Assign bar colors: medal colors for top 3, red for last, blue for rest
    bar_colors: list[str] = []
    for i, s in enumerate(ordered):
        rank = n - 1 - i  # rank 0 = first place (highest)
        if rank < 3:
            bar_colors.append(_MEDAL[rank])
        elif rank == n - 1:
            bar_colors.append(_LAST)
        else:
            bar_colors.append(_DEFAULT)

    fig, ax = plt.subplots(figsize=HERO_SIZE, facecolor=_BG)
    ax.set_facecolor(_BG)

    y = np.arange(n)
    bars = ax.barh(y, scores_per_round, color=bar_colors, height=0.6,
                   edgecolor=_BG, linewidth=1.5)

    # Score/round labels at end of each bar
    x_max = max(scores_per_round)
    for bar, spr, coop in zip(bars, scores_per_round, coop_rates):
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{spr:.2f} pts/rd  ·  {coop:.0%} coop",
            va="center", ha="left", fontsize=9.5, color=_FG, alpha=0.85,
        )

    # Rank labels on the left
    rank_labels = ["#1", "#2", "#3"] + [f"#{i+1}" for i in range(3, n - 1)] + [f"#{n}"]
    for i, (label, rank_lbl) in enumerate(zip(labels, reversed(rank_labels))):
        ax.text(
            -x_max * 0.01, i,
            f"{rank_lbl}  {label}",
            va="center", ha="right", fontsize=11, color=_FG,
            fontweight="bold" if i >= n - 3 or i == n - 1 else "normal",
        )

    ax.set_xlim(-x_max * 0.35, x_max * 1.38)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Subtle vertical gridlines
    for v in np.linspace(0, x_max, 5):
        ax.axvline(v, color=_GRID, linewidth=0.8, zorder=0)

    # Title
    top = 0.93 if not subtitle else 0.96
    fig.text(0.5, top, title, ha="center", va="top",
             fontsize=16, fontweight="bold", color=_FG)
    if subtitle:
        fig.text(0.5, 0.88, subtitle, ha="center", va="top",
                 fontsize=11, color=_FG, alpha=0.6)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.85))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


def coop_over_time(
    stats: list[AgentStats],
    out_path: Path,
    title: str = "Cooperation Rate Over Time",
    window: int = 5,
) -> None:
    """Smoothed cooperation rate per round for each agent."""
    fig, ax = plt.subplots(figsize=HERO_SIZE)
    for s in stats:
        if not s.coop_by_round:
            continue
        rates = np.array(s.coop_by_round)
        # Rolling mean smoothing
        if len(rates) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(rates, kernel, mode="valid")
            x = np.arange(window, len(rates) + 1)
        else:
            smoothed = rates
            x = np.arange(1, len(rates) + 1)
        ax.plot(x, smoothed, label=_short(s.agent), linewidth=2)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cooperation Rate", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def heatmap(
    h2h: dict[tuple[str, str], tuple[float, float]],
    agents: list[str],
    out_path: Path,
    title: str = "Head-to-Head: Avg Score per Round (P1 row vs P2 col)",
    rounds_per_match: int = 1,
) -> None:
    n = len(agents)
    grid = np.full((n, n), np.nan)
    for (p1, p2), (s1, _) in h2h.items():
        if p1 in agents and p2 in agents:
            i, j = agents.index(p1), agents.index(p2)
            grid[i, j] = s1 / rounds_per_match  # per-round score

    fig, ax = plt.subplots(figsize=SQUARE_SIZE)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=5, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Avg Score / Round", fontsize=10)

    labels = [_short(a) for a in agents]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Opponent (P2)", fontsize=11)
    ax.set_ylabel("Agent (P1)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for i in range(n):
        for j in range(n):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def coop_over_time_animated(
    stats: list[AgentStats],
    out_path: Path,
    title: str = "Cooperation Rate Over Time",
    window: int = 5,
    fps: int = 12,
    dpi: int = 100,
) -> Path:
    """Animated cooperation-rate line chart. Returns the path actually written (.mp4)."""
    import imageio.v3 as iio  # type: ignore[import-untyped]

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for s in stats:
        if not s.coop_by_round:
            continue
        rates = np.array(s.coop_by_round)
        if len(rates) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(rates, kernel, mode="valid")
            x = np.arange(window, len(rates) + 1)
        else:
            smoothed = rates
            x = np.arange(1, len(rates) + 1)
        series.append((_short(s.agent), x, smoothed))

    if not series:
        return out_path

    data_frames = max(len(x) for _, x, _ in series)
    hold_frames = fps  # 1-second freeze at the end
    n_frames = data_frames + hold_frames
    x_min = int(min(x[0] for _, x, _ in series))
    x_max = int(max(x[-1] for _, x, _ in series))

    fig, ax = plt.subplots(figsize=HERO_SIZE)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cooperation Rate", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore[index]
    line_artists: list[tuple[object, np.ndarray, np.ndarray]] = []
    for i, (label, x, y) in enumerate(series):
        (line,) = ax.plot([], [], label=label, linewidth=2, color=colors[i % len(colors)])
        line_artists.append((line, x, y))

    ax.legend(fontsize=10, framealpha=0.7)
    round_text = ax.text(0.97, 0.05, "", transform=ax.transAxes,
                         ha="right", va="bottom", fontsize=10, color="gray")
    fig.tight_layout()

    # Render each frame to a numpy array and write directly via imageio/pyav
    out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with iio.imopen(str(out_path), "w", plugin="pyav") as writer:
        writer.init_video_stream("mpeg4", fps=fps)  # type: ignore[attr-defined]
        for frame_idx in range(n_frames):
            end = min(frame_idx + 1, data_frames)
            for line, x, y in line_artists:
                clip = min(end, len(x))
                line.set_data(x[:clip], y[:clip])  # type: ignore[attr-defined]
            _, first_x, _ = line_artists[0]
            cur_round = int(first_x[min(end - 1, len(first_x) - 1)])
            round_text.set_text(f"Round {cur_round}")
            fig.canvas.draw()
            w_px, h_px = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
            writer.write_frame(buf.reshape(h_px, w_px, 4)[:, :, :3])  # type: ignore[attr-defined]

    plt.close(fig)
    return out_path


def behavioral(
    stats: list[AgentStats],
    out_path: Path,
    title: str = "Forgiveness vs Retaliation",
) -> None:
    fig, ax = plt.subplots(figsize=HERO_SIZE)
    x = np.arange(len(stats))
    w = 0.35
    labels = [_short(s.agent) for s in stats]
    fvals = [s.forgiveness for s in stats]
    rvals = [s.retaliation for s in stats]

    b1 = ax.bar(x - w / 2, fvals, w, label="Forgiveness", color="steelblue", edgecolor="white")
    b2 = ax.bar(x + w / 2, rvals, w, label="Retaliation", color="tomato", edgecolor="white")
    ax.bar_label(b1, labels=[f"{v:.0%}" for v in fvals], padding=2, fontsize=9)
    ax.bar_label(b2, labels=[f"{v:.0%}" for v in rvals], padding=2, fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
