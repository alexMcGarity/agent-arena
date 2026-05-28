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

# ── Theme ──────────────────────────────────────────────────────────────────────
_BG    = "#111827"
_BG2   = "#1F2937"   # slightly lighter for grid / colorbar bg
_FG    = "#F9FAFB"
_FG2   = "#9CA3AF"   # muted text (subtitles, labels)
_GRID  = "#374151"

_MEDAL  = ["#F59E0B", "#94A3B8", "#B45309"]   # gold, silver, bronze
_BLUE   = "#3B82F6"
_GREEN  = "#10B981"
_RED    = "#EF4444"

# Per-agent semantic colors — used consistently across all charts
_AGENT_COLORS: dict[str, str] = {
    "selfish":          "#F59E0B",   # gold
    "neutral":          "#B45309",   # bronze
    "cooperative":      "#3B82F6",   # blue
    "academic":         "#8B5CF6",   # purple
    "tit_for_tat":      "#94A3B8",   # silver
    "tit_for_two_tats": "#64748B",
    "grim_trigger":     "#F97316",   # orange
    "pavlov":           "#EC4899",   # pink
    "random":           "#6B7280",   # gray
    "always_cooperate": "#10B981",   # green
    "always_defect":    "#EF4444",   # red
}

# ── Label helpers ──────────────────────────────────────────────────────────────
_DISPLAY: dict[str, str] = {
    "selfish":          "Claude  ·  selfish",
    "neutral":          "Claude  ·  neutral",
    "cooperative":      "Claude  ·  cooperative",
    "academic":         "Claude  ·  academic",
    "tit_for_tat":      "Tit-for-Tat",
    "tit_for_two_tats": "Tit-for-Two-Tats",
    "always_cooperate": "Always Cooperate",
    "always_defect":    "Always Defect",
    "grim_trigger":     "Grim Trigger",
    "pavlov":           "Pavlov",
    "random":           "Random (50/50)",
}

_MODEL_PREFIX: dict[str, str] = {
    "haiku":  "Haiku",
    "opus":   "Opus",
    "sonnet": "Claude",
}


def _short(name: str) -> str:
    """claude-sonnet-4-6:selfish → selfish  |  tit_for_tat → tit_for_tat"""
    return name.split(":")[-1] if ":" in name else name


def _display(name: str) -> str:
    short = _short(name)
    if short in _DISPLAY:
        return _DISPLAY[short]
    parts = name.split(":")
    if len(parts) == 2:
        for key, prefix in _MODEL_PREFIX.items():
            if key in parts[0]:
                return f"{prefix}  ·  {parts[1]}"
    return short.replace("_", " ").title()


def _agent_color(name: str) -> str:
    return _AGENT_COLORS.get(_short(name), _BLUE)


def _apply_dark(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes) -> None:  # type: ignore[name-defined]
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG2, labelsize=10)
    ax.xaxis.label.set_color(_FG2)
    ax.yaxis.label.set_color(_FG2)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_color(_BG2)


# ── score_bar ──────────────────────────────────────────────────────────────────
def score_bar(
    stats: list[AgentStats],
    out_path: Path,
    title: str = "Tournament Scoreboard",
    subtitle: str = "",
) -> None:
    """Dark horizontal scoreboard with medal colors."""
    n = len(stats)
    ordered = list(reversed(stats))   # best at top

    labels          = [_display(s.agent) for s in ordered]
    scores_per_round = [s.mean_score_per_round for s in ordered]
    coop_rates      = [s.cooperation_rate for s in ordered]

    bar_colors: list[str] = []
    for i in range(n):
        rank = n - 1 - i
        if rank < 3:
            bar_colors.append(_MEDAL[rank])
        elif rank == n - 1:
            bar_colors.append(_RED)
        else:
            bar_colors.append(_BLUE)

    fig, ax = plt.subplots(figsize=HERO_SIZE, facecolor=_BG)
    _apply_dark(fig, ax)

    y    = np.arange(n)
    x_max = max(scores_per_round)
    ax.barh(y, scores_per_round, color=bar_colors, height=0.55,
            edgecolor=_BG, linewidth=1.5)

    for val, coop, yi in zip(scores_per_round, coop_rates, y):
        ax.text(val + x_max * 0.015, yi,
                f"{val:.2f} pts/rd  ·  {coop:.0%} coop",
                va="center", ha="left", fontsize=9.5, color=_FG2)

    rank_strs = [f"#{n - i}" for i in range(n)]
    for i, (lbl, rnk) in enumerate(zip(labels, rank_strs)):
        bold = (n - 1 - i) < 3
        ax.text(-x_max * 0.015, i, f"{rnk}  {lbl}",
                va="center", ha="right", fontsize=11, color=_FG,
                fontweight="bold" if bold else "normal")

    for v in np.linspace(0, x_max, 5):
        ax.axvline(v, color=_GRID, linewidth=0.7, zorder=0)

    ax.set_xlim(-x_max * 0.36, x_max * 1.40)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    y0 = 0.95 if not subtitle else 0.97
    fig.text(0.5, y0, title, ha="center", va="top",
             fontsize=16, fontweight="bold", color=_FG)
    if subtitle:
        fig.text(0.5, 0.89, subtitle, ha="center", va="top",
                 fontsize=10, color=_FG2)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ── coop_over_time ─────────────────────────────────────────────────────────────
def coop_over_time(
    stats: list[AgentStats],
    out_path: Path,
    title: str = "Cooperation Rate Over Time",
    window: int = 5,
) -> None:
    fig, ax = plt.subplots(figsize=HERO_SIZE, facecolor=_BG)
    _apply_dark(fig, ax)

    for s in stats:
        if not s.coop_by_round:
            continue
        rates = np.array(s.coop_by_round)
        if len(rates) >= window:
            smoothed = np.convolve(rates, np.ones(window) / window, mode="valid")
            x = np.arange(window, len(rates) + 1)
        else:
            smoothed, x = rates, np.arange(1, len(rates) + 1)
        ax.plot(x, smoothed, label=_display(s.agent),
                color=_agent_color(s.agent), linewidth=2.5, alpha=0.9)

    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cooperation Rate", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color=_FG)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(alpha=0.2, color=_GRID)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    leg = ax.legend(fontsize=10, framealpha=0.3, facecolor=_BG2,
                    edgecolor=_GRID, labelcolor=_FG)
    for text in leg.get_texts():
        text.set_color(_FG)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ── heatmap ────────────────────────────────────────────────────────────────────
def heatmap(
    h2h: dict[tuple[str, str], tuple[float, float]],
    agents: list[str],
    out_path: Path,
    title: str = "Head-to-Head  ·  Avg Score per Round",
    rounds_per_match: int = 1,
) -> None:
    n = len(agents)
    grid = np.full((n, n), np.nan)
    for (p1, p2), (s1, _) in h2h.items():
        if p1 in agents and p2 in agents:
            grid[agents.index(p1), agents.index(p2)] = s1 / rounds_per_match

    fig, ax = plt.subplots(figsize=SQUARE_SIZE, facecolor=_BG)
    _apply_dark(fig, ax)

    # Custom colormap: dark-red → dark-navy midpoint → bright-green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "arena", ["#7F1D1D", "#374151", "#065F46"], N=256)
    cmap.set_bad(color=_BG2)   # NaN cells (same-agent diagonal)

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=5, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
    cbar.set_label("Avg Score / Round", fontsize=10, color=_FG2)
    cbar.ax.yaxis.set_tick_params(color=_FG2, labelcolor=_FG2)
    cbar.outline.set_edgecolor(_BG2)

    labels = [_display(a) for a in agents]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, color=_FG2)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9, color=_FG2)
    ax.set_xlabel("Opponent (P2)", fontsize=11, color=_FG2)
    ax.set_ylabel("Agent (P1)",    fontsize=11, color=_FG2)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=_FG)

    for i in range(n):
        for j in range(n):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=_FG, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ── behavioral ─────────────────────────────────────────────────────────────────
def behavioral(
    stats: list[AgentStats],
    out_path: Path,
    title: str = "Forgiveness vs Retaliation",
) -> None:
    fig, ax = plt.subplots(figsize=HERO_SIZE, facecolor=_BG)
    _apply_dark(fig, ax)

    x      = np.arange(len(stats))
    w      = 0.35
    labels = [_display(s.agent) for s in stats]
    fvals  = [s.forgiveness for s in stats]
    rvals  = [s.retaliation for s in stats]

    _FORGIVE_COLOR = "#38BDF8"   # sky blue
    _RETALIATE_COLOR = "#FB7185" # rose

    b1 = ax.bar(x - w / 2, fvals, w, label="Forgiveness",
                color=_FORGIVE_COLOR, edgecolor=_BG, linewidth=1)
    b2 = ax.bar(x + w / 2, rvals, w, label="Retaliation",
                color=_RETALIATE_COLOR, edgecolor=_BG, linewidth=1)

    ax.bar_label(b1, labels=[f"{v:.0%}" for v in fvals],
                 padding=4, fontsize=9, color=_FG)
    ax.bar_label(b2, labels=[f"{v:.0%}" for v in rvals],
                 padding=4, fontsize=9, color=_FG)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10, color=_FG2)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color=_FG)
    ax.grid(axis="y", alpha=0.2, color=_GRID)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    leg = ax.legend(fontsize=10, framealpha=0.3, facecolor=_BG2,
                    edgecolor=_GRID, labelcolor=_FG)
    for text in leg.get_texts():
        text.set_color(_FG)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ── coop_over_time_animated ────────────────────────────────────────────────────
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

    series: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for s in stats:
        if not s.coop_by_round:
            continue
        rates = np.array(s.coop_by_round)
        if len(rates) >= window:
            smoothed = np.convolve(rates, np.ones(window) / window, mode="valid")
            x = np.arange(window, len(rates) + 1)
        else:
            smoothed, x = rates, np.arange(1, len(rates) + 1)
        series.append((_display(s.agent), _agent_color(s.agent), x, smoothed))

    if not series:
        return out_path

    data_frames = max(len(x) for _, _, x, _ in series)
    n_frames    = data_frames + fps   # +1 s hold
    x_min = int(min(x[0]  for _, _, x, _ in series))
    x_max = int(max(x[-1] for _, _, x, _ in series))

    fig, ax = plt.subplots(figsize=HERO_SIZE, facecolor=_BG)
    _apply_dark(fig, ax)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Cooperation Rate", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color=_FG)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(alpha=0.2, color=_GRID)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    line_artists: list[tuple[object, np.ndarray, np.ndarray]] = []
    for label, color, x, y in series:
        (line,) = ax.plot([], [], label=label, color=color, linewidth=2.5, alpha=0.9)
        line_artists.append((line, x, y))

    leg = ax.legend(fontsize=10, framealpha=0.3, facecolor=_BG2,
                    edgecolor=_GRID, labelcolor=_FG,
                    loc="center left", bbox_to_anchor=(0.02, 0.5))
    for text in leg.get_texts():
        text.set_color(_FG)

    round_text = ax.text(0.97, 0.05, "", transform=ax.transAxes,
                         ha="right", va="bottom", fontsize=10, color=_FG2)
    fig.tight_layout()

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
