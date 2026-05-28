from agent_arena.agents.models import CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_7, CLAUDE_SONNET_4_6

# Maps preset name → list of model IDs to include
TOURNAMENT_PRESETS: dict[str, list[str]] = {
    "claude-trio": [CLAUDE_OPUS_4_7, CLAUDE_SONNET_4_6, CLAUDE_HAIKU_4_5],
    "sonnet-only": [CLAUDE_SONNET_4_6],
    "haiku-only": [CLAUDE_HAIKU_4_5],
    "sonnet-haiku": [CLAUDE_SONNET_4_6, CLAUDE_HAIKU_4_5],
}

# Short model label used in agent names
MODEL_LABELS: dict[str, str] = {
    CLAUDE_OPUS_4_7: "opus",
    CLAUDE_SONNET_4_6: "sonnet",
    CLAUDE_HAIKU_4_5: "haiku",
}

DEFAULT_RULE_BOTS = ["tit-for-tat", "always-cooperate", "always-defect"]
