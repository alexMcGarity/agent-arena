CLAUDE_OPUS_4_7 = "claude-opus-4-7"
CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"

DEFAULT_MODEL = CLAUDE_SONNET_4_6

# USD per million tokens
PRICING: dict[str, dict[str, float]] = {
    CLAUDE_OPUS_4_7: {
        "input": 15.0,
        "output": 75.0,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    CLAUDE_SONNET_4_6: {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    CLAUDE_HAIKU_4_5: {
        "input": 0.80,
        "output": 4.0,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
}
