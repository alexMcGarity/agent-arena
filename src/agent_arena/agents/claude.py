import json
import time
from dataclasses import dataclass, field

import anthropic

from agent_arena.agents.base import MoveResult
from agent_arena.agents.models import CLAUDE_SONNET_4_6
from agent_arena.agents.prompts import FORMAT_INSTRUCTION, PERSONA_PREAMBLES, Persona
from agent_arena.games.base import GameState


@dataclass
class ClaudeAgent:
    model: str = CLAUDE_SONNET_4_6
    persona: Persona = Persona.NEUTRAL
    memory: str = "full"  # "full" | "last10" | "score_only" | "none"
    temperature: float = 1.0
    _game_description: str = field(default="", init=False, repr=False)
    _client: anthropic.Anthropic | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"{self.model}:{self.persona}"

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def reset(self, game_description: str = "") -> None:
        self._game_description = game_description

    def choose_move(self, state: GameState, legal_moves: list[str]) -> MoveResult:
        system = self._build_system(self._game_description)
        user_msg = self._build_user_message(state)

        t0 = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=self.temperature,
            system=system,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": user_msg}],
        )
        latency_ms = (time.monotonic() - t0) * 1000

        raw = response.content[0].text.strip() if response.content else ""
        move, reasoning = self._parse_response(raw, legal_moves)

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0

        return MoveResult(
            move=move,
            reasoning=reasoning,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=latency_ms,
        )

    def _build_system(self, game_description: str) -> list[dict[str, object]]:
        preamble = PERSONA_PREAMBLES[self.persona]
        text = (preamble + "\n\n" + game_description).strip() if preamble else game_description
        text += FORMAT_INSTRUCTION
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]

    def _build_user_message(self, state: GameState) -> str:
        prompt_suffix = f"\n\nRound {state.round_num + 1} of {state.max_rounds}. What is your move?"

        if self.memory == "none" or not state.history:
            return f"Round {state.round_num + 1} of {state.max_rounds}. What is your move?"

        if self.memory == "score_only":
            p1_total = sum(r.p1_score for r in state.history)
            p2_total = sum(r.p2_score for r in state.history)
            return (
                f"After {len(state.history)} rounds — your score: {p1_total:.0f}, "
                f"opponent score: {p2_total:.0f}."
                + prompt_suffix
            )

        start_idx = max(0, len(state.history) - 10) if self.memory == "last10" else 0
        history_slice = state.history[start_idx:]
        lines = [
            f"  R{start_idx + i + 1}: you={r.p1_move}, opp={r.p2_move}"
            f" → +{r.p1_score:.0f}/+{r.p2_score:.0f}"
            for i, r in enumerate(history_slice)
        ]
        return "History:\n" + "\n".join(lines) + prompt_suffix

    def _parse_response(self, raw: str, legal_moves: list[str]) -> tuple[str, str | None]:
        text = raw

        # Strip <thinking> block when COT persona is used
        if "<thinking>" in text:
            end = text.rfind("</thinking>")
            if end != -1:
                text = text[end + len("</thinking>"):].strip()

        # Try JSON extraction
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end])
                move = str(data.get("move", "")).lower().strip()
                if move in legal_moves:
                    return move, data.get("reasoning")
            except (json.JSONDecodeError, AttributeError):
                pass

        # Keyword scan fallback
        lower = text.lower()
        for move in legal_moves:
            if move in lower:
                return move, text

        return legal_moves[0], f"[parse_error] {raw!r}"
