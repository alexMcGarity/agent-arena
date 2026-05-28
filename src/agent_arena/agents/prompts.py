from enum import StrEnum


class Persona(StrEnum):
    NEUTRAL = "neutral"
    COOPERATIVE = "cooperative"
    SELFISH = "selfish"
    ACADEMIC = "academic"
    COT = "cot"


PERSONA_PREAMBLES: dict[Persona, str] = {
    Persona.NEUTRAL: "",
    Persona.COOPERATIVE: (
        "You value collaboration and long-term mutual benefit. "
        "You believe that working together leads to better outcomes for everyone."
    ),
    Persona.SELFISH: (
        "You value your own score above all else. "
        "Your sole objective is to maximize your personal total score."
    ),
    Persona.ACADEMIC: (
        "You are a rational game theorist. "
        "You analyze situations logically and choose the move that maximizes expected payoff."
    ),
    Persona.COT: (
        "Think step-by-step about your strategy in <thinking> tags before deciding your move."
    ),
}

FORMAT_INSTRUCTION = (
    '\n\nRespond with valid JSON only, on a single line: '
    '{"move": "<legal_move>", "reasoning": "<one sentence>"}'
)
