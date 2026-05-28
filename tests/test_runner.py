from agent_arena.agents.rule_based import AlwaysCooperate, AlwaysDefect, TitForTat
from agent_arena.games.prisoners_dilemma import IteratedPrisonersDilemma
from agent_arena.tournaments.runner import MatchRunner


def _runner(p1: object, p2: object, rounds: int = 5) -> MatchRunner:
    return MatchRunner(
        game=IteratedPrisonersDilemma(max_rounds=rounds),
        p1=p1,  # type: ignore[arg-type]
        p2=p2,  # type: ignore[arg-type]
    )


def test_match_runs_correct_number_of_rounds() -> None:
    result = _runner(AlwaysCooperate(), AlwaysDefect()).run()
    assert result.total_rounds == 5
    assert len(result.rounds) == 5


def test_cooperate_vs_defect_scores() -> None:
    # P1 always cooperates (S=0 each round), P2 always defects (T=5 each round)
    result = _runner(AlwaysCooperate(), AlwaysDefect(), rounds=10).run()
    assert result.p1_total_score == 0.0
    assert result.p2_total_score == 50.0


def test_mutual_cooperation_scores() -> None:
    result = _runner(AlwaysCooperate(), AlwaysCooperate(), rounds=10).run()
    assert result.p1_total_score == 30.0  # 10 × R=3
    assert result.p2_total_score == 30.0


def test_mutual_defection_scores() -> None:
    result = _runner(AlwaysDefect(), AlwaysDefect(), rounds=10).run()
    assert result.p1_total_score == 10.0  # 10 × P=1
    assert result.p2_total_score == 10.0


def test_tft_vs_always_cooperate_full_cooperation() -> None:
    result = _runner(TitForTat(), AlwaysCooperate(), rounds=10).run()
    assert result.p1_total_score == 30.0
    assert result.p2_total_score == 30.0


def test_tft_vs_always_defect() -> None:
    # TfT opens cooperate (round 1: S=0), then defects every subsequent round (P=1 each)
    result = _runner(TitForTat(), AlwaysDefect(), rounds=5).run()
    # Round 1: TfT=cooperate, AD=defect → TfT gets S=0
    # Rounds 2-5: TfT=defect, AD=defect → TfT gets P=1 each (4 rounds)
    assert result.p1_total_score == 4.0
    assert result.p2_total_score == 5.0 + 4.0  # T=5 + 4×P=1


def test_result_metadata_fields() -> None:
    result = _runner(AlwaysCooperate(), TitForTat(), rounds=3).run()
    assert result.match_id != ""
    assert result.game == "iterated_prisoners_dilemma"
    assert result.p1_name == "always_cooperate"
    assert result.p2_name == "tit_for_tat"
    assert result.started_at < result.ended_at


def test_state_flip_correctness() -> None:
    # TfT as P2 should behave symmetrically: it sees a flipped state,
    # so it mirrors what it perceives as the opponent (which is P1 = AlwaysCooperate).
    result = _runner(AlwaysCooperate(), TitForTat(), rounds=5).run()
    # TfT as P2 opens cooperate, then mirrors P1's cooperate → all C,C rounds
    for log in result.rounds:
        assert log.p1_move == "cooperate"
        assert log.p2_move == "cooperate"
