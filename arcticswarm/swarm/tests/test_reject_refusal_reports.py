"""Unit tests for the anti-give-up + canonical-answer report behavior
(``reject_refusal_reports`` / SendReportTool refusal bounce).

Covers the give-up detector and the SendReportTool bounce: default-off is a
no-op, a give-up FINAL ANSWER is bounced for retry, a committed answer passes
through, the bounce is bounded, the prompt guidance is gated, and the bounce is
ordered after the strict-DM drain (duo).
"""
from arcticswarm.swarm.empty_answer_recovery import (
    extract_final_answer,
    final_answer_is_giveup,
)
from arcticswarm.swarm.tools import SendReportTool


# ---- detector helpers -------------------------------------------------------

def test_extract_final_answer():
    assert extract_final_answer(
        "blah\nConfidence: 90\nFINAL ANSWER: The Redmond Monument"
    ) == "The Redmond Monument"
    # tolerate markdown bolding on both sides of the marker
    assert extract_final_answer(
        "**FINAL ANSWER:** Muhammad Qavi Khan"
    ) == "Muhammad Qavi Khan"
    # multi-line answer is preserved
    assert extract_final_answer(
        "FINAL ANSWER: line one\nline two"
    ) == "line one\nline two"
    # no marker -> empty
    assert extract_final_answer("just some prose, no marker") == ""


def test_final_answer_is_giveup_true():
    for r in (
        "Confidence: 0\nFINAL ANSWER: No candidate satisfies all constraints",
        "Confidence: 10\nFINAL ANSWER: Unable to determine - no person found",
        "FINAL ANSWER: No monument has been definitively identified",
        "FINAL ANSWER: insufficient evidence to conclude",
        "**FINAL ANSWER:** could not be identified",
        "FINAL ANSWER:",                      # present but empty
        "",                                   # empty report
        "we were unable to determine the answer",  # no marker -> scan body
    ):
        assert final_answer_is_giveup(r), r


def test_final_answer_is_giveup_false():
    # committed answers, incl. short ones (min_len regression guard)
    for r in (
        "Confidence: 90\nFINAL ANSWER: The Redmond Monument",
        "FINAL ANSWER: Tokyo",
        "FINAL ANSWER: Richard Andrew Palethorpe-Todd",
        # give-up phrasing in the BODY but a real committed FINAL ANSWER
        "We initially could not determine the city, but resolved it.\n"
        "Confidence: 80\nFINAL ANSWER: Muhammad Qavi Khan",
    ):
        assert not final_answer_is_giveup(r), r


# ---- SendReportTool bounce --------------------------------------------------

_GIVEUP = "Confidence: 0\nFINAL ANSWER: no candidate satisfies all constraints"
_COMMIT = "Confidence: 90\nFINAL ANSWER: The Redmond Monument"


def test_default_off_is_noop():
    """reject_refusal defaults False -> a give-up report is captured as-is."""
    t = SendReportTool(has_web_search=True)
    res = t.execute(report=_GIVEUP)
    assert not res.is_error
    assert t.captured_report is not None


def test_bounce_on_giveup():
    t = SendReportTool(has_web_search=True, reject_refusal=True)
    res = t.execute(report=_GIVEUP)
    assert res.is_error
    assert t.captured_report is None
    assert t._refusal_bounce_count == 1


def test_commit_passes_through():
    t = SendReportTool(has_web_search=True, reject_refusal=True)
    res = t.execute(report=_COMMIT)
    assert not res.is_error
    assert t.captured_report is not None


def test_bounce_is_bounded():
    """After max_refusal_bounces, a give-up is accepted (no infinite loop)."""
    t = SendReportTool(has_web_search=True, reject_refusal=True,
                       max_refusal_bounces=2)
    errors = [t.execute(report=_GIVEUP).is_error for _ in range(3)]
    assert errors == [True, True, False]
    assert t.captured_report is not None  # 3rd attempt accepted
    assert t._refusal_bounce_count == 2


def test_prompt_guidance_gated():
    on = SendReportTool(has_web_search=True, reject_refusal=True,
                        question="What is the actor's full name?")
    off = SendReportTool(has_web_search=True)
    for tool, present in ((on, True), (off, False)):
        desc = tool.description
        rdesc = tool.parameters_schema()["properties"]["report"]["description"]
        assert ("canonical" in desc.lower()) is present
        assert ("canonical" in rdesc.lower()) is present
    # the question-format clause only appears when a question is supplied
    assert "requested form" in on.description


def test_bounce_ordered_after_dm_drain():
    """A pending substantive teammate DM is surfaced before the refusal bounce."""
    from arcticswarm.swarm.mailbox import Mailbox

    mailbox = Mailbox()
    mailbox.register("leader")
    mailbox.register("auditor")
    # send a substantive finding to the leader from a teammate (default
    # message_type='peer_message' is substantive — not in the drain skip-set)
    mailbox.send(
        from_agent="auditor", to_agent="leader",
        content="New finding: the answer is X.",
    )
    t = SendReportTool(
        has_web_search=True, reject_refusal=True,
        mailbox=mailbox, agent_name="leader", strict_dm_drain=True,
    )
    res = t.execute(report=_GIVEUP)
    assert res.is_error
    # the DM-drain error fires first (mentions teammate findings), NOT the
    # refusal-bounce message; and the refusal counter has not incremented.
    assert "teammate" in (res.error or "").lower()
    assert t._refusal_bounce_count == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("all reject_refusal_reports tests passed")
