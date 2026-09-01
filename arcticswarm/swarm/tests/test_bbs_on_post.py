"""Unit tests for the BBS observer hook."""

from __future__ import annotations

from arcticswarm.swarm.bbs import BBS, BBSMessage, CHANNEL_DISCOVERIES


def test_bbs_observer_fires_after_post():
    captured: list[BBSMessage] = []
    bbs = BBS(on_post=captured.append)
    msg = bbs.post(channel=CHANNEL_DISCOVERIES, author="A", content="hello")
    assert captured == [msg]


def test_bbs_set_on_post_replaces_observer():
    bbs = BBS()
    seen_a: list[BBSMessage] = []
    seen_b: list[BBSMessage] = []
    bbs.set_on_post(seen_a.append)
    bbs.post(channel="discussion", author="A", content="msg-1")
    bbs.set_on_post(seen_b.append)
    bbs.post(channel="discussion", author="A", content="msg-2")
    assert len(seen_a) == 1
    assert len(seen_b) == 1


def test_bbs_observer_exception_does_not_break_post():
    def boom(_msg: BBSMessage) -> None:
        raise RuntimeError("observer broken")

    bbs = BBS(on_post=boom)
    msg = bbs.post(channel=CHANNEL_DISCOVERIES, author="A", content="hello")
    assert msg.content == "hello"
    # Subsequent posts also work.
    bbs.post(channel=CHANNEL_DISCOVERIES, author="B", content="again")
    assert len(bbs.read_all()) == 2


def test_bbs_observer_can_be_cleared():
    captured: list[BBSMessage] = []
    bbs = BBS(on_post=captured.append)
    bbs.set_on_post(None)
    bbs.post(channel="discussion", author="A", content="silent")
    assert captured == []
