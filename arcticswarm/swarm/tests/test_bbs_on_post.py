# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
