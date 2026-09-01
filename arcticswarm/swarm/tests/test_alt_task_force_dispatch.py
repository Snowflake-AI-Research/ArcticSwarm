"""Unit test for alt_task_force_dispatch — the candidate-emergence rival
sweep must ACTIVELY dispatch a worker (spawn_or_assign), not just add_task, when
the flag is on. Default-off must preserve the add_task-only behavior.
"""
from types import SimpleNamespace
from arcticswarm.swarm.bbs import BBS
from arcticswarm.swarm.answer_verification import wire_candidate_emergence_hook


class _Board:
    def __init__(self):
        self.added = []

    def add_task(self, spec):
        self.added.append(spec)


class _Ctx:
    def __init__(self):
        self.dispatched = []

    def spawn_or_assign(self, spec):
        self.dispatched.append(spec)
        return "Worker1"


def _run(flag):
    bbs = BBS()
    board = _Board()
    ctx = _Ctx()
    orch = SimpleNamespace(
        config=SimpleNamespace(
            candidate_emergence_min_chars=60,
            candidate_emergence_max_turns=8,
            alt_task_force_dispatch=flag,
        ),
        _rival_sweep_fired=False,
        _ctx=ctx,
    )
    wire_candidate_emergence_hook(orch, bbs=bbs, task_board=board, question_text="Find X.")
    # a qualifying candidate post (>=60 chars) on a candidate channel
    bbs.post(channel="key-findings", author="James",
             content="Answer: The Redmond Monument in Wexford, erected 1867, restored 2007, matches all constraints.")
    return board, ctx


def test_flag_on_dispatches():
    board, ctx = _run(True)
    assert len(board.added) == 1, "task still registered on board"
    assert len(ctx.dispatched) == 1, "flag on must dispatch a worker via spawn_or_assign"
    assert ctx.dispatched[0].name == "alternative-candidate-sweep"


def test_flag_off_is_board_only():
    board, ctx = _run(False)
    assert len(board.added) == 1
    assert len(ctx.dispatched) == 0, "default-off must NOT dispatch (preserve current behavior)"


if __name__ == "__main__":
    test_flag_on_dispatches()
    test_flag_off_is_board_only()
    print("alt_task_force_dispatch tests passed")
