"""Tests for the ``swarm.disable_auditor`` flag.

The flag suppresses the always-on dedicated auditor subagent (dynamic/BBS
mode) and forces the reviewer-diversity gate's dedicated side off, leaving
only builder subagents. It is unsupported in duo mode (leader + auditor by
construction), which must raise rather than silently no-op.
"""

import pytest

from arcticswarm.run_config import RunConfig
from arcticswarm.swarm.orchestrator_duo import DuoMixin


class TestDisableAuditorConfig:
    def test_default_is_false(self):
        rc = RunConfig()
        assert rc.swarm.disable_auditor is False
        assert rc.to_arcticswarm_config().disable_auditor is False

    def test_propagates_to_runtime_config(self):
        rc = RunConfig()
        rc.swarm.disable_auditor = True
        assert rc.to_arcticswarm_config().disable_auditor is True


class TestDisableAuditorDuoGuard:
    def _stub(self, disable_auditor: bool):
        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.disable_auditor = disable_auditor
        obj = object.__new__(DuoMixin)
        obj.config = cfg
        return obj

    def test_duo_mode_rejects_disable_auditor(self):
        obj = self._stub(disable_auditor=True)
        with pytest.raises(ValueError, match="duo mode"):
            obj._run_duo_turn("question")

    def test_duo_mode_allows_when_flag_off(self):
        # With the flag off the guard must not fire; the method proceeds past
        # the guard (and fails later for unrelated reasons on this bare stub,
        # which is NOT a ValueError about duo mode).
        obj = self._stub(disable_auditor=False)
        with pytest.raises(Exception) as ei:
            obj._run_duo_turn("question")
        assert not (
            isinstance(ei.value, ValueError) and "duo mode" in str(ei.value)
        )
