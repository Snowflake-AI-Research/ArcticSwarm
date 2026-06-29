"""Tests for ``BashTool`` CWD isolation.

Bug symptom: during image-heavy HLE runs the agent invoked ``tesseract``,
``convert``, ``wget`` etc. via ``bash``, and every output file landed in
whatever CWD ``arcticswarm-eval`` was launched from — typically the user's
home directory — producing dozens of stray ``*.jpg`` / ``*.tsv`` / ``*.txt``
files alongside their code.

Fix: ``BashTool`` now creates a per-instance scratch directory lazily and
uses it as ``cwd`` by default.  Explicit ``working_directory`` arguments
still override.
"""
from __future__ import annotations

import os


def test_bash_default_cwd_is_isolated_scratch_dir(tmp_path, monkeypatch):
    """Without ``working_directory``, the command's CWD must be an isolated
    temp directory — never the process-launch CWD."""
    from arcticswarm.tools.bash import BashTool

    # Pretend the user launched arcticswarm-eval from this directory.
    launch_cwd = tmp_path / "launch"
    launch_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)

    tool = BashTool()
    try:
        res = tool.execute(command="pwd")
        assert not res.is_error, res.error
        observed = (res.output or "").strip()
        # Must NOT equal the launch dir.
        assert os.path.realpath(observed) != os.path.realpath(str(launch_cwd)), (
            f"BashTool leaked the launch CWD {launch_cwd} into the shell: "
            f"{observed!r}.  Without ``working_directory`` it should default "
            "to an isolated per-agent scratch dir."
        )
        # And must be the scratch dir we just created.
        assert tool._scratch_dir is not None
        assert os.path.realpath(observed) == os.path.realpath(tool._scratch_dir)
    finally:
        tool.close()


def test_bash_scratch_dir_persists_across_calls(tmp_path, monkeypatch):
    """A file written by one bash call must be visible to the next one so
    multi-step workflows (download -> crop -> OCR) keep working."""
    from arcticswarm.tools.bash import BashTool

    monkeypatch.chdir(tmp_path)
    tool = BashTool()
    try:
        r1 = tool.execute(command="echo hello > marker.txt && cat marker.txt")
        assert not r1.is_error, r1.error
        assert "hello" in (r1.output or "")

        r2 = tool.execute(command="cat marker.txt")
        assert not r2.is_error, r2.error
        assert "hello" in (r2.output or ""), (
            "Second bash call lost the first call's file — scratch dir "
            "must persist across calls within one agent's lifetime."
        )
    finally:
        tool.close()


def test_bash_explicit_working_directory_still_honored(tmp_path, monkeypatch):
    """If the caller passes ``working_directory``, that wins over the
    scratch dir (same behaviour as before the fix)."""
    from arcticswarm.tools.bash import BashTool

    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "custom"
    custom.mkdir()

    tool = BashTool()
    try:
        res = tool.execute(command="pwd", working_directory=str(custom))
        assert not res.is_error, res.error
        observed = (res.output or "").strip()
        assert os.path.realpath(observed) == os.path.realpath(str(custom))
    finally:
        tool.close()


def test_bash_close_removes_scratch_dir(tmp_path, monkeypatch):
    """``close()`` must rm the scratch dir so repeated eval runs do not
    accumulate terabytes of ``/tmp/arcticswarm_bash_*`` directories."""
    from arcticswarm.tools.bash import BashTool

    monkeypatch.chdir(tmp_path)
    tool = BashTool()
    tool.execute(command="true")
    scratch = tool._scratch_dir
    assert scratch is not None
    assert os.path.isdir(scratch)

    tool.close()
    assert not os.path.exists(scratch), (
        f"close() left the scratch dir behind at {scratch}"
    )
    assert tool._scratch_dir is None


def test_bash_no_stray_files_in_launch_cwd(tmp_path, monkeypatch):
    """End-to-end: simulate an OCR-style workflow and confirm ZERO files are
    written to the launch CWD (the bug that motivated this fix)."""
    from arcticswarm.tools.bash import BashTool

    launch_cwd = tmp_path / "launch"
    launch_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)

    tool = BashTool()
    try:
        # Ten commands that create artefacts, mimicking the image/OCR calls
        # observed in the Duo GPT-low image-input runs.
        for i in range(10):
            r = tool.execute(command=f"echo data > crop_{i}.jpg && echo tsv > out_{i}.tsv")
            assert not r.is_error, r.error
    finally:
        tool.close()

    leaked = [p for p in launch_cwd.iterdir()]
    assert leaked == [], (
        f"BashTool leaked files into the launch CWD: {[p.name for p in leaked]}. "
        "Every artefact must stay in the per-agent scratch dir."
    )
