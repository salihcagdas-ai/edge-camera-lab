from pathlib import Path

import pytest

from edge.demo import run_demo


@pytest.mark.asyncio
async def test_demo_runs_offline_and_never_reuses_an_existing_output(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    output = tmp_path / "new-demo"
    summary = await run_demo(output)
    assert summary["event_counts"] == {"known_person": 1, "unknown_person": 1}
    assert summary["model_inference"] is False
    assert (output / "synthetic-preview.png").is_file()
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        await run_demo(output)
    assert (output / "summary.json").read_bytes() == before
