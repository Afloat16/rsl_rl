# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Logger."""

from __future__ import annotations

import pathlib
import torch
from torch.utils.tensorboard import SummaryWriter
from unittest.mock import Mock

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rsl_rl.utils import LogWriter, NeptuneLogWriter, WandbLogWriter
from rsl_rl.utils.logger import Logger


class RecordingWriter(LogWriter):
    """LogWriter that records scalar tags and uploaded videos."""

    def __init__(self) -> None:  # noqa: D107
        self.tags: list[str] = []
        self.scalars: dict[str, float] = {}
        self.videos: list[tuple[pathlib.Path, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:  # noqa: D102
        self.tags.append(tag)
        self.scalars[tag] = scalar_value

    def save_video(self, video: pathlib.Path, it: int) -> None:  # noqa: D102
        self.videos.append((video, it))


def _make_logger(log_dir: pathlib.Path, num_envs: int = 1) -> Logger:
    """Construct a Logger that writes to a RecordingWriter."""
    logger = Logger(
        log_dir=str(log_dir),
        cfg={"num_steps_per_env": 1, "algorithm": {}},
        env_cfg={},
        num_envs=num_envs,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    logger.writer = RecordingWriter()
    return logger


def _log(
    logger: Logger,
    it: int,
    loss_dict: dict[str, float] | None = None,
    stats_dict: dict[str, float] | None = None,
) -> None:
    """Log one iteration with placeholder metrics."""
    logger.log(
        it=it,
        start_it=0,
        total_it=it + 1,
        collect_time=0.1,
        learn_time=0.1,
        loss_dict=loss_dict or {},
        stats_dict=stats_dict or {},
    )


@pytest.mark.parametrize("method", ["stop_logging_writer", "abort_logging_writer"])
def test_tensorboard_writer_closes_and_can_restart(tmp_path: pathlib.Path, method: str) -> None:
    """Each finished logging run flushes its scalars and joins its local event worker."""
    logger = _make_logger(tmp_path)
    logger.git_status_repos = []
    for step in range(2):
        logger.init_logging_writer()
        writer = logger.writer
        worker = writer.file_writer.event_writer._async_writer._worker
        try:
            _log(logger, step, {"actor": float(step + 1)})
            getattr(logger, method)()
            assert not worker.is_alive()
            assert writer.file_writer is None
        finally:
            writer.close()

    events = EventAccumulator(str(tmp_path)).Reload().Scalars("Loss/actor")
    assert [(event.step, event.value) for event in events] == [(0, 1.0), (1, 2.0)]


@pytest.mark.parametrize("writer_class", [WandbLogWriter, NeptuneLogWriter])
@pytest.mark.parametrize("method", ["stop_logging_writer", "abort_logging_writer"])
@pytest.mark.parametrize("remote_failure", [False, True])
def test_external_writer_also_closes_local_events(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, writer_class: type, method: str, remote_failure: bool
) -> None:
    """Local events are finalized even if the remote run's finalization raises."""
    writer = writer_class.__new__(writer_class)
    SummaryWriter.__init__(writer, log_dir=str(tmp_path))
    remote_stop = Mock(side_effect=RuntimeError("remote finish failed") if remote_failure else None)
    if writer_class is WandbLogWriter:
        monkeypatch.setattr("rsl_rl.utils.wandb_log_writer.wandb", Mock(finish=remote_stop))
    else:
        writer.run = Mock(stop=remote_stop)
    logger = _make_logger(tmp_path)
    logger.writer = writer
    worker = writer.file_writer.event_writer._async_writer._worker
    try:
        SummaryWriter.add_scalar(writer, "last/value", 3.0, 7)
        if remote_failure:
            with pytest.raises(RuntimeError, match="remote finish failed"):
                getattr(logger, method)()
        else:
            getattr(logger, method)()
        remote_stop.assert_called_once()
        assert not worker.is_alive()
        events = EventAccumulator(str(tmp_path)).Reload().Scalars("last/value")
        assert [(event.step, event.value) for event in events] == [(7, 3.0)]
    finally:
        writer.close()


@pytest.mark.parametrize(
    "writer_class,missing_token", [(WandbLogWriter, False), (NeptuneLogWriter, False), (NeptuneLogWriter, True)]
)
def test_failed_external_writer_construction_closes_local_events(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, writer_class: type, missing_token: bool
) -> None:
    """A failed remote initialization must not leave a local event writer running."""
    failure = RuntimeError("remote initialization failed")
    monkeypatch.setenv("NEPTUNE_API_TOKEN", "test-token")
    monkeypatch.setenv("NEPTUNE_USERNAME", "test-user")
    if missing_token:
        monkeypatch.delenv("NEPTUNE_API_TOKEN")
    monkeypatch.setattr("rsl_rl.utils.wandb_log_writer.wandb", Mock(init=Mock(side_effect=failure)))
    monkeypatch.setattr("rsl_rl.utils.neptune_log_writer.neptune", Mock(init_run=Mock(side_effect=failure)))
    writer = writer_class.__new__(writer_class)
    workers = []
    original_init = SummaryWriter.__init__

    def record_init(instance: SummaryWriter, *args: object, **kwargs: object) -> None:
        original_init(instance, *args, **kwargs)
        workers.append(instance.file_writer.event_writer._async_writer._worker)

    monkeypatch.setattr(SummaryWriter, "__init__", record_init)
    try:
        with pytest.raises(KeyError if missing_token else RuntimeError) as raised:
            writer.__init__(log_dir=str(tmp_path), project_name="test-project")
        if not missing_token:
            assert raised.value is failure
        assert all(not worker.is_alive() for worker in workers)
    finally:
        if hasattr(writer, "all_writers"):
            SummaryWriter.close(writer)


@pytest.mark.parametrize("phase", ["code_state", "store_config"])
@pytest.mark.parametrize("abort_failure", [False, True])
def test_failed_logging_setup_closes_created_writer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, phase: str, abort_failure: bool
) -> None:
    """Setup errors after construction close the writer before training can begin."""
    writer = WandbLogWriter.__new__(WandbLogWriter)
    SummaryWriter.__init__(writer, log_dir=str(tmp_path))
    remote_stop = Mock(side_effect=RuntimeError("remote abort failed") if abort_failure else None)
    monkeypatch.setattr("rsl_rl.utils.wandb_log_writer.wandb", Mock(finish=remote_stop))
    logger = _make_logger(tmp_path)
    logger.cfg["logger"] = {"class_name": "WandbLogWriter", "project_name": "test-project"}
    monkeypatch.setattr("rsl_rl.utils.logger.resolve_class", lambda cfg: (lambda **kwargs: writer, {}))
    failure = RuntimeError("logging setup failed")
    logger._store_code_state = Mock(side_effect=failure if phase == "code_state" else None, return_value=[])
    writer.store_config = Mock(side_effect=failure if phase == "store_config" else None)
    worker = writer.file_writer.event_writer._async_writer._worker
    try:
        with pytest.raises(RuntimeError) as raised:
            logger.init_logging_writer()
        assert raised.value is failure
        assert not worker.is_alive()
        remote_stop.assert_called_once_with(exit_code=1)
    finally:
        writer.close()


def test_statistics_are_separate_from_losses(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Statistics retain their tags and console labels without being treated as optimization losses."""
    logger = _make_logger(tmp_path)
    stats_dict = {
        "Model/mean_policy_entropy": 0.2,
        "Model/mean_critic_q_value": 3.0,
        "Optimizer/learning_rate": 1e-3,
        "Model/mean_policy_std": 1.0,
        "Rnd/rnd_weight": 0.5,
        "Custom/statistic": 4.0,
        "statistic": 5.0,
        "Custom/nested/statistic": 6.0,
    }
    _log(logger, 0, {"actor": 1.0}, stats_dict)

    assert {
        "Loss/actor",
        "Performance/steps_per_second",
        "Performance/collection_time",
        "Performance/learning_time",
        *stats_dict,
    } <= set(logger.writer.tags)
    assert not any(tag.startswith("Stats/") for tag in logger.writer.tags)
    assert not {"Loss/entropy", "Loss/q_value"} & set(logger.writer.tags)
    output = capsys.readouterr().out
    assert "Mean actor loss:" in output
    for name in (
        "Mean policy entropy",
        "Mean critic q value",
        "Learning rate",
        "Mean policy std",
        "Rnd weight",
        "Statistic",
        "Nested/statistic",
    ):
        assert f"{name}:" in output
    for category in ("Model/", "Loss/", "Optimizer/", "Rnd/", "Custom/"):
        assert category not in output
    assert "entropy loss:" not in output
    assert "q_value loss:" not in output


class TestVideoUpload:
    """Tests for uploading videos from the log directory."""

    def test_each_video_is_uploaded_once(self, tmp_path: pathlib.Path) -> None:
        """Each video should be uploaded once, at the first iteration after it appears."""
        first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
        first.touch()
        logger = _make_logger(tmp_path)

        _log(logger, 0)
        second.touch()
        _log(logger, 1)
        _log(logger, 2)

        assert logger.writer.videos == [(first, 0), (second, 1)]


def test_step_metrics_accumulate_over_completed_episodes(tmp_path: pathlib.Path) -> None:
    """Custom metrics preserve partial episodes across logging calls and reset environments independently."""
    logger = _make_logger(tmp_path, num_envs=2)
    with torch.inference_mode():
        logger.process_env_step(
            torch.tensor([False, True]),
            {},
            {"Training/mean_reward": torch.tensor([1.0, 10.0]), "Custom/total": torch.tensor([2.0, 20.0])},
        )
    _log(logger, 0)
    assert logger.writer.scalars["Training/mean_reward"] == 10.0
    assert logger.writer.scalars["Custom/total"] == 20.0

    logger.process_env_step(
        torch.tensor([False, False]),
        {},
        {"Training/mean_reward": torch.tensor([3.0, 30.0]), "Custom/total": torch.tensor([4.0, 40.0])},
    )
    _log(logger, 1)
    assert logger.writer.scalars["Custom/total"] == 20.0

    logger.process_env_step(
        torch.tensor([True, True]),
        {},
        {"Training/mean_reward": torch.tensor([5.0, 50.0]), "Custom/total": torch.tensor([6.0, 60.0])},
    )
    _log(logger, 2)
    assert logger.writer.scalars["Training/mean_reward"] == 33.0
    assert logger.writer.scalars["Custom/total"] == 44.0
    assert logger.writer.scalars["Training/mean_episode_length"] == 2.0
    torch.testing.assert_close(logger.cur_step_sums["Custom/total"], torch.zeros(2))


class TestTimeSeries:
    """Tests for the series logged against wall-clock time."""

    def test_log_writer_receives_no_time_series(self, tmp_path: pathlib.Path) -> None:
        """A LogWriter should only receive series indexed by iteration."""
        logger = _make_logger(tmp_path)
        logger.process_env_step(torch.ones(1), {}, {"Training/mean_reward": torch.ones(1)})

        _log(logger, 0)

        assert "Training/mean_reward" in logger.writer.tags
        assert not any(tag.endswith("/time") for tag in logger.writer.tags)
