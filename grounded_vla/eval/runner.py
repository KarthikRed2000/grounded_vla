"""EvalRunner: ties agents, datasets, and metrics together.

Usage::

    runner = EvalRunner(agent, env_factory)
    result = runner.evaluate(dataset, limit=50)

Results are also serialized to disk (trajectories + aggregate numbers) so
the final report can load them directly without re-running anything.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from ..agents.base import Agent
from ..data.base import Dataset
from ..env import Environment, StaticQAEnv, TaskReplayEnv
from ..schemas import RunResult, Task, Trajectory
from ..utils.logging import get_logger
from .error_analysis import categorize_error
from .metrics import score_trajectory, step_efficiency

_log = get_logger(__name__)


def default_env_factory(task: Task) -> Environment:
    """Pick an environment based on the task source.

    ScienceQA -> StaticQAEnv; everything else -> TaskReplayEnv.
    """
    if task.source == "scienceqa":
        return StaticQAEnv()
    return TaskReplayEnv()


class EvalRunner:
    def __init__(
        self,
        agent: Agent,
        env_factory: Callable[[Task], Environment] = default_env_factory,
    ) -> None:
        self.agent = agent
        self.env_factory = env_factory

    def evaluate(
        self,
        dataset: Dataset,
        limit: Optional[int] = None,
        save_dir: Optional[Path | str] = None,
    ) -> RunResult:
        trajs: list[Trajectory] = []
        successes = 0
        step_counts: list[int] = []
        errors: Counter[str] = Counter()
        n = 0

        for i, task in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            env = self.env_factory(task)
            try:
                traj = self.agent.run(task, env)
            except Exception as e:  # never let one bad task kill the whole eval
                _log.exception("agent crashed on %s", task.task_id)
                traj = Trajectory(task_id=task.task_id)
                traj.truncated = True
                errors["agent_exception"] += 1
            score = score_trajectory(task, traj)
            if score.success:
                successes += 1
            step_counts.append(step_efficiency(traj))
            errors[categorize_error(task, traj, score.success).value] += 1
            trajs.append(traj)
            n += 1

        result = RunResult(
            agent_name=self.agent.name,
            dataset=dataset.name,
            n_tasks=n,
            task_completion_rate=(successes / n) if n else 0.0,
            mean_steps=(sum(step_counts) / len(step_counts)) if step_counts else 0.0,
            error_breakdown=dict(errors),
            trajectories=trajs,
        )

        if save_dir is not None:
            self._save(result, Path(save_dir))
        return result

    @staticmethod
    def _save(result: RunResult, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # Aggregate (small, always).
        (save_dir / "summary.json").write_text(
            json.dumps(
                {
                    "agent_name": result.agent_name,
                    "dataset": result.dataset,
                    "n_tasks": result.n_tasks,
                    "task_completion_rate": result.task_completion_rate,
                    "mean_steps": result.mean_steps,
                    "error_breakdown": result.error_breakdown,
                },
                indent=2,
            )
        )
        # Per-task trajectories (can be large; one JSON per task).
        tdir = save_dir / "trajectories"
        tdir.mkdir(exist_ok=True)
        for t in result.trajectories:
            (tdir / f"{t.task_id}.json").write_text(t.model_dump_json(indent=2))
