#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import time
from pathlib import Path
from typing import Any

import ray
from ray.exceptions import GetTimeoutError, RayActorError

from phi_agents.rl.eval import EvalWorker
from phi_agents.rl.train import RLOOTrainer
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


class Callback:
    def __init__(self, algo: RLOOTrainer) -> None:
        self._algo = algo

    def before_iteration(self, iteration: int, last_checkpoint_local_path: Path | None) -> None:
        pass

    def before_new_rollouts(self) -> None:
        """
        Called right before new RL rollouts are requested,
        and thus right before the LoRA adapter is updated on the inference server.
        """
        pass

    def new_rollouts_ready(self) -> bool:
        """
        Non-blocking poll for whether before_new_rollouts()'s work has completed.
        Called in a loop (with a NCCL barrier in between iterations) rather than
        blocking directly, so a slow/stuck task on rank 0 can't trip other ranks'
        collective watchdog. Return True once it's safe to swap in the new LoRA adapter.
        """
        return True

    def after_iteration(self, iteration: int) -> None:
        pass

    def shutdown(self) -> None:
        pass


class CallbackList:
    def __init__(self, callbacks: list[Callback]) -> None:
        self._callbacks = callbacks

    def __getattr__(self, name: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for cb in self._callbacks:
                getattr(cb, name)(*args, **kwargs)

        return wrapper

    def new_rollouts_ready(self) -> bool:
        return all(cb.new_rollouts_ready() for cb in self._callbacks)


EVAL_FUTURE_TIMEOUT_S = 30 * 60  # a healthy eval is smaller than a training iteration; treat anything
# past this as a wedged eval actor (e.g. its AppWorld server subprocess is stuck) rather than block
# the training loop's rollout-ready poll forever.


class EvalCallback(Callback):
    def __init__(
        self,
        algo: RLOOTrainer,
        wandb_project: str,
        wandb_group: str,
        wandb_run: str | None,
    ) -> None:
        super().__init__(algo)
        self._wandb_project = wandb_project
        self._wandb_group = wandb_group
        self._wandb_run = wandb_run

        self._eval_actor: ray.actor.ActorHandle[Any] | None = None
        self._eval_future: ray.ObjectRef | None = None
        self._eval_future_started_at: float | None = None

    def _start_actor(self) -> None:
        logger.info("Creating new EvalWorker actor...")
        self._eval_actor = EvalWorker.remote(self._algo._full_cfg)  # type: ignore

    def _ensure_actor(self) -> None:
        if self._eval_actor is None:
            self._start_actor()
            return

        # Bound this: ping.remote() queues behind any in-flight eval() call on the
        # same (single-threaded) actor, so a wedged eval() would otherwise block this
        # ray.get() forever -> rank 0 never reaches the next NCCL collective, and since
        # it's not inside torch.distributed there's no watchdog to catch it.
        try:
            ray.get(self._eval_actor.ping.remote(), timeout=30)
        except (RayActorError, GetTimeoutError):
            logger.warning("EvalWorker actor is dead or unresponsive -> restarting...")
            self._start_actor()
        except Exception as e:
            logger.exception(f"Unexpected error pinging EvalWorker: {e}")
            self._start_actor()

    def _poll_completion(self) -> bool:
        """Non-blocking check of self._eval_future. Returns True once it's resolved
        (or there's nothing to wait for)."""
        if self._eval_future is None:
            return True

        done, _ = ray.wait([self._eval_future], timeout=0)
        if not done:
            elapsed = time.monotonic() - (self._eval_future_started_at or time.monotonic())
            if elapsed > EVAL_FUTURE_TIMEOUT_S:
                logger.warning(
                    f"Eval task exceeded {EVAL_FUTURE_TIMEOUT_S}s ({elapsed:.0f}s elapsed) -> "
                    "treating EvalWorker as wedged and restarting actor."
                )
                try:
                    ray.kill(self._eval_actor)
                except RayActorError:
                    pass
                self._start_actor()
                self._eval_future = None
                self._eval_future_started_at = None
                return True
            return False

        try:
            ray.get(self._eval_future)
            logger.info("Eval task completed.")
        except RayActorError:
            logger.warning("EvalWorker crashed during eval. Restarting actor...")
            self._start_actor()
        except Exception as e:
            logger.exception(f"Eval failed: {e}")
        finally:
            self._eval_future = None
            self._eval_future_started_at = None
        return True

    def before_iteration(self, iteration: int, last_checkpoint_local_path: Path | None) -> None:
        cfg = self._algo._cfg
        if not cfg.eval.enable or not self._wandb_run:
            logger.debug("Eval callback is disabled...")
            return

        if iteration % cfg.eval.eval_every_n_iterations != 0:
            logger.debug(f"Skipping eval due to {cfg.eval.eval_every_n_iterations=}")
            return

        self._ensure_actor()

        assert self._eval_future is None, f"Previous eval still running: {self._eval_future=}"
        assert self._eval_actor is not None, f"{self._eval_actor=}"
        try:
            self._eval_future = self._eval_actor.eval.remote(last_checkpoint_local_path)
            self._eval_future_started_at = time.monotonic()
        except RayActorError:
            logger.exception(f"Could not start eval for {last_checkpoint_local_path=}")

    def before_new_rollouts(self) -> None:
        # Make sure eval doesn't overlap LoRA swapping. Actual waiting happens via
        # new_rollouts_ready(), polled from the train loop so other ranks aren't
        # blocked on a NCCL collective for however long eval takes.
        pass

    def new_rollouts_ready(self) -> bool:
        return self._poll_completion()

    def shutdown(self) -> None:
        if self._eval_actor is not None:
            try:
                logger.info("Shutting down EvalWorker actor...")
                ray.get(self._eval_actor.shutdown.remote())
            except RayActorError:
                logger.warning("EvalWorker actor already dead during shutdown.")
