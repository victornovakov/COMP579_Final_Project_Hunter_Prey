"""
Experiment logger — writes per-episode metrics to CSV and optionally to
TensorBoard.

Usage:
    from Core_Scripts.logger import ExperimentLogger

    logger = ExperimentLogger("results/ppo_run1", use_tensorboard=True)
    logger.log_episode(episode=1, metrics={"predator_reward": 42.0, ...})
    logger.close()
"""

import csv
import os
from collections import defaultdict


class ExperimentLogger:
    """
    Dual-backend logger: always writes a CSV, optionally streams scalars to
    TensorBoard.

    Parameters
    ----------
    log_dir : str
        Directory for all outputs (csv + tensorboard subfolder).
    use_tensorboard : bool
        Enable TensorBoard logging (requires torch).
    csv_filename : str
        Name of the CSV file inside log_dir.
    """

    def __init__(self, log_dir, use_tensorboard=False, csv_filename="metrics.csv"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, csv_filename)
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields = None

        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            except ImportError:
                print("[Logger] tensorboard requested but torch.utils.tensorboard "
                      "not found — falling back to CSV only.")

        self._episode_cache = defaultdict(list)

    # ── public API ────────────────────────────────────────────────────────

    def log_episode(self, episode, metrics: dict):
        """
        Record one episode's metrics.

        Parameters
        ----------
        episode : int
        metrics : dict[str, float]
            Arbitrary key-value pairs (e.g. predator_reward, prey_reward,
            episode_length, loss, …).
        """
        row = {"episode": episode, **metrics}
        self._write_csv_row(row)

        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, global_step=episode)

    def log_scalar(self, tag, value, step):
        """Log a single scalar (e.g. learning rate, loss) outside episodes."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, global_step=step)

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()

    # ── internals ─────────────────────────────────────────────────────────

    def _write_csv_row(self, row: dict):
        if self._csv_writer is None:
            self._csv_fields = list(row.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()

        if set(row.keys()) != set(self._csv_fields):
            new_keys = set(row.keys()) - set(self._csv_fields)
            if new_keys:
                self._csv_file.close()
                self._csv_fields = list(row.keys())
                self._csv_file = open(self.csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)

        self._csv_writer.writerow(row)
        self._csv_file.flush()
