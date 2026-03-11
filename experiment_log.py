# experiment logging — saves loss curves, metrics, and config per run
# usage:
#   log = ExperimentLog("stage1/multiscale_koch")
#   log.config({"n_pts": 8000, "epochs": 10000, ...})
#   for ep in range(epochs):
#       log.step({"loss": loss, "phys": phys_loss, "lam": lam})
#   log.metric("final_lambda", 19.87)
#   log.save()  # writes to logs/stage1_multiscale_koch_20260311_030000.json

import json
import os
import time
from datetime import datetime


class ExperimentLog:
    def __init__(self, name, log_dir="logs"):
        self.name = name
        self.log_dir = log_dir
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._config = {}
        self._steps = []
        self._metrics = {}
        os.makedirs(log_dir, exist_ok=True)

    def config(self, cfg):
        """Record experiment configuration."""
        self._config.update(cfg)

    def step(self, data):
        """Record one training step. Call every epoch or every N epochs."""
        self._steps.append(data)

    def metric(self, key, value):
        """Record a final metric (e.g. final_lambda, mean_residual)."""
        self._metrics[key] = value

    def save(self):
        """Write everything to a JSON file."""
        elapsed = time.time() - self.start_time
        record = {
            "name": self.name,
            "timestamp": self.timestamp,
            "elapsed_seconds": round(elapsed, 1),
            "config": self._config,
            "metrics": self._metrics,
            "history": self._steps,
        }
        safe_name = self.name.replace("/", "_").replace(" ", "_")
        path = os.path.join(self.log_dir, f"{safe_name}_{self.timestamp}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        print(f"Experiment log saved: {path}")
        return path

    def summary(self):
        """Print a one-line summary."""
        elapsed = time.time() - self.start_time
        n = len(self._steps)
        mets = ", ".join(f"{k}={v}" for k, v in self._metrics.items())
        print(f"[{self.name}] {n} steps, {elapsed:.0f}s | {mets}")
