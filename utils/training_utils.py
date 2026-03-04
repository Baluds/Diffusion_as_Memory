import time

class ETATracker:
    """
    Tracks per-epoch wall-clock time and computes an estimated time remaining.
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self._epoch_times: list[float] = []
        self._epoch_start: float | None = None
        self._completed: int = 0

    def start_epoch(self):
        """Call at the beginning of each epoch."""
        self._epoch_start = time.time()

    def end_epoch(self) -> tuple[float, float, str]:
        """
        Call at the end of each epoch.
        """
        if self._epoch_start is None:
            raise RuntimeError("end_epoch() called before start_epoch()")

        elapsed = time.time() - self._epoch_start
        self._epoch_times.append(elapsed)
        self._completed += 1
        self._epoch_start = None

        remaining = self.total_epochs - self._completed
        avg = sum(self._epoch_times) / len(self._epoch_times)
        eta_s = avg * remaining

        eta_h = int(eta_s // 3600)
        eta_m = int((eta_s % 3600) // 60)
        eta_s_part = int(eta_s % 60)
        eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s_part:02d}"

        return elapsed, eta_s, eta_str

    def wandb_metrics(self, elapsed: float, eta_s: float) -> dict:
        """
        Returns a dict of W&B-ready timing metrics to pass to wandb.log() func.
        """
        return {
            "epoch_time_s": elapsed,
            "eta_seconds": eta_s,
        }
