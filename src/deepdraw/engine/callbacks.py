import csv
import time

import numpy as np

from lightning.pytorch import Callback
from lightning.pytorch.callbacks import BasePredictionWriter


# This ensures CSVLogger logs training and evaluation metrics on the same line
# CSVLogger only accepts numerical values, not strings
class LoggingCallback(Callback):
    """Lightning callback to log various training metrics and device
    information."""

    def __init__(self, resource_monitor):
        super().__init__()
        self.training_loss = []
        self.validation_loss = []
        self.start_training_time = 0
        self.start_epoch_time = 0

        self.resource_monitor = resource_monitor
        self.max_queue_retries = 2

    def on_train_start(self, trainer, pl_module):
        self.start_training_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_epoch_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.training_loss.append(outputs["loss"].item())

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self.validation_loss.append(outputs["validation_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.resource_monitor.trigger_summary()

        self.epoch_time = time.time() - self.start_epoch_time
        eta_seconds = self.epoch_time * (
            trainer.max_epochs - trainer.current_epoch
        )
        current_time = time.time() - self.start_training_time

        self.log("total_time", current_time)
        self.log("eta", eta_seconds)
        self.log("loss", np.average(self.training_loss))
        self.log("learning_rate", pl_module.optimizer.param_groups[0]['lr'])
        self.log("validation_loss", np.average(self.validation_loss))

        queue_retries = 0
        # In case the resource monitor takes longer to fetch data from the queue, we wait
        # Give up after self.resource_monitor.interval * self.max_queue_retries if cannot retrieve metrics from queue
        while (
            self.resource_monitor.data is None
            and queue_retries < self.max_queue_retries
        ):
            queue_retries = queue_retries + 1
            print(
                f"Monitor queue is empty, retrying in {self.resource_monitor.interval}s"
            )
            time.sleep(self.resource_monitor.interval)

        if queue_retries >= self.max_queue_retries:
            print(
                f"Unable to fetch monitoring information from queue after {queue_retries} retries"
            )

        assert self.resource_monitor.q.empty()

        for metric_name, metric_value in self.resource_monitor.data:
            self.log(metric_name, float(metric_value))

        self.resource_monitor.data = None

        self.training_loss = []
        self.validation_loss = []


class PredictionsWriter(BasePredictionWriter):
    """Lightning callback to write predictions to a file."""

    def __init__(self, logfile_name, logfile_fields, write_interval):
        super().__init__(write_interval)
        self.logfile_name = logfile_name
        self.logfile_fields = logfile_fields

    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ):
        with open(self.logfile_name, "w") as logfile:
            logwriter = csv.DictWriter(logfile, fieldnames=self.logfile_fields)
            logwriter.writeheader()

            for prediction in predictions:
                logwriter.writerow({"filename": prediction[0]})
            logfile.flush()