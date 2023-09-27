import wandb
import logging
from tqdm import tqdm
from typing import Dict


class WandbLogger:
    """ 
    Source: https://github.com/UKPLab/sentence-transformers/issues/705#issuecomment-833521213
    """
    def __init__(self, project_name: str, run_name: str, run_config: Dict[str, str], log_dir: str):
        if wandb.run is not None:
            self.experiment = wandb.run
        else:
            self.experiment = wandb.init(project=project_name, name=run_name, dir=log_dir, config=run_config)

    def log_training(self, train_idx: int, epoch: int, global_step: int, current_lr: float, loss_value: float):
        self.experiment.log(step=global_step, data={
            "train/loss": loss_value, 
            "train/lr": current_lr, 
            "train/epoch": epoch,
        })

    def log_eval(self, epoch: int, global_step: int, prefix: str, value: float):
        self.experiment.log(step=global_step, data={prefix: value})

    def finish(self):
        self.experiment.finish()


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
