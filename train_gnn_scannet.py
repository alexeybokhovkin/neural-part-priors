import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from src.utils.config import load_config
from src.lightning_models.gnn_deepsdf import GNNPartnetLightning


class CheckpointEveryEpoch(pl.Callback):
    def __init__(self, start_epoc, save_path, n_every_epoch=1):
        self.start_epoc = start_epoc
        self.file_path = save_path
        self.n_every_epoch = n_every_epoch

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        if epoch >= self.start_epoc and epoch % self.n_every_epoch == 0:
            ckpt_path = os.path.join(self.file_path, f"{epoch}.ckpt")
            trainer.save_checkpoint(ckpt_path)


def main(args):
    config = load_config(args)
    CHECKPOINTS = os.path.join(config.checkpoint_dir, config.model, config.version, 'checkpoints')
    os.makedirs(os.path.join(config.checkpoint_dir, config.model, config.version), exist_ok=True)
    os.makedirs(os.path.join(config.checkpoint_dir, config.model, config.version, 'latents'), exist_ok=True)
    os.makedirs(CHECKPOINTS, exist_ok=True)


    tb_logger = TensorBoardLogger(os.path.join(config.checkpoint_dir),
                                  name=config.model,
                                  version=config.version)
    model = GNNPartnetLightning(config, mode=config.mode)

    print('Experiment name:', config.version)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        devices=-1,
        precision=32,
        callbacks=[CheckpointEveryEpoch(0, CHECKPOINTS, config.save_every), lr_monitor],
        logger=tb_logger,
        accelerator='cuda',
        num_nodes=1,
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        fast_dev_run=False,
        # resume_from_checkpoint=config.resume_from_checkpoint,
        accumulate_grad_batches=1,
        strategy = DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(model)


    # trainer = pl.Trainer(devices=-1,
    #                      accelerator='gpu',
    #                     #  strategy="ddp",
    #                      precision=32, max_epochs=specs["num_epochs"], callbacks=callbacks, log_every_n_steps=1,
    #                      default_root_dir=os.path.join("tensorboard_logs", args.exp_dir),
    #                      num_sanity_val_steps=0,
    #                     #  fast_dev_run=16
    #                     #  val_check_interval=0.01,
    #                      limit_val_batches=400,
    #                      limit_train_batches=8000,
    #                      strategy = DDPStrategy(find_unused_parameters=False),
    #                      )


if __name__ == '__main__':
    main(sys.argv[1:])
