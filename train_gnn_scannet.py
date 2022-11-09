import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from src.utils.config import load_config
# from src.lightning_models.gnn_scannet_old import GNNPartnetLightning
# from src.lightning_models.gnn_scannet_contrastive import GNNPartnetLightning
from src.lightning_models.gnn_deepsdf import GNNPartnetLightning
# from src.lightning_models.gnn_scannet_byol import GNNPartnetLightning
# from src.lightning_models.gnn_latent_learner import LatentLearner


class CheckpointEveryEpoch(pl.Callback):
    def __init__(self, start_epoc, save_path, n_every_epoch=1):
        self.start_epoc = start_epoc
        self.file_path = save_path
        self.n_every_epoch = n_every_epoch

    def on_epoch_end(self, trainer: pl.Trainer, _):
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

    # torch.multiprocessing.set_start_method('spawn')

    tb_logger = TensorBoardLogger(os.path.join(config.checkpoint_dir),
                                  name=config.model,
                                  version=config.version)
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath=CHECKPOINTS,
    #     filename='{epoch}-{val_loss:.4f}',
    #     save_top_k=50
    # )
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        mode='max',
        dirpath=CHECKPOINTS,
        filename='{epoch}-{train_loss:.4f}',
        period=config.save_every
    )
    model = GNNPartnetLightning(config)

    print('Experiment:', config.version)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        callbacks=[CheckpointEveryEpoch(0, CHECKPOINTS, config.save_every), lr_monitor],
        logger=tb_logger,
        gpus=config.gpus,
        accelerator=config.distributed_backend,
        num_nodes=1,
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        amp_level=config.amp_level,
        log_every_n_steps=config.log_every_n_steps,
        fast_dev_run=False,
        # resume_from_checkpoint=config.resume_from_checkpoint,
        accumulate_grad_batches=1
    )
    trainer.fit(model)


if __name__ == '__main__':
    main(sys.argv[1:])
