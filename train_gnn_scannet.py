import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.config import load_config
from src.lightning_models.gnn_scannet_old import GNNPartnetLightning


def main(args):
    config = load_config(args)
    CHECKPOINTS = os.path.join(config.base, config.checkpoint_dir, config.model, config.version, 'checkpoints')
    os.makedirs(os.path.join(config.base, config.checkpoint_dir, config.model, config.version), exist_ok=True)
    os.makedirs(CHECKPOINTS, exist_ok=True)

    tb_logger = TensorBoardLogger(os.path.join(config.base, config.checkpoint_dir),
                                  name=config.model,
                                  version=config.version)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=CHECKPOINTS,
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=50
    )
    model = GNNPartnetLightning(config)

    trainer = Trainer(
        callbacks=[checkpoint_callback],
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
        accumulate_grad_batches=4
    )
    trainer.fit(model)


if __name__ == '__main__':
    main(sys.argv[1:])
