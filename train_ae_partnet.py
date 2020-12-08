import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.config import load_config
from src.lightning_models.ae import AELightning


def main(args):
    config = load_config(args)

    tb_logger = TensorBoardLogger(config.checkpoint_dir,
                                  name=config.model,
                                  version=config.version)
    CHECKPOINTS = os.path.join(config.checkpoint_dir, config.model, config.version, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINTS,
        save_top_k=100
    )
    os.makedirs(CHECKPOINTS, exist_ok=True)
    model = AELightning(config)

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        logger=tb_logger,
        early_stop_callback=False,
        gpus=config.gpus,
        distributed_backend=config.distributed_backend,
        num_nodes=1,
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        amp_level=config.amp_level,
        log_save_interval=config.log_save_interval,
        fast_dev_run=False
    )
    trainer.fit(model)


if __name__ == '__main__':
    main(sys.argv[1:])
