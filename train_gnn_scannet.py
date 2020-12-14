import os, sys

from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import Namespace
import torch
import numpy as np
import random

from utils.config import load_config
from unet3d.lightning_model_scannet import Unet3DGNNPartnetLightning
from unet3d.lightning_model import GeoEncoderLightning
from unet3d.lightning_pretrain_model import GeoPretrainLightning


def main(args):
    config = load_config(args)

    random.seed(config.manual_seed)
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed(config.manual_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    tb_logger = TensorBoardLogger(config.checkpoint_dir,
                                  name=config.model,
                                  version=config.version)
    CHECKPOINTS = os.path.join(config.checkpoint_dir, config.model, config.version, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINTS,
        save_top_k=100
    )
    os.makedirs(CHECKPOINTS, exist_ok=True)
    if config.model == 'Unet3DGNNPartnet':
        model = Unet3DGNNPartnetLightning(config)
    elif config.model == 'GeoEncoder':
        model = GeoEncoderLightning(config)
    elif config.model == 'GeoPretrain':
        model = GeoPretrainLightning(config)

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
        log_save_interval=10,
        fast_dev_run=False,
        # resume_from_checkpoint=config.resume_from_checkpoint,
        accumulate_grad_batches=4
    )
    trainer.fit(model)


if __name__ == '__main__':
    main(sys.argv[1:])
