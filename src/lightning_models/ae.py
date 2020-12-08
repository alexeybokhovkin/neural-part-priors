import os
from argparse import Namespace

import pytorch_lightning as pl

from ..models.ae import AE_Encoder, AE_Decoder

class AELightning(pl.LightningModule):
    def __init__(self, hparams):
        super(AELightning, self).__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams
        config = hparams.__dict__
        self.config = config
        for _k, _v in self.config.items():
            if _v is None:
                if _k == "gpus":
                    self.config[_k] = "cpu"
                else:
                    self.config[_k] = "null"