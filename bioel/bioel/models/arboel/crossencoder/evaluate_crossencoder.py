from bioel.models.arboel.biencoder.model.common.params import BlinkParser

import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from bioel.models.arboel.crossencoder.data.CrossEncoderLightningDataModule import (
    CrossEncoderDataModule,
)
from bioel.models.arboel.crossencoder.model.CrossEncoderLightningModule import (
    LitCrossEncoder,
)


def evaluate_model(params, model):
    data_module = CrossEncoderDataModule(params=params)

    trainer = L.Trainer(
        # limit_test_batches=5,
        limit_test_batches=(
            params["limit_test_batches"] if params["limit_test_batches"] else 1.0
        ),
        devices=params["devices"],
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
    )

    trainer.test(model=model, datamodule=data_module)
