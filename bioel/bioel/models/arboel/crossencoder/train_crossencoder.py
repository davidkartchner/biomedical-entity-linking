from bioel.models.arboel.crossencoder.data.CrossEncoderLightningDataModule import (
    CrossEncoderDataModule,
    prepare_data,
)
from bioel.models.arboel.crossencoder.model.CrossEncoderLightningModule import (
    LitCrossEncoder,
)
from datetime import datetime, timedelta
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import lightning as L
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
import os
import torch


def train_model(params, model):
    print("Current seed:", params["seed"])

    prepare_data(params)

    data_module = CrossEncoderDataModule(params=params)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_checkpoint = ModelCheckpoint(
        monitor="Accuracy",  # Metric to monitor
        dirpath=params["output_path"],  # Directory to save the model
        filename=f"crossencoder_{current_time}-{{epoch}}-{{Accuracy:.2f}}",  # Saves the model with epoch and val_loss in the filename
        save_top_k=1,  # Number of best models to save; -1 means save all of them
        mode="max",  # 'max' means the highest max_acc will be considered as the best model
        verbose=True,  # Logs a message whenever a model checkpoint is saved
    )

    wandb_logger = WandbLogger(project=params["experiment"])

    trainer = L.Trainer(
        limit_train_batches=params.get("limit_train_batches", 1.0),
        limit_val_batches=params.get("limit_val_batches", 20),
        # num_sanity_val_steps=0,
        max_epochs=params["num_train_epochs"],
        devices=params["devices"],
        accelerator="gpu" if params["devices"] else "cpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        callbacks=[model_checkpoint],
        accumulate_grad_batches=params["gradient_accumulation_steps"],
        precision="16-mixed",
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
