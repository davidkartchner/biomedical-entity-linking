from bioel.models.arboel.model.LightningModule import ArboelDataModule, LitArboel
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import lightning as L
from bioel.models.arboel.model.common.params import BlinkParser


def main(args):
    print("Current seed:", args["seed"])

    data_module = ArboelDataModule(params=args)

    model = LitArboel(params=args)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_checkpoint = ModelCheckpoint(
        monitor="max_acc",  # Metric to monitor
        dirpath=args["output_path"],  # Directory to save the model
        filename=f"{current_time}-{{epoch}}-{{max_acc:.2f}}",  # Saves the model with epoch and val_loss in the filename
        save_top_k=1,  # Number of best models to save; -1 means save all of them
        mode="max",  # 'max' means the highest max_acc will be considered as the best model
        verbose=True,  # Logs a message whenever a model checkpoint is saved
    )

    wandb_logger = WandbLogger(project=args["experiment"])

    trainer = L.Trainer(
        limit_val_batches=0.3,
        max_epochs=args["num_train_epochs"],
        devices=args["devices"],
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        callbacks=[model_checkpoint],
        accumulate_grad_batches=args["gradient_accumulation_steps"],
        precision="16-mixed",
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
