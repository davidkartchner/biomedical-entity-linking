from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    ArboelDataModule,
)
from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import (
    LitArboel,
)
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import lightning as L
from bioel.ontology import BiomedicalOntology
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
import os
import pickle


def main(args):
    os.environ["WANDB_SILENT"] = "true"
    print("Current seed:", args["seed"])

    # Check if the function name provided is an actual function of BiomedicalOntology
    if hasattr(BiomedicalOntology, args["load_function"]):
        load_func = getattr(BiomedicalOntology, args["load_function"])

        if args["ontology_dict"]:
            ontology_object = load_func(**args["ontology_dict"])
            print(f"Ontology loaded successfully. Name: {ontology_object.name}")
        else:
            raise ValueError("No ontology data provided.")
    else:
        raise ValueError(
            f"Error: {args['load_function']} is not a valid function for BiomedicalOntology."
        )

    with open(
        os.path.join(args["data_path"], f"{args['ontology']}_object.pickle"), "wb"
    ) as f:
        pickle.dump(ontology_object, f)

    data_module = ArboelDataModule(params=args)

    model = LitArboel(params=args)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_checkpoint = ModelCheckpoint(
        monitor="max_acc",  # Metric to monitor
        dirpath=args["output_path"],  # Directory to save the model
        filename=f"biencoder_{current_time}-{{epoch}}-{{max_acc:.2f}}",  # Saves the model with epoch and val_loss in the filename
        save_top_k=1,  # Number of best models to save; -1 means save all of them
        mode="max",  # 'max' means the highest max_acc will be considered as the best model
        verbose=True,  # Logs a message whenever a model checkpoint is saved
    )

    wandb_logger = WandbLogger(project=args["experiment"])

    trainer = L.Trainer(
        limit_val_batches=1,
        max_epochs=args["num_train_epochs"],
        devices=args["devices"],
        accelerator="gpu" if args["devices"] else "cpu",
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
