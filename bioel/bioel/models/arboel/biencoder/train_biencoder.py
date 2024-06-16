from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    ArboelDataModule,
)
from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import LitArboel
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.utilities import rank_zero_only
import lightning.pytorch as L
from bioel.ontology import BiomedicalOntology
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
import os
import pickle


@rank_zero_only
def create_ontology_object(args):
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

    file_path = os.path.join(args["data_path"], f"{args['ontology']}_object.pickle")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(ontology_object, f)
    return ontology_object


def train_model(params, model):
    print("params :", params)

    os.environ["WANDB_SILENT"] = "true"
    os.environ["NCCL_TIMEOUT"] = "10800"

    create_ontology_object(params)
    os.makedirs(params["output_path"], exist_ok=True)

    data_module = ArboelDataModule(params=params)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_checkpoint = ModelCheckpoint(
        monitor="max_acc",
        dirpath=params["output_path"],
        filename=f"biencoder_{current_time}-{{epoch}}-{{max_acc:.2f}}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )

    if wandb.run is None:
        wandb.init()
    wandb_logger = WandbLogger(
        project=params["experiment"] if params["experiment"] else wandb.run.name
    )
    trainer = L.Trainer(
        num_sanity_val_steps=0,
        limit_train_batches=(
            params["limit_train_batches"] if params["limit_train_batches"] else 1.0
        ),
        limit_val_batches=1,
        max_epochs=params["num_train_epochs"],
        devices=params["devices"],
        accelerator="gpu" if params["devices"] else "cpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        callbacks=[model_checkpoint],
        accumulate_grad_batches=params["gradient_accumulation_steps"],
        precision="16-mixed",
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
