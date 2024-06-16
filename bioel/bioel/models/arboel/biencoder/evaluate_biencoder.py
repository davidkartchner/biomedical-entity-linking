from bioel.models.arboel.biencoder.model.common.params import BlinkParser

import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import LitArboel
from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    ArboelDataModule,
)
from pytorch_lightning.utilities import rank_zero_only
from bioel.ontology import BiomedicalOntology
import pickle
import os


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


def evaluate_model(params, model):

    create_ontology_object(params)
    os.makedirs(params["output_path"], exist_ok=True)

    data_module = ArboelDataModule(params=params)

    trainer = L.Trainer(
        limit_test_batches=1,
        devices=(
            params["devices"][:1]
            if isinstance(params["devices"], list)
            else params["devices"]
        ),
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        precision="16-mixed",
    )

    trainer.test(model=model, datamodule=data_module)
