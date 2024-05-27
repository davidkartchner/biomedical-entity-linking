from bioel.models.arboel.biencoder.model.common.params import BlinkParser

import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import LitArboel
from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    ArboelDataModule,
)
from bioel.ontology import BiomedicalOntology
import pickle
import os


def main(args):

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

    MyModel = LitArboel.load_from_checkpoint(
        params=args, checkpoint_path=args["biencoder_checkpoint"]
    )

    trainer = L.Trainer(
        limit_test_batches=1,
        devices=args["devices"],
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        precision="16-mixed",
    )

    trainer.test(model=MyModel, datamodule=data_module)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)
