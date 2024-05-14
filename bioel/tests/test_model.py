import unittest
import numpy as np

from tqdm import tqdm

from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    ArboelDataModule,
)
from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import (
    LitArboel,
)
from bioel.models.arboel.crossencoder.data.CrossEncoderLightningDataModule import (
    CrossEncoderDataModule,
)
from bioel.models.arboel.crossencoder.model.CrossEncoderLightningModule import (
    LitCrossEncoder,
)
import lightning as L
from bioel.ontology import BiomedicalOntology
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
import os
import pickle


class TestModel(unittest.TestCase):

    # def test_train_arboel_biencoder(self):
    #     """
    #     TestCase - 1: Test the training part of Biencoder
    #     """

    #     test_cases = [
    #         {
    #             "data_path": "/home2/cye73/data_test2/arboel/ncbi_disease",
    #             "output_path": "/home2/cye73/results2/arboel/ncbi_disease",
    #             "dataset": "ncbi_disease",
    #             "ontology": "medic",
    #             "load_function": "load_medic",
    #             "ontology_dict": {
    #                 "name": "medic",
    #                 "filepath": "/mitchell/entity-linking/kbs/medic.tsv",
    #             },
    #         },
    #         {
    #             "data_path": "/home2/cye73/data_test2/arboel/bc5cdr",
    #             "output_path": "/home2/cye73/results2/arboel/bc5cdr",
    #             "dataset": "bc5cdr",
    #             "ontology": "mesh",
    #             "load_function": "load_mesh",
    #             "ontology_dict": {
    #                 "name": "mesh",
    #                 "filepath": "/mitchell/entity-linking/2017AA/META/",
    #             },
    #         },
    #     ]

    #     parser = BlinkParser(add_model_args=True)
    #     parser.add_training_args()

    #     for case in test_cases:
    #         test_args = [
    #             "--data_path",
    #             case["data_path"],
    #             "--output_path",
    #             case["output_path"],
    #             "--dataset",
    #             case["dataset"],
    #             "--ontology",
    #             case["ontology"],
    #             "--load_function",
    #             case["load_function"],
    #             "--model_name_or_path",
    #             "dmis-lab/biobert-base-cased-v1.2",
    #         ]

    #         args = parser.parse_args(test_args)
    #         args_dict = args.__dict__
    #         args_dict["ontology_dict"] = case["ontology_dict"]

    #         # Check if the function name provided is an actual function of BiomedicalOntology
    #         if hasattr(BiomedicalOntology, args_dict["load_function"]):
    #             load_func = getattr(BiomedicalOntology, args_dict["load_function"])

    #             if args_dict["ontology_dict"]:
    #                 ontology_object = load_func(**args_dict["ontology_dict"])
    #                 print(f"Ontology loaded successfully. Name: {ontology_object.name}")
    #             else:
    #                 raise ValueError("No ontology data provided.")
    #         else:
    #             raise ValueError(
    #                 f"Error: {args_dict['load_function']} is not a valid function for BiomedicalOntology."
    #             )

    #         with open(
    #             os.path.join(
    #                 args_dict["data_path"], f"{args_dict['ontology']}_object.pickle"
    #             ),
    #             "wb",
    #         ) as f:
    #             pickle.dump(ontology_object, f)

    #         data_module = ArboelDataModule(params=args_dict)

    #         model = LitArboel(params=args_dict)

    #         trainer = L.Trainer(
    #             num_sanity_val_steps=0,
    #             limit_train_batches=1,
    #             limit_val_batches=1,
    #             max_epochs=1,
    #             devices=[0],
    #             accelerator="gpu",
    #             strategy="ddp_find_unused_parameters_true",
    #         )

    #         trainer.fit(model, datamodule=data_module)

    # def test_evaluate_arboel_biencoder(self):
    #     """
    #     TestCase - 2: Test the evaluate part of Biencoder
    #     """

    #     test_cases = [
    #         {
    #             "data_path": "/home2/cye73/data_test2/arboel/ncbi_disease",
    #             "output_path": "/home2/cye73/results2/arboel/ncbi_disease",
    #             "dataset": "ncbi_disease",
    #             "ontology": "medic",
    #             "load_function": "load_medic",
    #             "biencoder_checkpoint": "/home2/cye73/results2/arboel/ncbi_disease/2024-05-04_10-55-27-epoch=5-max_acc=87.33.ckpt",
    #             "ontology_dict": {
    #                 "name": "medic",
    #                 "filepath": "/mitchell/entity-linking/kbs/medic.tsv",
    #             },
    #         },
    #         {
    #             "data_path": "/home2/cye73/data_test2/arboel/bc5cdr",
    #             "output_path": "/home2/cye73/results2/arboel/bc5cdr",
    #             "dataset": "bc5cdr",
    #             "ontology": "mesh",
    #             "load_function": "load_mesh",
    #             "biencoder_checkpoint": "/home2/cye73/results2/arboel/bc5cdr/2024-05-02_20-56-16-epoch=5-max_acc=85.13.ckpt",
    #             "ontology_dict": {
    #                 "name": "mesh",
    #                 "filepath": "/mitchell/entity-linking/2017AA/META/",
    #             },
    #         },
    #     ]

    #     parser = BlinkParser(add_model_args=True)
    #     parser.add_eval_args()

    #     for case in test_cases:

    #         test_args = [
    #             "--data_path",
    #             case["data_path"],
    #             "--output_path",
    #             case["output_path"],
    #             "--dataset",
    #             case["dataset"],
    #             "--ontology",
    #             case["ontology"],
    #             "--load_function",
    #             case["load_function"],
    #             "--biencoder_checkpoint",
    #             case["biencoder_checkpoint"],
    #             "--model_name_or_path",
    #             "dmis-lab/biobert-base-cased-v1.2",
    #         ]

    #         args = parser.parse_args(test_args)
    #         args_dict = args.__dict__
    #         args_dict["ontology_dict"] = case["ontology_dict"]
    #         # Check if the function name provided is an actual function of BiomedicalOntology
    #         if hasattr(BiomedicalOntology, args_dict["load_function"]):
    #             load_func = getattr(BiomedicalOntology, args_dict["load_function"])

    #             if args_dict["ontology_dict"]:
    #                 ontology_object = load_func(**args_dict["ontology_dict"])
    #                 print(f"Ontology loaded successfully. Name: {ontology_object.name}")
    #             else:
    #                 raise ValueError("No ontology data provided.")
    #         else:
    #             raise ValueError(
    #                 f"Error: {args_dict['load_function']} is not a valid function for BiomedicalOntology."
    #             )

    #         with open(
    #             os.path.join(
    #                 args_dict["data_path"], f"{args_dict['ontology']}_object.pickle"
    #             ),
    #             "wb",
    #         ) as f:
    #             pickle.dump(ontology_object, f)

    #         data_module = ArboelDataModule(params=args_dict)

    #         model = LitArboel.load_from_checkpoint(
    #             params=args_dict, checkpoint_path=args_dict["biencoder_checkpoint"]
    #         )

    #         trainer = L.Trainer(
    #             limit_test_batches=1,
    #             devices=[0],
    #             accelerator="gpu",
    #             strategy="ddp_find_unused_parameters_true",
    #         )

    #         trainer.test(model=model, datamodule=data_module)

    # def test_train_arboel_crossencoder(self):
    #     """
    #     TestCase - 3: Test the training part of Crossencoder
    #     """

    #     test_cases = [
    #         {
    #             "data_path": "/home2/cye73/data_test2/arboel/ncbi_disease",
    #             "output_path": "/home2/cye73/results2/arboel/ncbi_disease",
    #             # "dataset": "ncbi_disease",
    #             # "ontology": "medic",
    #             # "load_function": "load_medic",
    #             "biencoder_checkpoint": "/home2/cye73/results2/arboel/ncbi_disease/2024-05-04_10-55-27-epoch=5-max_acc=87.33.ckpt",
    #             "biencoder_indices_path": "/home2/cye73/data_test2/arboel/ncbi_disease",
    #             # "ontology_dict": {
    #             #     "name": "medic",
    #             #     "filepath": "/mitchell/entity-linking/kbs/medic.tsv",
    #             # },
    #         },
    #         {
    #             "data_path": "/home2/cye73/data_test2/arboel/bc5cdr",
    #             "output_path": "/home2/cye73/results2/arboel/bc5cdr",
    #             # "dataset": "bc5cdr",
    #             # "ontology": "mesh",
    #             # "load_function": "load_mesh",
    #             "biencoder_checkpoint": "/home2/cye73/results2/arboel/bc5cdr/2024-05-02_20-56-16-epoch=5-max_acc=85.13.ckpt",
    #             "biencoder_indices_path": "/home2/cye73/data_test2/arboel/bc5cdr",
    #             # "ontology_dict": {
    #             #     "name": "mesh",
    #             #     "filepath": "/mitchell/entity-linking/2017AA/META/",
    #             # },
    #         },
    #     ]
    #     parser = BlinkParser(add_model_args=True)
    #     parser.add_training_args()

    #     for case in test_cases:
    #         test_args = [
    #             "--data_path",
    #             case["data_path"],
    #             "--output_path",
    #             case["output_path"],
    #             "--biencoder_checkpoint",
    #             case["biencoder_checkpoint"],
    #             "--biencoder_indices_path",
    #             case["biencoder_indices_path"],
    #             "--model_name_or_path",
    #             "dmis-lab/biobert-base-cased-v1.2",
    #             "--add_linear",
    #             "--train_batch_size",
    #             "2",
    #         ]

    #         args = parser.parse_args(test_args)
    #         args_dict = args.__dict__

    #         data_module = CrossEncoderDataModule(params=args_dict)

    #         model = LitCrossEncoder(params=args_dict)

    #         trainer = L.Trainer(
    #             limit_train_batches=1,
    #             limit_val_batches=1,
    #             num_sanity_val_steps=0,
    #             max_epochs=1,
    #             devices=[0],
    #             accelerator="gpu",
    #             strategy="ddp_find_unused_parameters_true",
    #         )

    #         trainer.fit(model, datamodule=data_module)

    def test_evaluate_arboel_crossencoder(self):
        """
        TestCase - 4: Test the evaluate part of Crossencoder
        """

        test_cases = [
            {
                "data_path": "/home2/cye73/data_test2/arboel/ncbi_disease",
                "output_path": "/home2/cye73/results2/arboel/ncbi_disease",
                # "dataset": "ncbi_disease",
                # "ontology": "medic",
                # "load_function": "load_medic",
                "biencoder_checkpoint": "/home2/cye73/results2/arboel/ncbi_disease/2024-05-04_10-55-27-epoch=5-max_acc=87.33.ckpt",
                "biencoder_indices_path": "/home2/cye73/data_test2/arboel/ncbi_disease",
                "crossencoder_checkpoint": "/home2/cye73/results2/arboel/ncbi_disease/2024-05-04_12-20-07-epoch=4-Accuracy=0.90.ckpt",
                # "ontology_dict": {
                #     "name": "medic",
                #     "filepath": "/mitchell/entity-linking/kbs/medic.tsv",
                # },
            },
            {
                "data_path": "/home2/cye73/data_test2/arboel/bc5cdr",
                "output_path": "/home2/cye73/results2/arboel/bc5cdr",
                # "dataset": "bc5cdr",
                # "ontology": "mesh",
                # "load_function": "load_mesh",
                "biencoder_checkpoint": "/home2/cye73/results2/arboel/bc5cdr/2024-05-02_20-56-16-epoch=5-max_acc=85.13.ckpt",
                "biencoder_indices_path": "/home2/cye73/data_test2/arboel/bc5cdr",
                "crossencoder_checkpoint": "/home2/cye73/results2/arboel/bc5cdr/2024-05-03_17-06-57-epoch=4-max_acc=0.00.ckpt",
                # "ontology_dict": {
                #     "name": "mesh",
                #     "filepath": "/mitchell/entity-linking/2017AA/META/",
                # },
            },
        ]

        parser = BlinkParser(add_model_args=True)
        parser.add_eval_args()
        for case in test_cases:

            test_args = [
                "--data_path",
                case["data_path"],
                "--output_path",
                case["output_path"],
                "--biencoder_checkpoint",
                case["biencoder_checkpoint"],
                "--biencoder_indices_path",
                case["biencoder_indices_path"],
                "--crossencoder_checkpoint",
                case["crossencoder_checkpoint"],
                "--model_name_or_path",
                "dmis-lab/biobert-base-cased-v1.2",
                "--add_linear",
            ]

            args = parser.parse_args(test_args)
            args_dict = args.__dict__
            args_dict["recall_k_list"] = [2]

            data_module = CrossEncoderDataModule(params=args_dict)

            model = LitCrossEncoder.load_from_checkpoint(
                params=args_dict, checkpoint_path=args_dict["crossencoder_checkpoint"]
            )

            trainer = L.Trainer(
                limit_test_batches=1,
                devices=[0],
                accelerator="gpu",
                strategy="ddp_find_unused_parameters_true",
            )

            trainer.test(model=model, datamodule=data_module)
