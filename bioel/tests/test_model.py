import unittest
import numpy as np

from tqdm import tqdm

from bioel.ontology import BiomedicalOntology
from bioel.model import Model_Wrapper
import os
import pickle
import ujson


class TestModel(unittest.TestCase):

    def test_arboel_biencoder(self):
        """
        TestCase - 1: Test the training part of Biencoder
        """
        with open("config_arboel.json", "r") as f:
            config = ujson.load(f)
        test_cases = config["test_cases_biencoder"]
        for params_file in test_cases:
            arboel_biencoder = Model_Wrapper.load_arboel_biencoder(
                name="arboel_biencoder",
                params_file=params_file,
            )
            arboel_biencoder.training()
            arboel_biencoder.inference()

    def test_arboel_crossencoder(self):
        """
        TestCase - 1: Test the training part of Biencoder
        """
        with open("config_arboel.json", "r") as f:
            config = ujson.load(f)
        test_cases = config["test_cases_crossencoder"]
        for params_file in test_cases:
            arboel_cross = Model_Wrapper.load_arboel_crossencoder(
                name="arboel_crossencoder",
                params_file=params_file,
            )
            arboel_cross.training()
            arboel_cross.inference()


# class TestModel(unittest.TestCase):

#     def test_train_arboel_biencoder(self):
#         """
#         TestCase - 1: Test the training part of Biencoder
#         """

#         # Load configuration file
#         with open("config_arboel.json", "r") as f:
#             config = ujson.load(f)

#         test_cases = config["test_cases_train_biencoder"]

#         parser = BlinkParser(add_model_args=True)
#         parser.add_training_args()

#         for case in test_cases:
#             test_args = [
#                 "--data_path",
#                 case["data_path"],
#                 "--output_path",
#                 case["output_path"],
#                 "--dataset",
#                 case["dataset"],
#                 "--ontology",
#                 case["ontology"],
#                 "--load_function",
#                 case["load_function"],
#                 "--model_name_or_path",
#                 "dmis-lab/biobert-base-cased-v1.2",
#             ]

#             args = parser.parse_args(test_args)
#             args_dict = args.__dict__
#             args_dict["ontology_dict"] = case["ontology_dict"]

#             # Check if the function name provided is an actual function of BiomedicalOntology
#             if hasattr(BiomedicalOntology, args_dict["load_function"]):
#                 load_func = getattr(BiomedicalOntology, args_dict["load_function"])

#                 if args_dict["ontology_dict"]:
#                     ontology_object = load_func(**args_dict["ontology_dict"])
#                     print(f"Ontology loaded successfully. Name: {ontology_object.name}")
#                 else:
#                     raise ValueError("No ontology data provided.")
#             else:
#                 raise ValueError(
#                     f"Error: {args_dict['load_function']} is not a valid function for BiomedicalOntology."
#                 )

#             with open(
#                 os.path.join(
#                     args_dict["data_path"], f"{args_dict['ontology']}_object.pickle"
#                 ),
#                 "wb",
#             ) as f:
#                 pickle.dump(ontology_object, f)

#             data_module = ArboelDataModule(params=args_dict)

#             model = LitArboel(params=args_dict)

#             trainer = L.Trainer(
#                 num_sanity_val_steps=0,
#                 limit_train_batches=1,
#                 limit_val_batches=1,
#                 max_epochs=1,
#                 devices=[0],
#                 accelerator="gpu",
#                 strategy="ddp_find_unused_parameters_true",
#             )

#             trainer.fit(model, datamodule=data_module)

#     def test_evaluate_arboel_biencoder(self):
#         """
#         TestCase - 2: Test the evaluate part of Biencoder
#         """

#         # Load configuration file
#         with open("config_arboel.json", "r") as f:
#             config = ujson.load(f)

#         test_cases = config["test_cases_eval_biencoder"]

#         parser = BlinkParser(add_model_args=True)
#         parser.add_eval_args()

#         for case in test_cases:

#             test_args = [
#                 "--data_path",
#                 case["data_path"],
#                 "--output_path",
#                 case["output_path"],
#                 "--dataset",
#                 case["dataset"],
#                 "--ontology",
#                 case["ontology"],
#                 "--load_function",
#                 case["load_function"],
#                 "--biencoder_checkpoint",
#                 case["biencoder_checkpoint"],
#                 "--model_name_or_path",
#                 "dmis-lab/biobert-base-cased-v1.2",
#             ]

#             args = parser.parse_args(test_args)
#             args_dict = args.__dict__
#             args_dict["ontology_dict"] = case["ontology_dict"]
#             # Check if the function name provided is an actual function of BiomedicalOntology
#             if hasattr(BiomedicalOntology, args_dict["load_function"]):
#                 load_func = getattr(BiomedicalOntology, args_dict["load_function"])

#                 if args_dict["ontology_dict"]:
#                     ontology_object = load_func(**args_dict["ontology_dict"])
#                     print(f"Ontology loaded successfully. Name: {ontology_object.name}")
#                 else:
#                     raise ValueError("No ontology data provided.")
#             else:
#                 raise ValueError(
#                     f"Error: {args_dict['load_function']} is not a valid function for BiomedicalOntology."
#                 )

#             with open(
#                 os.path.join(
#                     args_dict["data_path"], f"{args_dict['ontology']}_object.pickle"
#                 ),
#                 "wb",
#             ) as f:
#                 pickle.dump(ontology_object, f)

#             data_module = ArboelDataModule(params=args_dict)

#             model = LitArboel.load_from_checkpoint(
#                 params=args_dict, checkpoint_path=args_dict["biencoder_checkpoint"]
#             )

#             trainer = L.Trainer(
#                 limit_test_batches=1,
#                 devices=[0],
#                 accelerator="gpu",
#                 strategy="ddp_find_unused_parameters_true",
#             )

#             trainer.test(model=model, datamodule=data_module)

#     def test_train_arboel_crossencoder(self):
#         """
#         TestCase - 3: Test the training part of Crossencoder
#         """

#         # Load configuration file
#         with open("config_arboel.json", "r") as f:
#             config = ujson.load(f)

#         test_cases = config["test_cases_train_crossencoder"]

#         parser = BlinkParser(add_model_args=True)
#         parser.add_training_args()

#         for case in test_cases:
#             test_args = [
#                 "--data_path",
#                 case["data_path"],
#                 "--output_path",
#                 case["output_path"],
#                 "--biencoder_checkpoint",
#                 case["biencoder_checkpoint"],
#                 "--biencoder_indices_path",
#                 case["biencoder_indices_path"],
#                 "--model_name_or_path",
#                 "dmis-lab/biobert-base-cased-v1.2",
#                 "--add_linear",
#                 "--train_batch_size",
#                 "2",
#             ]

#             args = parser.parse_args(test_args)
#             args_dict = args.__dict__

#             data_module = CrossEncoderDataModule(params=args_dict)

#             model = LitCrossEncoder(params=args_dict)

#             trainer = L.Trainer(
#                 limit_train_batches=1,
#                 limit_val_batches=1,
#                 num_sanity_val_steps=0,
#                 max_epochs=1,
#                 devices=[0],
#                 accelerator="gpu",
#                 strategy="ddp_find_unused_parameters_true",
#             )

#             trainer.fit(model, datamodule=data_module)

#     def test_evaluate_arboel_crossencoder(self):
#         """
#         TestCase - 4: Test the evaluate part of Crossencoder
#         """

#         # Load configuration file
#         with open("config_arboel.json", "r") as f:
#             config = ujson.load(f)

#         test_cases = config["test_cases_eval_crossencoder"]

#         parser = BlinkParser(add_model_args=True)
#         parser.add_eval_args()
#         for case in test_cases:

#             test_args = [
#                 "--data_path",
#                 case["data_path"],
#                 "--output_path",
#                 case["output_path"],
#                 "--biencoder_checkpoint",
#                 case["biencoder_checkpoint"],
#                 "--biencoder_indices_path",
#                 case["biencoder_indices_path"],
#                 "--crossencoder_checkpoint",
#                 case["crossencoder_checkpoint"],
#                 "--model_name_or_path",
#                 "dmis-lab/biobert-base-cased-v1.2",
#                 "--add_linear",
#             ]

#             args = parser.parse_args(test_args)
#             args_dict = args.__dict__
#             args_dict["recall_k_list"] = [2]

#             data_module = CrossEncoderDataModule(params=args_dict)

#             model = LitCrossEncoder.load_from_checkpoint(
#                 params=args_dict, checkpoint_path=args_dict["crossencoder_checkpoint"]
#             )

#             trainer = L.Trainer(
#                 limit_test_batches=1,
#                 devices=[0],
#                 accelerator="gpu",
#                 strategy="ddp_find_unused_parameters_true",
#             )

#             trainer.test(model=model, datamodule=data_module)
