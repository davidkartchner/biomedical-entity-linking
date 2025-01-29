from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import LitArboel
from bioel.models.arboel.crossencoder.model.CrossEncoderLightningModule import (
    LitCrossEncoder,
)
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
from bioel.models.krissbert.model.model import Krissbert
from bioel.models.biogenel.LightningModule import BioGenElLightningModule
from bioel.models.sapbert.model.metric_learning import Sap_Metric_Learning
from bioel.models.sapbert.model.model_wrapper import Model_Wrapper
from bioel.models.sapbert.train import parse_args

import json
import importlib
import argparse
import lightning as L
import os
import torch


class BioEL_Model:
    def __init__(self, model, name, train_script_path, evaluate_script_path, params):
        self.name = name  # name of the model
        self.model = model  # model object
        self.train_script_path = train_script_path  # path to the training script
        self.evaluate_script_path = (
            evaluate_script_path  # path to the evaluation script
        )
        self.params = params  # model arguments for training and inferencing

    @classmethod
    def load_arboel_biencoder(cls, name, params_file, checkpoint_path=None):
        # Check if params_file is a path or a dictionary
        if isinstance(params_file, str) and os.path.isfile(params_file):
            with open(params_file, "r") as f:
                json_params = json.load(f)
        elif isinstance(params_file, dict):
            json_params = params_file
        else:
            raise TypeError("params_file must be a valid file path or a dictionary")

        parser = BlinkParser(add_model_args=True)
        parser.add_training_args()
        parser.add_eval_args()

        # Set default values and remove required flag for params in JSON
        for action in parser._actions:
            if action.dest in json_params:
                parser.set_defaults(**{action.dest: json_params[action.dest]})
                action.required = False

        # Parse arguments
        args = parser.parse_args()
        params = vars(args)
        params.update(json_params)

        if checkpoint_path:
            model = LitArboel.load_from_checkpoint(
                checkpoint_path=checkpoint_path, params=params
            )
        else:
            model = LitArboel(params=params)

        train_script_path = f"bioel/models/arboel/biencoder/train_biencoder.py"
        evaluate_script_path = f"bioel/models/arboel/biencoder/evaluate_biencoder.py"
        return cls(
            model,
            name,
            train_script_path,
            evaluate_script_path,
            params,
        )

    @classmethod
    def load_arboel_crossencoder(cls, name, params_file, checkpoint_path=None):
        # Check if params_file is a path or a dictionary
        if isinstance(params_file, str) and os.path.isfile(params_file):
            with open(params_file, "r") as f:
                json_params = json.load(f)
        elif isinstance(params_file, dict):
            json_params = params_file
        else:
            raise TypeError("params_file must be a valid file path or a dictionary")

        parser = BlinkParser(add_model_args=True)
        parser.add_training_args()
        parser.add_eval_args()

        # Set default values and remove required flag for params in JSON
        for action in parser._actions:
            if action.dest in json_params:
                parser.set_defaults(**{action.dest: json_params[action.dest]})
                action.required = False

        # Parse arguments
        args = parser.parse_args()
        params = vars(args)
        params.update(json_params)

        if checkpoint_path:
            model = LitCrossEncoder.load_from_checkpoint(
                checkpoint_path=checkpoint_path, params=params
            )
        else:
            model = LitCrossEncoder(params=params)

        train_script_path = f"bioel/models/arboel/crossencoder/train_crossencoder.py"
        evaluate_script_path = (
            f"bioel/models/arboel/crossencoder/evaluate_crossencoder.py"
        )
        return cls(
            model,
            name,
            train_script_path,
            evaluate_script_path,
            params,
        )

    @classmethod
    def load_sapbert(cls, name, params_file=None, checkpoint_path=None):
        if isinstance(params_file, str) and os.path.isfile(params_file):
            with open(params_file, "r") as f:
                json_params = json.load(f)
        elif isinstance(params_file, dict):
            json_params = params_file
        else:
            raise TypeError("params_file must be a valid file path or a dictionary")

        params = parse_args(json_params)

        model = Model_Wrapper()  # model wrapper

        if checkpoint_path:
            # Load the pretrained model from the specified path
            model.load_pretrained_model(checkpoint_path, use_cuda=params["use_cuda"])

        train_script_path = f"bioel/models/sapbert/train.py"
        evaluate_script_path = f"bioel/models/sapbert/evaluate.py"

        return cls(
            model,  # model wrapper
            name,
            train_script_path,
            evaluate_script_path,
            params,
        )

    @classmethod
    def load_krissbert(cls, name, params_file, checkpoint_path=None):
        if isinstance(params_file, str) and os.path.isfile(params_file):
            with open(params_file, "r") as f:
                params = json.load(f)
        elif isinstance(params_file, dict):
            params = params_file
        else:
            raise TypeError("params_file must be a valid file path or a dictionary")

        if checkpoint_path:
            model = Krissbert(checkpoint_path)
        else:
            model = Krissbert(params["model_name_or_path"])

        train_script_path = f"bioel/models/krissbert/train.py"
        evaluate_script_path = f"bioel/models/krissbert/evaluate.py"
        return cls(
            model,
            name,
            train_script_path,
            evaluate_script_path,
            params,
        )

    @classmethod
    def load_scispacy(cls, name, params_file):
        # No model object, just a string identifier
        model = "scispacy"
        train_script_path = None  # ScispaCy does not require training
        evaluate_script_path = "bioel/models/scispacy/evaluate.py"

        return cls(model, name, train_script_path, evaluate_script_path, params_file)

    @classmethod
    def load_biobart(cls, name, params_file):
        params = Config(params_file)
        train_script_path = f"bioel/models/biogenel/train_biogenel_biobart.py"
        evaluate_script_path = f"bioel/models/biogenel/eval_biogenel_biobart.py"
        model = BioGenElLightningModule(config=params)
        return cls(
            model,
            name,
            train_script_path,
            evaluate_script_path,
            params,
        )

    @classmethod
    def load_biogenel(cls, name, params_file):
        params = Config(params_file)
        train_script_path = f"bioel/models/biogenel/train_biogenel_biobart.py"
        evaluate_script_path = f"bioel/models/biogenel/eval_biogenel_biobart.py"
        model = BioGenElLightningModule(config=params)
        return cls(
            model,
            name,
            train_script_path,
            evaluate_script_path,
            params,
        )

    def training(self):
        if self.name.lower() == "scispacy":
            print('ScispaCy "training" is done directly in the inference.')
        else:
            module_name = self.train_script_path.replace("/", ".").rsplit(".", 1)[0]
            train_module = importlib.import_module(module_name)
            train_module.train_model(params=self.params, model=self.model)

    def inference(self):
        module_name = self.evaluate_script_path.replace("/", ".").rsplit(".", 1)[0]
        evaluate_module = importlib.import_module(module_name)
        evaluate_module.evaluate_model(self.params, self.model)


class Config:
    def __init__(self, json_file=None):
        if json_file:
            self.load_from_json(json_file)

    def load_from_json(self, json_file):
        with open(json_file, "r") as f:
            json_params = json.load(f)
            for key, value in json_params.items():
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"


if __name__ == "__main__":
    # print("Start work on arboel_biencoder")
    dataset = "ncbi_disease"
    arboel_biencoder_model = BioEL_Model.load_arboel_biencoder(
        name="arboel_biencoder",
        params_file=f"/home2/cye73/data_test2/arboel/{dataset}/params_biencoder.json",
    )
    arboel_biencoder_model.training()
    arboel_biencoder_model.inference()

    # print("Start work on arboel_crossencoder")
    # arboel_crossencoder_model = BioEL_Model.load_arboel_crossencoder(
    #     name="arboel_bcrossencoder",
    #     params_file="/home2/cye73/data_test2/arboel/medmentions_st21pv/params_crossencoder.json",
    #     # checkpoint_path="/home2/cye73/results2/arboel/medmentions_st21pv/crossencoder_2024-09-25_13-22-32-epoch=1-Accuracy=0.89.ckpt",
    # )
    # arboel_crossencoder_model.training()
    # arboel_crossencoder_model.inference()

    # print("Start work on biobart")
    # biobart_model = BioEL_Model.load_biobart(
    #     name="biobart",
    #     params_file="/home2/cye73/data_test2/biogenel/ncbi_config.json",
    # )
    # biobart_model.training()
    # biobart_model.inference()

    # # krissbert = BioEL_Model.load_krissbert(
    # #     name="krissbert",
    # #     params_file="/home2/cye73/data_test2/krissbert/ncbi_disease/params.json",
    # # )
    # # krissbert.training()
    # # krissbert.inference()

    # scispacy_params = {
    #     "dataset": "ncbi_disease",
    #     "load_function": "load_medic",
    #     "ontology_dict": {
    #         "name": "medic",
    #         "filepath": "/mitchell/entity-linking/kbs/medic.tsv",
    #     },
    #     "k": 10,
    #     "path_to_save": "/home2/cye73/data_test2/scispacy/kb_paths_scispacy/ncbi_disease",
    #     "output_path": "/home2/cye73/results2/scispacy/ncbi_disease_output.json",
    #     "equivalant_cuis": True,
    #     "path_to_abbrev": "/home2/cye73/data_test2/abbreviations.json",
    # }
    # scispacy = BioEL_Model.load_scispacy(name="scispacy", params_file=scispacy_params)
    # scispacy.training()
    # scispacy.inference()

    # # dataset = "nlm_gene"
    # dataset = "nlmchem"
    # print("Start work on sapbert :", dataset)
    # sapbert_model = BioEL_Model.load_sapbert(
    #     name="sapbert",
    #     params_file=f"/home2/cye73/data_test2/sapbert/{dataset}/params.json",
    #     checkpoint_path=f"/home2/cye73/data_test2/sapbert/{dataset}/finetuned_models_nice/eager-voice-55",
    # )
    # # sapbert_model.training()
    # sapbert_model.inference()
