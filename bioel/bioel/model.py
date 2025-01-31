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


def load_dict(params_file):
    if isinstance(params_file, str) and os.path.isfile(params_file):
        with open(params_file, "r") as f:
            params = json.load(f)
    elif isinstance(params_file, dict):
        params = params_file
    else:
        raise TypeError("params_file must be a valid file path or a dictionary")
    return params


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
        json_params = load_dict(params_file)

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
        json_params = load_dict(params_file)

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
        json_params = load_dict(params_file)
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
        params = load_dict(params_file)

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
        params = load_dict(params_file)

        model = "scispacy"
        train_script_path = None  # ScispaCy does not require training
        evaluate_script_path = "bioel/models/scispacy/evaluate.py"

        return cls(model, name, train_script_path, evaluate_script_path, params)

    @classmethod
    def load_biobart(cls, name, params_file):
        params = Config(params_file)
        print("type params :", type(params))
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
