from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import LitArboel
from bioel.models.arboel.crossencoder.model.CrossEncoderLightningModule import (
    LitCrossEncoder,
)
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
from bioel.models.krissbert.model.model import Krissbert
import json
import importlib
import argparse
import lightning as L
import os
import torch


class Model_Wrapper:
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
    def load_sapbert(cls, name, params):
        pass

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
    def load_scispacy(cls, name, params):
        pass

    @classmethod
    def load_biobart(cls, name, params):
        pass

    @classmethod
    def load_biogenel(cls, name, params):
        pass

    def training(self):
        module_name = self.train_script_path.replace("/", ".").rsplit(".", 1)[0]
        train_module = importlib.import_module(module_name)
        train_module.train_model(params=self.params, model=self.model)

    def inference(self):
        module_name = self.evaluate_script_path.replace("/", ".").rsplit(".", 1)[0]
        evaluate_module = importlib.import_module(module_name)
        evaluate_module.evaluate_model(self.params, self.model)


if __name__ == "__main__":
    print("Start work on krissbert")
    krissbert = Model_Wrapper.load_krissbert(
        name="krissbert",
        params_file="/home2/cye73/data_test2/krissbert/ncbi_disease/params.json",
    )
    krissbert.training()
    krissbert.inference()
    print("Finish work on krissbert")
