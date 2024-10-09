#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from pytorch_metric_learning import samplers
import logging
import time
import pdb
import os
import json
import random
from tqdm import tqdm

import wandb

from bioel.models.sapbert.data.lightingDataModule import SapbertDataModule
from bioel.models.sapbert.model.model_wrapper import Model_Wrapper
from bioel.models.sapbert.model.metric_learning import Sap_Metric_Learning

LOGGER = logging.getLogger(__name__)


def parse_args(config):
    """
    Parse the input arguments
    """
    print(config)
    parser = argparse.ArgumentParser(description="SAPBERT Training")
    # Required
    parser.add_argument(
        "--model_dir",
        default=config["model_dir"],
        help="Directory for pretrained model",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=config["train_dir"],
        help="training set directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config["output_dir"],
        help="Directory for output",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[
            "medmentions_full",
            "medmentions_st21pv",
            "bc5cdr",
            "nlmchem",
            "nlm_gene",
            "gnormplus",
            "ncbi_disease",
        ],
        default=config["dataset_name"],
        help="data set to evaluate",
    )
    parser.add_argument(
        "--dict_cache_path", type=str, default=config["dict_cache_path"]
    )
    # Tokenizer settings
    parser.add_argument("--max_length", type=int, default=config.get("max_length", 25))

    # Train config
    parser.add_argument("--use_cuda", default=config["use_cuda"])
    parser.add_argument(
        "--learning_rate",
        help="learning rate",
        default=config.get("learning_rate", 0.0001),
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="weight decay",
        default=config.get("weight_decay", 0.01),
        type=float,
    )
    parser.add_argument(
        "--train_batch_size",
        help="train batch size",
        default=config.get("train_batch_size", 240),
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        help="eval batch size",
        default=config.get("eval_batch_size", 240),
        type=int,
    )
    parser.add_argument(
        "--split",
        default=config.get("split", "test"),
        type=str,
        help="Which datasplit to evaluate model performance on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run model with small data subset as a test of functionality",
    )
    parser.add_argument(
        "--dict_batch_size", type=int, default=config.get("dict_batch_size", 8192)
    )
    parser.add_argument("--topk", type=int, default=config.get("topk", 10))

    parser.add_argument(
        "--epoch", help="epoch to train", default=config.get("epoch", 3), type=int
    )
    parser.add_argument(
        "--save_checkpoint_all", default=config.get("save_checkpoint_all", False)
    )
    parser.add_argument(
        "--checkpoint_step", type=int, default=config.get("checkpoint_step", 1000000)
    )
    parser.add_argument(
        "--amp",
        default=config.get("amp", False),
        help="automatic mixed precision training",
    )
    parser.add_argument("--parallel", default=config.get("parallel", False))
    # parser.add_argument('--cased', action="store_true")
    parser.add_argument(
        "--pairwise",
        default=config.get("pairwise", False),
        help="if loading pairwise formatted datasets",
    )
    parser.add_argument(
        "--random_seed",
        help="epoch to train",
        default=config.get("random_seed", 1992),
        type=int,
    )
    parser.add_argument(
        "--loss",
        help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
        default=config.get("loss", "ms_loss"),
    )
    parser.add_argument("--use_miner", default=config.get("use_miner", False))
    parser.add_argument(
        "--miner_margin", default=config.get("miner_margin", 0.2), type=float
    )
    parser.add_argument(
        "--type_of_triplets", default=config.get("type_of_triplets", "all"), type=str
    )
    parser.add_argument(
        "--agg_mode",
        default=config.get("agg_mode", "cls"),
        type=str,
        help="{cls|mean|mean_all_tok}",
    )
    parser.add_argument(
        "--num_workers", default=config.get("num_workers", 16), type=int
    )
    parser.add_argument("--mode", default=config.get("mode", "pretrain"), type=str)
    parser.add_argument(
        "--project",
        default=config.get("project", "SAPBERT"),
        type=str,
        help="Wandb Project",
    )
    parser.add_argument(
        "--resolve_abbreviations",
        default=config.get("resolve_abbreviations", False),
        help="consider abbreviations for the BigBio",
    )
    parser.add_argument(
        "--path_to_abbreviation",
        default=config["path_to_abbreviation"],
        type=str,
        help="Path to the abbreviation dictionary",
    )

    args = parser.parse_args()
    return vars(args)


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(config, data_loader, model, scaler=None, model_wrapper=None, step_global=0):
    LOGGER.info("train!")

    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        batch_x1, batch_x2, batch_y = data
        batch_x_cuda1, batch_x_cuda2 = {}, {}
        for k, v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k, v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()

        batch_y_cuda = batch_y.cuda()

        if config["amp"]:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        if config["amp"]:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1
        # if (i+1) % 10 == 0:
        # LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1,train_loss / (train_steps+1e-9)))
        # LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1, loss.item()))

        # save model every K iterations
        if step_global % config["checkpoint_step"] == 0:
            checkpoint_dir = os.path.join(
                config["output_dir"], "checkpoint_iter_{}".format(str(step_global))
            )
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
    train_loss /= train_steps + 1e-9
    return train_loss, step_global


def train_model(params, model):
    model_wrapper = model  # model wrapper
    init_logging()
    torch.manual_seed(params["random_seed"])

    # Initialize Wandb
    wandb.init(project=params["project"], config=params)

    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    # Load BERT Tokenizer, dense encoder
    # model_wrapper = Model_Wrapper()
    encoder, tokenizer = model_wrapper.load_bert(
        path=params["model_dir"],
        max_length=params["max_length"],
        use_cuda=params["use_cuda"],
    )

    # Load the SAP Model
    model_true = Sap_Metric_Learning(
        encoder=encoder,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        use_cuda=params["use_cuda"],
        pairwise=params["pairwise"],
        loss=params["loss"],
        use_miner=params["use_miner"],
        miner_margin=params["miner_margin"],
        type_of_triplets=params["type_of_triplets"],
        agg_mode=params["agg_mode"],
    )

    # paramsure for Parallel Processing
    if params["parallel"]:
        model_true.encoder = torch.nn.DataParallel(model_true.encoder)
        LOGGER.info("Parallel Processing enabled")

    # Prepare the date module
    data_module = SapbertDataModule(params)
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Mixed Precision Training
    if params["amp"]:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    step_global = 0
    for epoch in range(1, params["epoch"] + 1):
        LOGGER.info("Epoch {}/{}".format(epoch, params["epoch"]))

        # train the model
        train_loss, step_global = train(
            params,
            data_loader=train_loader,
            model=model_true,
            scaler=scaler,
            model_wrapper=model_wrapper,
            step_global=step_global,
        )

        LOGGER.info("loss/train_per_epoch={}/{}".format(train_loss, epoch))

        if params["save_checkpoint_all"]:
            checkpoint_dir = os.path.join(
                params["output_dir"], "checkpoint_epoch_{}".format(str(epoch))
            )
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)

        # Save the model Last Epoch
        if epoch == params["epoch"]:
            model_wrapper.save_model(os.path.join(params["output_dir"], wandb.run.name))

    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_min = int(training_time / 60 % 60)
    training_sec = int(training_time % 60)

    LOGGER.info(
        "Training Time: {} hours {} minutes {} seconds".format(
            training_hour, training_min, training_sec
        )
    )

    with open(os.path.join(params["output_dir"], "training_time.txt"), "w") as f:
        f.write(
            "Training Time: {} hours {} minutes {} seconds \n".format(
                training_hour, training_min, training_sec
            )
        )


# if __name__ == "__main__":
#     config = parse_args()
#     main(config)
