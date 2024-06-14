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

def parse_args():
    """
    Parse the input arguments
    """
    parser = argparse.ArgumentParser(description="SAPBERT Training")
    # Required
    parser.add_argument("--model_dir", help="Directory for pretrained model")
    parser.add_argument(
        "--train_dir", type=str, required=True, help="training set directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for output"
    )

    # Tokenizer settings
    parser.add_argument("--max_length", default=25, type=int)

    # Train config
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument(
        "--learning_rate", help="learning rate", default=0.0001, type=float
    )
    parser.add_argument("--weight_decay", help="weight decay", default=0.01, type=float)
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=240, type=int
    )
    parser.add_argument("--epoch", help="epoch to train", default=3, type=int)
    parser.add_argument("--save_checkpoint_all", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=10000000)
    parser.add_argument(
        "--amp", action="store_true", help="automatic mixed precision training"
    )
    parser.add_argument("--parallel", action="store_true")
    # parser.add_argument('--cased', action="store_true")
    parser.add_argument(
        "--pairwise", action="store_true", help="if loading pairwise formatted datasets"
    )
    parser.add_argument("--random_seed", help="epoch to train", default=1996, type=int)
    parser.add_argument(
        "--loss",
        help="{ms_loss|cosine_loss|circle_loss|triplet_loss}}",
        default="ms_loss",
    )
    parser.add_argument("--use_miner", action="store_true")
    parser.add_argument("--miner_margin", default=0.2, type=float)
    parser.add_argument("--type_of_triplets", default="all", type=str)
    parser.add_argument(
        "--agg_mode", default="cls", type=str, help="{cls|mean|mean_all_tok}"
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--mode", default="pretrain", type=str)
    parser.add_argument("--project", type = str, help = "Wandb Project", required = True)
    parser.add_argument("--resolve_abbreviations", action="store_true", help = "consider abbreviations for the BigBio"  )
    parser.add_argument("--path_to_abbreviation_dict", default= "/home/pbathala3/entity_linking/biomedical-entity-linking/bioel/bioel/utils/solve_abbreviation", type = str, help = "Path to the abbreviation dictionary")

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

        if config['amp']:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        if config['amp']:
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
        if step_global % config['checkpoint_step'] == 0:
            checkpoint_dir = os.path.join(
                config['output_dir'], "checkpoint_iter_{}".format(str(step_global))
            )
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
    train_loss /= train_steps + 1e-9
    return train_loss, step_global

def main(config):
    init_logging()
    torch.manual_seed(config["random_seed"])

    # Initialize Wandb
    wandb.init(project = config["project"], config = config)

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    
    # Load BERT Tokenizer, dense encoder
    model_wrapper = Model_Wrapper()
    encoder, tokenizer = model_wrapper.load_bert(
        path = config["model_dir"],
        max_length = config["max_length"],
        use_cuda = config["use_cuda"],
    )

    # Load the SAP Model
    model = Sap_Metric_Learning(
        encoder = encoder,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        use_cuda=config["use_cuda"],
        pairwise=config["pairwise"],
        loss=config["loss"],
        use_miner=config["use_miner"],
        miner_margin=config["miner_margin"],
        type_of_triplets=config["type_of_triplets"],
        agg_mode=config["agg_mode"],
    )

    # configure for Parallel Processing
    if config["parallel"]:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("Parallel Processing enabled")

    # Prepare the date module
    data_module = SapbertDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Mixed Precision Training
    if config["amp"]:
        scaler = GradScaler()
    else:
        scaler = None
    
    start = time.time()
    step_global = 0
    for epoch in range(1, config["epoch"] + 1):
        LOGGER.info("Epoch {}/{}".format(epoch, config["epoch"]))

        # train the model
        train_loss, step_global = train(
            config, 
            data_loader=train_loader,
            model=model,
            scaler=scaler,
            model_wrapper=model_wrapper,
            step_global=step_global,
        )

        LOGGER.info("loss/train_per_epoch={}/{}".format(train_loss, epoch))

        if config['save_checkpoint_all']:
            checkpoint_dir = os.path.join(
                config['output_dir'], "checkpoint_epoch_{}".format(str(epoch))
            )
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
        

        # Save the model Last Epoch
        if epoch == config["epoch"]:
            if not os.path.exists(config["output_dir"]):
                os.makedirs(config["output_dir"])
            model_wrapper.save_model(os.path.join(config["output_dir"]))
    
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

    with open(os.path.join(config["output_dir"], "training_time.txt"), "w") as f:
        f.write(
            "Training Time: {} hours {} minutes {} seconds \n".format(
                training_hour, training_min, training_sec
            )
        )

if __name__ == "__main__":
    config = parse_args()
    main(config)