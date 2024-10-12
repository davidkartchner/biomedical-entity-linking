from bioel.models.krissbert.data.utils import BigBioDataset
import json
import torch
import os


def train_model(params, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    ds = BigBioDataset(
        params["dataset_name"],
        splits=["train"],
        abbreviations_path=params["path_to_abbrev"],
    )

    model.generate_prototypes(
        ds, params["output_dir"], params["batch_size"], params["max_length"]
    )

    print("Done generating prototypes!")
