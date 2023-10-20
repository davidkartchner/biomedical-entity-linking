# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large set of entity mentions
 based on the pretrained mention encoder.
"""
import logging
import os
import pathlib
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoTokenizer, AutoModel

from utils import generate_vectors


# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


@hydra.main(config_path="conf", config_name="bigbio_prototypes", version_base=None)
def main(cfg: DictConfig):
    logger.info("Configuration:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=True,
    )
    encoder = AutoModel.from_pretrained(cfg.model_name_or_path, config=config)
    encoder.cuda()
    encoder.eval()

    ds = hydra.utils.instantiate(cfg.train_data)
    data = generate_vectors(
        encoder, tokenizer, ds, cfg.batch_size, cfg.max_length, is_prototype=True
    )

    if cfg.train_data._target_ == 'utils.BigBioDataset':
        output_prototypes = f'{cfg.output_dir}/{cfg.train_data.dataset_name}_embeddings.pickle'
        output_name_cuis = f'{cfg.output_dir}/{cfg.train_data.dataset_name}_name_cuis.txt'

    else:
        output_prototypes = cfg.output_prototypes
        output_name_cuis = cfg.output_name_cuis

    pathlib.Path(os.path.dirname(output_prototypes)).mkdir(
        parents=True, exist_ok=True
    )
    logger.info("Writing results to %s" % output_prototypes)
    with open(output_prototypes, mode="wb") as f:
        pickle.dump(data, f)
    with open(output_name_cuis, "w") as f:
        for name, cuis in ds.name_to_cuis.items():
            f.write("|".join(cuis) + "||" + name + "\n")
    logger.info(
        "Total data processed %d. Written to %s", len(data), output_prototypes
    )


if __name__ == "__main__":
    main()
