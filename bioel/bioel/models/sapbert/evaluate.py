import logging
import sys
import ujson


from bioel.models.sapbert.utils import (
    sapbert_predict,
    parse_args_bigbio,
    init_logging,
    load_dictionary,
    make_unique_model_savepath,
)
from torch.utils.data import DataLoader


import wandb

from bioel.models.sapbert.data.utils import SapBertBigBioDataset
from bioel.ontology import BiomedicalOntology
from bioel.models.sapbert.model.model_wrapper import Model_Wrapper


LOGGER = logging.getLogger()


def sapbert_collate_fn(batch):
    mentions = [x["text"] for x in batch]
    # labels = [x["cuis"] for x in batch]
    labels = [x["db_ids"] for x in batch]
    metadata = batch

    return mentions, labels, metadata


def evaluate_model(config, model_wrapper):

    # Get the Alias from the Big Bio dataset into a dictionary
    eval_dictionary = load_dictionary(dictionary_path=config["train_dir"])

    if config["path_to_abbreviation"] is None:
        resolve_abbreviations = False
    else:
        resolve_abbreviations = True

    # Load data
    data = SapBertBigBioDataset(
        config["dataset_name"],
        splits_to_include=[config["split"]],
        path_to_abbreviation_dict=config["path_to_abbreviation"],
        resolve_abbreviations=resolve_abbreviations,
    )
    loader = DataLoader(
        data, collate_fn=sapbert_collate_fn, batch_size=config["eval_batch_size"]
    )

    # Load Model
    model = model_wrapper.load_model(
        path=config["model_dir"],
        max_length=config["max_length"],
        use_cuda=config["use_cuda"],
    )

    # Get unique save path
    preds_save_path = make_unique_model_savepath(
        dir=config["output_dir"],
        model="sapbert",
        dataset=config["dataset_name"],
        file_ext="json",
    )

    dict_cache_filepath = config["dict_cache_path"]
    LOGGER.info(f"Dict CACHE Path: {dict_cache_filepath}")
    LOGGER.info(f"Debugging mode? {config['debug']}")

    results = sapbert_predict(
        model_wrapper=model,
        eval_dictionary=eval_dictionary,
        dataloader=loader,
        batch_size=config["eval_batch_size"],
        topk=config["topk"],
        dict_cache_filepath=dict_cache_filepath,
        debug=config["debug"],
    )

    if resolve_abbreviations:
        for element in results:
            element["mention_id"] = element["mention_id"] + ".abbr_resolved"

    LOGGER.info(f"Saving results to {preds_save_path}")
    with open(preds_save_path, "w") as f:
        f.write(ujson.dumps(results, indent=2))


# if __name__ == "__main__":
#     init_logging()
#     args = parse_args_bigbio()
#     main(args)
