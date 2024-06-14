import logging
import sys
import ujson



from utils import (
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

def main(args):

    # Get the Alias from the Big Bio dataset into a dictionary
    eval_dictionary = load_dictionary(dictionary_path=args.dictionary_path)
    
    if args.abbreviations_path is None:
        resolve_abbreviations = False
    else:
        resolve_abbreviations = True

    

    # Load data
    data = SapBertBigBioDataset(
        args.dataset_name,
        splits_to_include=[args.split],
        path_to_abbreviation_dict=args.abbreviations_path,
        resolve_abbreviations=resolve_abbreviations,
    )
    loader = DataLoader(data, collate_fn=sapbert_collate_fn, batch_size=args.batch_size)

    # Load Model
    model_wrapper = Model_Wrapper().load_model(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
    )

    # Get unique save path
    preds_save_path = make_unique_model_savepath(
        dir=args.output_dir,
        model="sapbert",
        dataset=args.dataset_name,
        file_ext="json",
    )

    dict_cache_filepath = args.dict_cache_path
    LOGGER.info(f"Dict CACHE Path: {dict_cache_filepath}")
    LOGGER.info(f"Debugging mode? {args.debug}")

    results = sapbert_predict(
        model_wrapper=model_wrapper,
        eval_dictionary=eval_dictionary,
        dataloader=loader,
        batch_size=args.batch_size,
        topk=args.topk,
        dict_cache_filepath=dict_cache_filepath,
        debug=args.debug,
    )

    LOGGER.info(f"Saving results to {preds_save_path}")
    with open(preds_save_path, "w") as f:
        f.write(ujson.dumps(results, indent=2))


if __name__ == "__main__":
    init_logging()
    args = parse_args_bigbio()

    main(args)
