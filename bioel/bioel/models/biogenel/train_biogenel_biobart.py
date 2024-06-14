from bioel.models.biogenel.trie import Trie
from bioel.models.biogenel.utils import reform_input
from bioel.models.biogenel.LightningDataModule import BioGenELDataModule
from bioel.models.biogenel.LightningModule import BioGenElLightningModule
from bioel.ontology import BiomedicalOntology
from transformers import BartTokenizer

def train_model(params, model: BioGenElLightningModule):
    tokenizer = BartTokenizer.from_pretrained(f"{params.model_token_path}")
    if hasattr(BiomedicalOntology, params.load_function):
        load_func = getattr(BiomedicalOntology, params.load_function)
        if params.ontology_dict:
            ontology_object = load_func(**params.ontology_dict)
            print(f"Ontology loaded successfully. Name: {ontology_object.name}")
        else:
            raise ValueError("No ontology data provided.")
    else:
        raise ValueError(
            f"Error: {params.load_function} is not a valid function for BiomedicalOntology."
        )
    datamodule = BioGenELDataModule(tokenizer, 
                    save_data_dir=f"{params.dataset_path}/", 
                    dataset_name = params.dataset_name, 
                    ontology = ontology_object,
                    prefix_mention_is= params.prefix_mention_is,
                    evaluate= params.evaluation,
                    resolve_abbrevs= params.resolve_abbrevs,
                    preprocess_data = params.preprocess_data,
                    path_to_abbrev = params.path_to_abbrev,
                    batch_size = 1)
    
    model.init_datamodule(datamodule=datamodule)
    model.train()



