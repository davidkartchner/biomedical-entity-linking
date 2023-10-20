import pandas as pd
import ujson
import os

from argparse import ArgumentParser
from tqdm import tqdm


def filter_entrez(
    entrez,
    taxa=None,
    prefix="NCBIGene",
    excluded_gene_types=["unknown", "tRNA", "biological-region"],
    excluded_desc=["hypothetical protein"],
    exclude_predicted_genes=True,
):
    """
    Filter Entrez dataset to include only relevant taxa and gene types
    """
    mask = (~entrez.type_of_gene.isin(excluded_gene_types)) & (
        ~entrez.description.isin(excluded_desc)
    )

    if exclude_predicted_genes:
        mask = mask & (
            ~entrez.official_name.map(lambda x: x.lower().startswith("predicted"))
        )

    if taxa is not None:
        mask = mask & (entrez.tax_id.isin(taxa))

    # taxa_mask = (entrez.tax_id.isin(taxa)) & (~entrez.type_of_gene.isin(['unknown','tRNA','biological-region'])) & (entrez.description != 'hypothetical protein') & (~entrez.official_name.map(lambda x: x.lower().startswith("predicted")))
    filtered = entrez[mask]

    # Find duplicated symbols
    print("Dedup")
    symbols = filtered.symbol.value_counts()
    duplicated_symbols = symbols[symbols > 1].index.tolist()

    # Add additional canonical symbol for symbols that are repeated across different organisms
    duplicated_symbol_mask = filtered.symbol.isin(duplicated_symbols)

    filtered["canonical_symbol"] = filtered["symbol"]
    # filtered.loc[duplicated_symbol_mask, 'canonical_symbol'] = filtered.loc[duplicated_symbol_mask, ['tax_id','symbol']].apply(lambda x: f"{x[1]} ({tax2name[x[0]]})", axis=1)

    # Complie list of all symbols (except primary name)
    print("Complilng symbols")
    filtered["all_symbols"] = filtered[
        [
            "symbol",
            "synonyms",
            "official_symbol",
            "official_name",
            "other_designations",
            "canonical_symbol",
        ]
    ].progress_apply(
        lambda x: "|".join(list(set([i for i in x if i.strip() != "-"]))), axis=1
    )
    # filtered['all_symbols'] = filtered[['synonyms','official_symbol','official_name','other_designations', 'canonical_symbol']].progress_apply(lambda x: '|'.join(list(set([i.strip() for i in x if i.strip() != '-']))), axis=1)

    filtered["geneid"] = filtered.geneid.map(lambda x: f"{prefix}:{x}")

    return filtered

    # geneid2synonym = filtered.set_index("geneid")["all_symbols"].to_dict()


def get_synonyms(filtered):
    """
    Get synonyms for each gene.  Input should be a filtered dataframe
    """
    return filtered.set_index("geneid")["all_symbols"].map(lambda x: set([x for x in d["all_symbols"].split("|")])).to_dict()

def get_entities(filtered):
    all_records = []
    for d in tqdm(filtered.to_dict(orient="records")):
        all_records.append({
            ''
        })


def format_for_arboel(filtered):
    """
    Format Entrez data to be used with ArboEL
    """
    all_records = []
    for d in tqdm(filtered.to_dict(orient="records")):
        entity_dict = {}
        entity_dict["cui"] = d["geneid"]
        entity_dict["title"] = d["symbol"]
        dedup_symbols = list(
            set([x for x in d["all_symbols"].split("|") if x != d["symbol"]])
        )

        # Create a single string of description that includes aliases, type, species, and definition (when available)
        joined_aliases = " ; ".join(dedup_symbols)
        ent_desc = d["description"]
        if len(dedup_symbols) > 0:
            if ent_desc not in ["-", ""] and ent_desc not in dedup_symbols:
                entity_dict[
                    "description"
                ] = f"{d['symbol']} ( {tax2name[str(d['tax_id'])]}, {d['type_of_gene']} : {joined_aliases} ) [ {d['description']} ]"
            else:
                entity_dict[
                    "description"
                ] = f"{d['symbol']} ( {tax2name[str(d['tax_id'])]}, {d['type_of_gene']} : {joined_aliases} )"
        else:
            if ent_desc not in ["-", ""] and ent_desc not in dedup_symbols:
                entity_dict[
                    "description"
                ] = f"{d['symbol']} ( {tax2name[str(d['tax_id'])]}, {d['type_of_gene']} ) [ {d['description']} ]"
            else:
                entity_dict[
                    "description"
                ] = f"{d['symbol']} ( {tax2name[str(d['tax_id'])]}, {d['type_of_gene']} )"

        entity_dict["type"] = d["type_of_gene"]
        all_records.append(entity_dict)

    return all_records


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ncbi_gene_path", default="./data/gene_info.tsv")
    parser.add_argument("--feather_path", default="./data/ncbigene.feather")
    parser.add_argument(
        "--taxa_to_include", help=".json file of taxa to include in final result"
    )
    parser.add_argument(
        "--include_hypothetical_genes",
        action="store_true",
        help="Include hypothetical genes/proteins",
    )
    parser.add_argument(
        "--include_uncommon_gene_types",
        action="store_true",
        help='Include genes of type "tRNA", "biological-region", and "unknown"',
    )
    parser.add_argument(
        "--include_predicted_genes",
        action="store_true",
        help="Include predicted genes/proteins",
    )
    args = parser.parse_args()

    if os.path.isfile(args.feather_path):
        entrez = pd.read_feather(args.feather_path)
    elif os.path.isfile(args.ncbi_gene_path):
        entrez = pd.read_csv(
            args.ncbi_gene_path,
            delimiter="\t",
            usecols=[
                "#tax_id",
                "GeneID",
                "Symbol",
                "Synonyms",
                "Symbol_from_nomenclature_authority",
                "Full_name_from_nomenclature_authority",
                "Other_designations",
                "type_of_gene",
                "description",
                "dbXrefs",
            ],
            na_filter=False,
            low_memory=False,
        ).rename(
            {
                "Symbol_from_nomenclature_authority": "official_symbol",
                "Full_name_from_nomenclature_authority": "official_name",
                "#tax_id": "tax_id",
            },
            axis=1,
        )
    else:
        raise ValueError("Either ncbi_gene_path or feather_path must be specified!")


tax2name = ujson.load(open("../data/tax2name.json", "r"))
