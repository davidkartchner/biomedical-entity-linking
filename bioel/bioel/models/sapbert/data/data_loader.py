

class SapBertBigBioDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "medmentions_st21pv",
        splits_to_include: List[str] = ["train"],
        resolve_abbreviations: bool = True,
        path_to_abbreviation_dict: Optional[str] = None,
    ):
        """
        Load initial BigBio dataset
        """
        # Pull in data
        self.conhelps = BigBioConfigHelpers()
        self.data = self.conhelps.for_config_name(
            dataset_name + "_bigbio_kb"
        ).load_dataset(from_hub=False)
        self.splits_to_include = splits_to_include

        # Resolve abbreviations if desired
        self.resolve_abbreviations = resolve_abbreviations
        if self.resolve_abbreviations:
            self.abbreviations = ujson.load(open(path_to_abbreviation_dict, "r"))

        self.cuis_to_exclude = CUIS_TO_EXCLUDE[dataset_name]
        self.cuis_to_remap = CUIS_TO_REMAP[dataset_name]

        # Put examples into list for retrieval
        self._data_to_flat_instances()

    # def dataset_to_df(self, dataset, splits_to_include: List = None):
    #     """
    #     Convert BigBio dataset to pandas DataFrame

    #     Params:
    #     ------------------
    #         dataset: BigBio Dataset
    #             Dataset to load from BigBio

    #         splits_to_include: list of str
    #             List of splits to include in mo
    #     """
    #     columns = [
    #         "document_id",
    #         "mention_id",
    #         "text",
    #         "type",
    #         "offsets",
    #         # "db_name",
    #         "db_ids",
    #         "split",
    #     ]
    #     all_lines = []

    #     if splits_to_include is None:
    #         splits_to_include = dataset.keys()

    #     for split in splits_to_include:
    #         if split not in dataset.keys():
    #             warnings.warn(f"Split '{split}' not in dataset.  Omitting.")
    #         for doc in dataset[split]:
    #             pmid = doc["document_id"]
    #             for e in doc["entities"]:
    #                 if len(e["normalized"]) == 0:
    #                     continue
    #                 text = " ".join(e["text"])
    #                 offsets = ";".join(
    #                     [",".join([str(y) for y in x]) for x in e["offsets"]]
    #                 )
    #                 # db_name = e["normalized"][0]["db_name"]
    #                 db_ids = [x["db_name"] + ":" + x["db_id"] for x in e["normalized"]]
    #                 all_lines.append(
    #                     [
    #                         pmid,
    #                         e["id"],
    #                         text,
    #                         e["type"],
    #                         # e['offsets'],
    #                         offsets,
    #                         # db_name,
    #                         db_ids,
    #                         split,
    #                     ]
    #                 )

    #     df = pd.DataFrame(all_lines, columns=columns)

    #     deduplicated = (
    #         df.groupby(["document_id", "offsets"])
    #         .agg(
    #             {
    #                 "text": "first",
    #                 "type": lambda x: list(set([a for a in x])),
    #                 "db_ids": lambda db_ids: list(set([y for x in db_ids for y in x])),
    #                 "split": "first",
    #             }
    #         )
    #         .reset_index()
    #     )
    #     deduplicated["offsets"] = deduplicated["offsets"].map(
    #         lambda x: [y.split(",") for y in x.split(";")]
    #     )

    #     return deduplicated

    def _data_to_flat_instances(self):
        """
        Convert dataset into flat set of examples to use with dataloader
        """
        df = dataset_to_df(
            self.data,
            self.splits_to_include,
            entity_remapping_dict=self.cuis_to_remap,
            cuis_to_exclude=self.cuis_to_exclude,
        )
        if self.resolve_abbreviations:
            df["text"] = df[["document_id", "text"]].apply(
                lambda x: resolve_abbreviation(x[0], x[1], self.abbreviations), axis=1
            )
        self.flat_instances = df.to_dict(orient="records")

    def __len__(self):
        return len(self.flat_instances)

    def __getitem__(self, idx):
        return self.flat_instances[idx]


def sapbert_collate_fn(batch):
    mentions = [x["text"] for x in batch]
    # labels = [x["cuis"] for x in batch]
    labels = [x["db_ids"] for x in batch]
    metadata = batch

    return mentions, labels, metadata


if __name__ == "__main__":
    dataset = SapBertBigBioDataset(
        path_to_abbreviation_dict="../data/abbreviations.json"
    )
    dataloader = DataLoader(dataset, collate_fn=sapbert_collate_fn, batch_size=64)
    for i, batch in enumerate(tqdm(dataloader)):
        if i < 3:
            print(batch)
            


class MetricLearningDataset_pairwise(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        for line in lines:
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("||")
            self.query_ids.append(query_id)
            self.query_names.append((name1, name2))
        self.tokenizer = tokenizer
        self.query_id_2_index_id = {k: v for v, k in enumerate(list(set(self.query_ids)))}
    
    def __getitem__(self, query_idx):

        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]
        query_id = self.query_ids[query_idx]
        query_id = int(self.query_id_2_index_id[query_id])

        return query_name1, query_name2, query_id


    def __len__(self):
        return len(self.query_names)



class MetricLearningDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer): #d_ratio, s_score_matrix, s_candidate_idxs):
        logger.info("Initializing metric learning data set! ...")
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        cuis = []
        for line in lines:
            cui, _ = line.split("||")
            cuis.append(cui)

        self.cui2id = {k: v for v, k in enumerate(cuis)}
        for line in lines:
            line = line.rstrip("\n")
            cui, name = line.split("||")
            query_id = self.cui2id[cui]
            #if query_id.startswith("C"):
            #    query_id = query_id[1:]
            #query_id = int(query_id)
            self.query_ids.append(query_id)
            self.query_names.append(name)
        self.tokenizer = tokenizer
    
    def __getitem__(self, query_idx):

        query_name = self.query_names[query_idx]
        query_id = self.query_ids[query_idx]
        query_token = self.tokenizer.transform([query_name])[0]

        return torch.tensor(query_token), torch.tensor(query_id)

    def __len__(self):
        return len(self.query_names)

class SapbertDataModule(L.LightningDataModule):
    def prepare_data(self):
        ...