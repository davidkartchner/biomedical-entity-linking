
            


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