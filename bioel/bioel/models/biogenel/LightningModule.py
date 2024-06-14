import os
import sys

import numpy as np
from tqdm import tqdm
import pickle
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import TrainingArguments
from transformers import BartTokenizer
import torch 
import torch.nn as nn
from transformers import TrainingArguments
from transformers import Seq2SeqTrainingArguments
from bioel.models.biogenel.trainer import modifiedSeq2SeqTrainer
from bioel.models.biogenel.trie import Trie
from bioel.models.biogenel.utils import reform_input
from bioel.models.biogenel.LightningDataModule import BioGenELDataModule
from bioel.models.biogenel.fairseq_beam import (SequenceGenerator,PrefixConstrainedBeamSearch,PrefixConstrainedBeamSearchWithSampling)
import copy
import json
import ujson

from bioel.ontology import BiomedicalOntology

class BioGenElLightningModule(pl.LightningModule):
    def __init__(self, config, datamodule: Optional[BioGenELDataModule] = None):
        super().__init__()
        self.config = config
        if datamodule:
            self.datamodule = datamodule
        self.trie = None  # Initialize trie
        config.max_steps = config.max_steps // config.gradient_accumulate
        config.save_steps = config.max_steps

        self.training_args = Seq2SeqTrainingArguments(
                        output_dir=config.model_save_path,          # output directory
                        num_train_epochs=config.num_train_epochs,              # total number of training epochs
                        per_device_train_batch_size=config.per_device_train_batch_size,  # batch size per device during training
                        per_device_eval_batch_size=config.per_device_eval_batch_size,   # batch size for evaluation
                        warmup_steps=config.warmup_steps,                # number of warmup steps for learning rate scheduler
                        weight_decay=config.weight_decay,               # strength of weight decay
                        logging_dir=config.logging_path,            # directory for storing logs
                        logging_steps=config.logging_steps,
                        save_steps=config.save_steps,
                        evaluation_strategy=config.evaluation_strategy,
                        learning_rate=config.init_lr,
                        label_smoothing_factor=config.label_smoothing_factor,
                        max_grad_norm=config.max_grad_norm,
                        max_steps=config.max_steps,
                        lr_scheduler_type=config.lr_scheduler_type,
                        seed=config.seed,
                        gradient_accumulation_steps=config.gradient_accumulate,
                        generation_config = None,
                        save_safetensors=False,
                        )

        # Initialize model and tokenizer based on config
        if config.evaluation:
            if config.t5:
                from bioel.models.biogenel.models import T5EntityPromptModel
                from transformers import T5Tokenizer, T5Config
                #If one day we can get the code for prepare_trainer_dataset_t5
                #from bioel.models.biogenel.LightningDataModule import prepare_trainer_dataset_t5 as prepare_trainer_dataset

                t5conf = T5Config.from_pretrained("./t5-large")
                t5conf.dropout_rate = config.dropout

                self.tokenizer = T5Tokenizer.from_pretrained("./t5-large")

                self.model = T5EntityPromptModel.from_pretrained(
                    config.model_load_path,
                    config=t5conf,
                    n_tokens=(config.prompt_tokens_enc, config.prompt_tokens_dec),
                    load_prompt=True,
                    soft_prompt_path=config.model_load_path,
                )
            else:
                from bioel.models.biogenel.models import BartEntityPromptModel
                from transformers import BartTokenizer, BartConfig
                #from bioel.models.biogenel.LightningDataModule import prepare_trainer_dataset_fine as prepare_trainer_dataset

                self.tokenizer = BartTokenizer.from_pretrained(config.model_token_path)

                bartconf = BartConfig.from_pretrained(config.model_load_path)
                bartconf.max_position_embeddings = config.max_position_embeddings
                bartconf.attention_dropout = config.attention_dropout
                bartconf.dropout = config.dropout
                bartconf.max_length = config.max_length

                self.model = BartEntityPromptModel.from_pretrained(
                    config.model_load_path,
                    config=bartconf,
                    n_tokens=(config.prompt_tokens_enc, config.prompt_tokens_dec),
                    load_prompt=True,
                    soft_prompt_path=config.model_load_path,
                )
        else:
            if config.t5:
                # Initialize T5 model and tokenizer
                from bioel.models.biogenel.models import T5EntityPromptModel
                from transformers import T5Tokenizer, T5Config
                #If one day we can get the code for prepare_trainer_dataset_t5
                #from bioel.models.biogenel.LightningDataModule import prepare_trainer_dataset_t5 as prepare_trainer_dataset

                t5conf = T5Config.from_pretrained('./t5-large')
                t5conf.dropout_rate = config.dropout
                self.tokenizer = T5Tokenizer.from_pretrained('./t5-large')

                self.model = T5EntityPromptModel.from_pretrained(config.model_load_path, 
                                                            config = t5conf,
                                                            finetune = config.finetune, 
                                                            n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                            load_prompt = config.load_prompt,
                                                            soft_prompt_path = config.model_load_path,
                                                            initialize_from_vocab = config.init_from_vocab,
                                                            )
            else:
                # Initialize Bart model and tokenizer
                from bioel.models.biogenel.models import BartEntityPromptModel
                from transformers import BartTokenizer, BartConfig

                bartconf = BartConfig.from_pretrained(config.model_load_path)
                bartconf.max_position_embeddings = config.max_position_embeddings
                bartconf.attention_dropout = config.attention_dropout
                bartconf.dropout = config.dropout

                self.tokenizer = BartTokenizer.from_pretrained(config.model_token_path, 
                                                        max_length=1024,
                                                        )

                self.model = BartEntityPromptModel.from_pretrained(config.model_load_path, 
                                                            config = bartconf,
                                                            finetune = config.finetune, 
                                                            n_tokens = (config.prompt_tokens_enc, config.prompt_tokens_dec),
                                                            load_prompt = config.load_prompt,
                                                            soft_prompt_path = config.model_load_path,
                                                            no_finetune_decoder = config.no_finetune_decoder,
                                                            )
            
        self.model = self.model.cuda().to(self.model.device) 

            
    def trainer_choice(self):
        if self.config.preprocess_data:
            self.datamodule.prepare_data()
        self.datamodule.setup()
        train_dataset = self.datamodule.train_dataloader().dataset
        if self.config.unlikelihood_loss:
            print("loading trie......")
            with open(self.config.trie_path, "rb") as f:
                self.trie = Trie.load_from_dict(pickle.load(f))
            print("trie loaded.......")
            

            self.trainer = modifiedSeq2SeqTrainer(
                        model=self.model,                         # the instantiated  Transformers model to be trained
                        args=self.training_args,                  # training arguments, defined above
                        train_dataset=train_dataset, 
                        fairseq_loss=self.config.fairseq_loss,  
                        enc_num = self.config.prompt_tokens_enc,
                        dec_num = self.config.prompt_tokens_dec,
                        prefix_allowed_tokens_fn = lambda batch_id, sent: self.trie.get(sent.tolist()),
                        rdrop = self.config.rdrop,
                    )
        else:
            self.trainer = modifiedSeq2SeqTrainer(
                                    model=self.model,                         # the instantiated  Transformers model to be trained
                                    args=self.training_args,                  # training arguments, defined above
                                    train_dataset=train_dataset, 
                                    fairseq_loss=self.config.fairseq_loss,  
                                    enc_num = self.config.prompt_tokens_enc,
                                    dec_num = self.config.prompt_tokens_dec,
                                    rdrop = self.config.rdrop,
                                )

    def train(self):
        self.trainer_choice()
        self.trainer.train()

    def evaluate(self):
        self.evaluate_choice()
        self.on_evaluate()
        self.apply_fairseq_beam_search()
    
    def init_datamodule(self, datamodule = BioGenELDataModule):
        self.datamodule = datamodule

    def evaluate_choice(self):
        if self.config.preprocess_data:
            self.datamodule.prepare_data()
        self.datamodule.setup()
        dataframe = self.datamodule.deduplicated
        
        if self.config.testset:
            print('eval on test set')
            self.eval_dataset = self.datamodule.test_dataloader().dataset
            self.eval_docs = dataframe[dataframe['split'] == 'test']
        else:
            print('eval on develop set')
            self.eval_dataset = self.datamodule.dev_dataloader().dataset
            self.eval_docs = dataframe[dataframe['split'] == 'validation']
    
    def on_evaluate(self):
        print("loading cui2str dictionary....")
        dict_path = self.config.dict_path
        print("dictpath", dict_path)
        if "json" in dict_path:
            with open(dict_path, "r") as f:
                cui2str = ujson.load(f)
        else:
            with open(dict_path, "rb") as f:
                cui2str = pickle.load(f)

        str2cui = {}
        for cui in cui2str:
            if isinstance(cui2str[cui], list):
                for name in cui2str[cui]:
                    if name in str2cui:
                        str2cui[name].append(cui)
                    else:
                        str2cui[name] = [cui]
            else:
                name = cui2str[cui]
                if name in str2cui:
                    str2cui[name].append(cui)
                    print("duplicated vocabulary")
                else:
                    str2cui[name] = [cui]
        print("dictionary loaded......")

        if self.config.rerank:
            print("loading retrieved names......")
            with open(self.config.retrieved_path, "r") as f:
                retrieved_names = [line.split("\t")[0].split(" ") for line in f.readlines()]
            print("retrieved names loaded.")
            for i, l in tqdm(enumerate(retrieved_names)):
                for cui in list(l):
                    if cui in cui2str:
                        continue
                    else:
                        retrieved_names[i].remove(cui)

            print("loading tokenized names......")
            with open(self.config.dataset_path + "/tokenized.json", "r") as f:
                tokenized_names = ujson.load(f)
            print("tokenized names loaded.")
            self.retrieved_names = retrieved_names
            self.tokenized_names = tokenized_names

        if self.config.gold_sty:
            print("loading tokenized names......")
            with open(self.config.dataset_path + "/tokenized.json", "r") as f:
                tokenized_names = ujson.load(f)
            print("tokenized names loaded.")

            print("loading sty to cui dict.....")
            with open(self.config.dataset_path + "/sty2cui.json", "r") as f:
                sty2cuis = ujson.load(f)
            with open(self.config.dataset_path + "/sty.json", "r") as f:
                cuis2sty = ujson.load(f)
            print("sty to cui dict loaded.")

            trie_dict = {}
            for sty in sty2cuis:
                names = []
                for cui in tqdm(sty2cuis[sty]):
                    names += tokenized_names[cui]
                trie_dict[sty] = Trie(names)
            
            self.tokenized_names = tokenized_names
            self.cuis2sty = cuis2sty
            self.sty2cuis = sty2cuis
            self.trie_dict = trie_dict

        print("loading trie......")
        with open(self.config.trie_path, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
        print("trie loaded.......")

        print("loading label cuis......")
        with open(self.config.dataset_path + "/testlabel.txt", "r") as f:
            cui_labels = [
                set(cui.strip("\n").replace("+", "|").split("|")) for cui in f.readlines()
            ]
        print("label cuis loaded")

        self.str2cui = str2cui
        self.trie = trie
        self.cui_labels = cui_labels 


    def apply_fairseq_beam_search(self):
        if self.config.beam_threshold == 0:
            print('without using beam threshold')
            beam_strategy = PrefixConstrainedBeamSearch(
                tgt_dict=None, 
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist())
            )
        else:
            beam_strategy = PrefixConstrainedBeamSearchWithSampling(
                tgt_dict=None, 
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                logit_thresholding=self.config.beam_threshold,
            )
        
        fairseq_generator = SequenceGenerator(
            models = self.model,
            tgt_dict = None,
            beam_size=self.config.num_beams,
            max_len_a=0,
            max_len_b=self.config.max_length,
            min_len=self.config.min_length,
            eos=self.model.config.eos_token_id,
            search_strategy=beam_strategy,

            ##### all hyperparams below are set to default
            normalize_scores=True,
            len_penalty=self.config.length_penalty,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            symbols_to_strip_from_output=None,
            lm_model=None,
            lm_weight=1.0,
        )

        results = list()
        cui_results = list()
        results_score = list()
        text = list()
        candidates_metadata = list()
        output = list()

        input_ids = []
        decoder_input_ids = []
        attention_mask = []
        count_top1 = 0
        count_top5 = 0
        for i in tqdm(range(0, len(self.eval_dataset))):
            
            if self.config.rerank:
                self.trie = Trie(sum([self.tokenized_names[cui] for cui in self.retrieved_names[i]], []))
                fairseq_generator.search = (PrefixConstrainedBeamSearch(
                                                                        tgt_dict=None, 
                                                                        prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist())
                                                                        ))
            if self.config.gold_sty:
                self.trie = self.trie_dict[self.cuis2sty[self.cui_labels[i]]]
                fairseq_generator.search = (PrefixConstrainedBeamSearch(
                                                                        tgt_dict=None, 
                                                                        prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist())
                                                                        ))
                
            input_ids.append(self.eval_dataset[i]['input_ids'])
            attention_mask.append(self.eval_dataset[i]['attention_mask'])
            decoder_input_ids.append(self.eval_dataset[i]['decoder_input_ids_test'])

            if i%self.config.per_device_eval_batch_size == 0:

                input_ids, attention_mask = reform_input(torch.stack(input_ids), attention_mask=torch.stack(attention_mask), ending_token=self.model.config.eos_token_id)
                sample = {'net_input':{'input_ids':input_ids, 'attention_mask':attention_mask}}
                
                result_tokens, posi_scores = fairseq_generator.forward(
                    sample=sample,
                    prefix_mention_is = self.config.prefix_mention_is,
                    prefix_tokens=decoder_input_ids[0].unsqueeze(0).cuda() if self.config.prefix_mention_is else None,
                )

                for ba, beam_sent in enumerate(result_tokens):
                    result = []
                    cui_result = []
                    candidates = []
                    for be, sent in enumerate(beam_sent):
                        if self.config.prefix_mention_is:
                            result.append(self.tokenizer.decode(sent[len(decoder_input_ids[0]):], skip_special_tokens=True))
                        else:
                            result.append(self.tokenizer.decode(sent, skip_special_tokens=True))
                    
                    for idx, r in enumerate(result):
                        if r.strip(" ") in self.str2cui:
                            if idx == 0:
                                text.append(r.strip(" "))
                            cui_result.append(self.str2cui[r.strip(" ")])
                            candidates.append(
                                {"text": r.strip(" "), "db_id": self.str2cui[r.strip(" ")]}
                            )
                        else:
                            print(f"Candidate not found in str2cui: {r}")

                            if (
                                "MESH" in r
                                or "mesh" in r
                                or "omim" in r
                                or "OMIM" in r
                                or "umls" in r
                                or "UMLS" in r
                                or "NCBIGene" in r
                                or "NCBI" in r
                            ):
                                if idx == 0:
                                    text.append(r)
                                cui_result.append(r)
                                candidates.append({"text": r.strip(" "), "db_id": r})
                            else:
                                continue

                    if len(cui_result) == 0:
                        continue

                    cui_results.append(cui_result)
                    results.append(result)
                    results_score.append(posi_scores)
                    candidates_metadata.append(candidates)
                    if 'deabbreviated_text' in self.eval_docs:
                        output.append(
                            {
                                "document_id": self.eval_docs['document_id'].iloc[i],
                                "offsets": self.eval_docs['offsets'].iloc[i],
                                "text": self.eval_docs['text'].iloc[i],
                                "type": self.eval_docs['type'].iloc[i],
                                "db_ids": list(self.cui_labels[i]),
                                "split": self.eval_docs['split'].iloc[i],
                                "deabbreviated_text":self.eval_docs['deabbreviated_text'].iloc[i],
                                "mention_id": self.eval_docs['mention_id'].iloc[i] + ".abbr_resolved",
                                "candidates": cui_result,
                                "candidates_metadata": candidates,
                            })
                    else:
                        output.append(
                        {
                            "document_id": self.eval_docs['document_id'].iloc[i],
                            "offsets": self.eval_docs['offsets'].iloc[i],
                            "text": self.eval_docs['text'].iloc[i],
                            "type": self.eval_docs['type'].iloc[i],
                            "db_ids": list(self.cui_labels[i]),
                            "split": self.eval_docs['split'].iloc[i],
                            "mention_id": self.eval_docs['mention_id'].iloc[i],
                            "candidates": cui_result,
                            "candidates_metadata": candidates,
                        })
                    # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
                    # print(posi_scores)
                    # print(result)
                    # print(cui_result)
                    # print(self.cui_labels[i])
                    # print(self.cui2str[self.cui_labels[i]])
                    # input()

                    if self.cui_labels[i].intersection(set(cui_result[0])):
                        count_top1 += 1
                        count_top5 += 1
                    elif self.cui_labels[i].intersection(set(sum(cui_result,[]))):
                        count_top5 += 1

                if i % 50 == 49:
                    print('=============Top1 Precision:\t',count_top1/(i+1))
                    print('=============Top5 Precision:\t',count_top5/(i+1))

                input_ids = []
                decoder_input_ids = []
                attention_mask = []
        
        print("last")
        print('=============Top1 Precision:\t',count_top1/(i+1))
        print('=============Top5 Precision:\t',count_top5/(i+1))

        with open('./logs.txt', 'a+') as f:
            f.write(str(self.config.seed) + '======\n')
            f.write(self.config.model_load_path + '\n')
            f.write('Top1 Precision:\t'+str(count_top1/(i+1))+'\n')
            f.write('Top5 Precision:\t'+str(count_top5/(i+1))+'\n\n')

        with open(os.path.join(self.config.model_load_path, 'output.json'), 'w') as f:
            f.write(ujson.dumps(output, indent=2))
        
        if self.config.testset:

            with open(os.path.join(self.config.model_load_path, 'results_test.pkl'), 'wb') as f:
                pickle.dump([results, results_score], f)

        else:

            with open(os.path.join(self.config.model_load_path, 'results_dev.pkl'), 'wb') as f:
                pickle.dump([results, results_score], f)
    
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.dev_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training configuration')

    parser.add_argument("dataset_path",type=str,
                        help="path of the medmentions dataset")
    parser.add_argument("-model_save_path",type=str,default='./model_saved',
                        help="path of the pretrained model")
    parser.add_argument("-trie_path",type=str,default='./trie.pkl',
                        help="path of the Trie")
    parser.add_argument("-dict_path",type=str,default='./trie.pkl',
                        help="path of the cui2str dictionary")
    parser.add_argument("-retrieved_path",type=str,default='./trie.pkl',
                        help="path of the cui2str dictionary")
    parser.add_argument("-model_load_path",type=str,default='facebook/bart-large',
                        help="path of the pretrained model")
    parser.add_argument("-model_token_path",type=str,default='facebook/bart-large',
                        help="path of the pretrained model")
    parser.add_argument("-logging_path",type=str, default='./logs',
                        help="path of saved logs")
    parser.add_argument('-logging_steps', type=int, default=500,
                        help='save logs per logging step')                  
    parser.add_argument('-save_steps', type=int, default = 20000,
                        help='save checkpoints per save steps')
    parser.add_argument('-num_train_epochs', type=int, default = 8,
                        help="number of training epochs")
    parser.add_argument('-per_device_train_batch_size', type=int, default = 4,
                        help='training batch size')
    parser.add_argument('-per_device_eval_batch_size', type=int, default = 5,
                        help='evaluation batch size')
    parser.add_argument('-warmup_steps', type=int, default = 500,
                        help='warmup steps')
    parser.add_argument('-finetune', action='store_true',
                        help='if finetune the bart params')
    parser.add_argument('-t5', action='store_true',
                        help='if use t5 pretrained model')
    parser.add_argument('-fairseq_loss', action='store_true',
                        help='if use label smoothed loss in fairseq')
    parser.add_argument('-evaluation', action='store_true',
                        help='whether to train or evaluate')
    parser.add_argument('-testset', action='store_true',
                        help='whether evaluate with testset or devset')
    parser.add_argument('-load_prompt', action='store_true',
                        help='whether to load prompt')
    parser.add_argument('-weight_decay', type=float, default = 0.01,
                        help='weigth decay of optimizer')
    parser.add_argument('-length_penalty', type=float, default = 1,
                        help='length penaltyof beam search')
    parser.add_argument('-beam_threshold', type=float, default = 0,
                        help='logit threshold of beam search')
    parser.add_argument('-unlikelihood_loss', action='store_true',
                        help='whether using unlikelihood loss')
    parser.add_argument('-init_lr', type=float, default = 5e-5,
                        help='initial learning rate of AdamW')
    parser.add_argument('-evaluation_strategy', type=str, default='no',
                        help='evaluation strategy')
    parser.add_argument('-prompt_tokens_enc', type=int, default = 0,
                        help='a tuple containing number of soft prompt tokens in encoder and decoder respectively')
    parser.add_argument('-prompt_tokens_dec', type=int, default = 0,
                        help='a tuple containing number of soft prompt tokens in encoder and decoder respectively')
    parser.add_argument('-seed', type=int, default = 42,
                        help='the seed of huggingface seq2seq training, 42 is also the default of huggingface train default')
    parser.add_argument('-label_smoothing_factor', type=float, default = 0.1,
                        help='label smoothig factor')
    parser.add_argument('-unlikelihood_weight', type=float, default = 0.1,
                        help='label smoothig factor')
    parser.add_argument('-max_grad_norm', type=float, default = 0.1,
                        help='gradient clipping value')
    parser.add_argument('-max_steps', type=int, default = 200000,
                        help='max training steps override num_train_epoch')
    parser.add_argument('-gradient_accumulate', type=int, default = 1,
                        help='max training steps override num_train_epoch')
    parser.add_argument('-lr_scheduler_type', type=str, default = 'polynomial',
                        help='the learning rate schedule type')
    parser.add_argument('-attention_dropout', type=float, default = 0.1,
                        help='the attention dropout')
    parser.add_argument('-dropout', type=float, default = 0.1,
                        help='dropout')
    parser.add_argument('-max_position_embeddings', type=int, default = 1024,
                        help="the max length for position embedding")
    parser.add_argument('-num_beams', type=int, default = 5,
                        help='the attention dropout')
    parser.add_argument('-max_length', type=int, default = 1024,
                        help='the attention dropout')
    parser.add_argument('-min_length', type=int, default = 1,
                        help='the attention dropout')
    parser.add_argument('-sample_train', action='store_true',
                        help='if to use training target sampled by tfidf similarity')
    parser.add_argument('-prefix_prompt', action='store_true',
                        help='wheather use prefix prompt tokens')
    parser.add_argument('-rerank', action='store_true',
                        help='wheather to rerank the retrieved names')
    parser.add_argument('-init_from_vocab', action='store_true',
                        help='wheather initialize prompt from the mean of token embeddings')
    parser.add_argument('-no_finetune_decoder', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-syn_pretrain', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-gold_sty', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-prefix_mention_is', action='store_true',
                        help='whether only finetune encoder')
    parser.add_argument('-rdrop', type=float, default=0.0)
    parser.add_argument('-preprocess_data', type=bool, default = False,
                        help='preprocess the data or not')
