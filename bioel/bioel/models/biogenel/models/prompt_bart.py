import os
import sys
# import Path
import numpy as np
import tqdm
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import BartForConditionalGeneration
from bioel.models.biogenel.models.bart_model import BartForConditionalGeneration

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.file_utils import ModelOutput
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.utils import logging
from transformers.generation.utils import *
from bioel.models.biogenel.models.beam_scorer import BeamScorer, BeamSearchScorer


class BartEntityPromptModel(BartForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        config,
        vocabs: list = [[],[]],
        finetune: bool = False,
        n_tokens: tuple = (10, 10),
        load_prompt: bool = False,
        soft_prompt_path: str = None, 
        initialize_from_vocab: bool = False,
        random_range: float = 0.5,
        no_finetune_decoder: bool=False,
        **kwargs,
    ):

        model = super().from_pretrained(pretrained_model_name_or_path, config = config)

        if not finetune:
            print('no finetune!')
            for param in model.parameters():
                param.requires_grad = False

        if no_finetune_decoder:
            print('not finetune decoder!')
            for param in model.model.decoder.parameters():
                param.requires_grad = False
            for param in model.model.encoder.embed_tokens.parameters():
                param.requires_grad = False
            for param in model.model.shared.parameters():
                param.requires_grad = False
            for param in model.lm_head.parameters():
                param.requires_grad = False
            
        #initialize soft prompt
        model.set_prompt_number(n_tokens)

        if n_tokens[0] > 0 and n_tokens[1] > 0:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                vocabs=vocabs,
                random_range=random_range,
            )
            #load soft prompts
            if load_prompt:
                print('Loading soft prompts.....')
                model.set_soft_prompt_embeds(soft_prompt_path)
        else:
            print('Setting no soft prompts!')

        return model

    def set_prompt_number(
        self,
        n_tokens: tuple,
        ):
        self.n_tokens_enc = n_tokens[0]
        self.n_tokens_dec = n_tokens[1]
        self.soft_prompt_enc = None
        self.soft_prompt_dec = None

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ):
        params = torch.load(os.path.join(soft_prompt_path, 'pytorch_model.bin'))
        if 'soft_prompt_enc.weight' in params:
            self.soft_prompt_enc.weight.data = params['soft_prompt_enc.weight']
            self.soft_prompt_dec.weight.data = params['soft_prompt_dec.weight']
            print('Load from pytorch_model.bin')
        elif os.path.exists(os.path.join(soft_prompt_path, 'soft_prompt_enc.pth')):
            self.soft_prompt_enc = torch.load(os.path.join(soft_prompt_path, 'soft_prompt_enc.pth'), map_location=torch.device(self.device))
            self.soft_prompt_dec = torch.load(os.path.join(soft_prompt_path, 'soft_prompt_dec.pth'), map_location=torch.device(self.device))
        else:
            print('Find no soft prompt parameter in the path')
        print(f"Loaded encoder soft prompt! (n_tokens: {self.n_tokens_enc})")
        print(f"Loaded decoder soft prompt! (n_tokens: {self.n_tokens_dec})")

    def save_soft_prompt(
        self, 
        path: str
        ):
        # Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt_enc, os.path.join(path, 'soft_prompt_enc.pth'))
        torch.save(self.soft_prompt_dec, os.path.join(path, 'soft_prompt_dec.pth'))

    def initialize_soft_prompt(
        self,
        n_tokens: tuple,
        initialize_from_vocab: bool,
        vocabs: list,
        random_range: float = 0.5,
    ):
        if initialize_from_vocab:
            init_prompt_value_enc = self.model.encoder.embed_tokens(vocabs[0]).data
            init_prompt_value_dec = self.model.decoder.embed_tokens(vocabs[1]).data
        else:
            init_prompt_value_enc = torch.FloatTensor(self.n_tokens_enc, self.config.d_model).uniform_(
                -random_range, random_range
            )
            init_prompt_value_dec = torch.FloatTensor(self.n_tokens_dec, self.config.d_model).uniform_(
                -random_range, random_range
            )
        self.soft_prompt_enc = nn.Embedding(self.n_tokens_enc, self.config.d_model)
        self.soft_prompt_dec = nn.Embedding(self.n_tokens_dec, self.config.d_model)
        print(f"Set encoder soft prompt! (n_tokens: {self.n_tokens_enc})")
        print(f"Set decoder soft prompt! (n_tokens: {self.n_tokens_dec})")
    
    def id2emb_cat_prompt(self, input_ids, coder_part = 'encoder', insert_idx = None):
        
        if coder_part == 'encoder':
            coder = self.model.encoder
            n_tokens = self.n_tokens_enc
            prompt_embeds = self.soft_prompt_enc
        else:
            coder = self.model.decoder
            n_tokens = self.n_tokens_dec
            prompt_embeds = self.soft_prompt_dec

        inputs_embeds = coder.embed_tokens(input_ids)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if n_tokens > 0:
            prompt_embeds = prompt_embeds.weight.repeat(input_ids.size(0), 1, 1)
            if insert_idx is not None:
                inputs = []
                for i in range(inputs_embeds.shape[0]):
                    inputs.append(torch.cat([inputs_embeds[i,0:insert_idx[i][0],:], 
                                                prompt_embeds[i,:n_tokens//2,:], 
                                                inputs_embeds[i,insert_idx[i][0]:insert_idx[i][1],:],
                                                prompt_embeds[i,n_tokens//2:,:], 
                                                inputs_embeds[i,insert_idx[i][1]:,:]], dim=0) )
                inputs_embeds = torch.stack(inputs)

            else:
                inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
                
        return inputs_embeds.to(self.device)
    
    def extend_attention_mask(
        self, 
        attention_mask,
        n_tokens,
        ):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if n_tokens > 0:
            attention_mask = torch.cat([torch.full((attention_mask.shape[0], n_tokens), 1).to(self.device), attention_mask],dim=1)

        return attention_mask.to(self.device)

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        labels=None,
        attention_mask=None,
        decoder_attention_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        encoder_outputs=None,
        use_cache=None,
        return_dict=None,
        mention_indexes=None,
        **kwargs,
        ):

        if encoder_outputs is None:
            if input_ids is not None:
                inputs_embeds_enc = self.id2emb_cat_prompt(input_ids, coder_part = 'encoder', insert_idx = mention_indexes)
            else:
                inputs_embeds_enc = inputs_embeds.to(self.device)
            
            if decoder_input_ids is not None:
                inputs_embeds_dec = self.id2emb_cat_prompt(decoder_input_ids, coder_part = 'decoder')
            else:
                inputs_embeds_dec = decoder_inputs_embeds.to(self.device)
            
            if attention_mask is not None:
                attention_mask_enc = self.extend_attention_mask(attention_mask, self.n_tokens_enc)
            else:
                attention_mask_enc = attention_mask
            
            if decoder_attention_mask is not None:
                attention_mask_dec = self.extend_attention_mask(decoder_attention_mask, self.n_tokens_dec)
            else:
                attention_mask_dec = decoder_attention_mask
            
            return super().forward(

                attention_mask=attention_mask_enc,
                decoder_attention_mask=attention_mask_dec,

                inputs_embeds=inputs_embeds_enc,
                decoder_inputs_embeds=inputs_embeds_dec,

                labels=labels,

                use_cache=use_cache,
                return_dict=return_dict,

                **kwargs,
            )
                
        else:
            
            if decoder_input_ids is not None:
                inputs_embeds_dec = self.id2emb_cat_prompt(decoder_input_ids, coder_part = 'decoder')
            else:
                inputs_embeds_dec = decoder_inputs_embeds.to(self.device)

            if decoder_attention_mask is not None:
                attention_mask_dec = self.extend_attention_mask(decoder_attention_mask, self.n_tokens_dec)
            else:
                attention_mask_dec = decoder_attention_mask

            return super().forward(
                input_ids=None,
                decoder_input_ids=None,

                attention_mask=None,
                decoder_attention_mask=attention_mask_dec,

                inputs_embeds=None,
                decoder_inputs_embeds=inputs_embeds_dec,
                encoder_outputs=encoder_outputs,

                use_cache=use_cache,
                return_dict=return_dict,

                **kwargs,
            )
