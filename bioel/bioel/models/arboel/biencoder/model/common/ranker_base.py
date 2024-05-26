# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn

from IPython import embed

import torch


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class BertEncoder(nn.Module):
    def __init__(
        self,
        bert_model,
        output_dim,
        layer_pulled=-1,
        add_linear=None,
        get_all_outputs=False,
    ):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        self.get_all_outputs = get_all_outputs
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        outputs = self.bert_model(
            input_ids=token_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        output_bert = outputs.last_hidden_state
        output_pooler = outputs.pooler_output

        if self.get_all_outputs:
            result = output_bert
        else:
            # get embedding of [CLS] token
            if self.additional_linear is not None:
                embeddings = output_pooler
            else:
                embeddings = output_bert[:, 0, :]

            # in case of dimensionality reduction
            if self.additional_linear is not None:
                result = self.additional_linear(self.dropout(embeddings))
            else:
                result = embeddings

        return result
