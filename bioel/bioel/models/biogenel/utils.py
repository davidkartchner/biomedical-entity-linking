import torch


def reform_input(inputs, attention_mask = None, ending_token = 2):
    
    ## input a tensor of size BSZ x Length
    max_idx = torch.max(torch.where(inputs==ending_token)[1])
    inputs = inputs[:, :max_idx+1]
    if attention_mask is not None:
        attention_mask = attention_mask[:, :max_idx+1]

    return inputs, attention_mask