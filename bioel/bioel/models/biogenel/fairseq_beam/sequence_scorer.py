import torch
import torch.nn.functional as F

@torch.no_grad()
def sequence_score(model, sample, len_penalty=1):
    """Score a batch of translations."""

    model.eval()
    hypos = []
    decoder_out = model(
                input_ids = sample['input_ids'],
                decoder_input_ids = sample['decoder_input_ids'],
                attention_mask = sample['attention_mask'],
                decoder_attention_mask = sample['decoder_attention_mask'],
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

    for b in range(sample['input_ids'].shape[0]):

        logit = decoder_out.logits[b, sample['anchor'][b]:sample['anchor'][b]+len(sample['labels'][b]), :]
        probs = F.log_softmax(logit, dim=-1, dtype=torch.float32)
        target_probs = probs.gather(-1, sample['labels'][b].reshape(-1,1))

        score_i = torch.sum(target_probs) / (len(target_probs)) ** len_penalty

        hypos.append(
            [
                {
                    "tokens": sample['labels'][b],
                    "score": score_i,
                    "positional_scores": target_probs,
                }
            ]
        )
    return hypos