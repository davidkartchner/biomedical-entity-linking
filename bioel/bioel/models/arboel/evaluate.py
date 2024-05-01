from bioel.models.arboel.model.common.params import BlinkParser

import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from bioel.models.arboel.model.LightningModule import LitArboel
from bioel.models.arboel.data.LightningDataModule import ArboelDataModule


def main(args):
    data_module = ArboelDataModule(params=args)

    MyModel = LitArboel.load_from_checkpoint(
        params=args, checkpoint_path=args["model_checkpoint"]
    )

    trainer = L.Trainer(
        limit_test_batches=1,  # for ncbi_disease
        devices=args["devices"],
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        precision="16-mixed",
    )

    trainer.test(model=MyModel, datamodule=data_module)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)

def evaluate(
    reranker,
    eval_dataloader,
    logger,
    context_length,
    device,
    silent=True,
    unfiltered_length=None,
    mention_data=None,
    compute_macro_avg=False,
    store_failure_success=False,
):
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    if mention_data is not None:
        processed_mention_data = mention_data["mention_data"]
        n_mentions_per_type = collections.defaultdict(int)
        if compute_macro_avg:
            for men in processed_mention_data:
                n_mentions_per_type[men["type"]] += 1
        dictionary = mention_data["entity_dictionary"]
        stored_candidates = mention_data["stored_candidates"]
        if store_failure_success:
            failsucc = {"failure": [], "success": []}
    n_evaluated_per_type = collections.defaultdict(int)
    n_hits_per_type = collections.defaultdict(int)
    for step, batch in enumerate(iter_):
        context_input, label_input, mention_idxs = batch
        context_input = context_input.to(device)
        label_input = label_input.to(device)
        mention_idxs = mention_idxs.to(device)

        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        tmp_eval_hits, predicted = accuracy(logits, label_ids, return_bool_arr=True)
        tmp_eval_accuracy = np.sum(tmp_eval_hits)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

        if compute_macro_avg:
            for i, m_idx in enumerate(mention_idxs):
                type = processed_mention_data[m_idx]["type"]
                n_evaluated_per_type[type] += 1
                is_hit = tmp_eval_hits[i]
                n_hits_per_type[type] += is_hit

        if store_failure_success:
            for i, m_idx in enumerate(mention_idxs):
                m_idx = m_idx.item()
                men_query = processed_mention_data[m_idx]
                dict_pred = dictionary[
                    stored_candidates["candidates"][m_idx][predicted[i]]
                ]
                report_obj = {
                    "mention_id": men_query["mention_id"],
                    "mention_name": men_query["mention_name"],
                    "mention_gold_cui": "|".join(men_query["label_cuis"]),
                    "mention_gold_cui_name": "|".join(
                        [
                            dictionary[i]["title"]
                            for i in men_query["label_idxs"][: men_query["n_labels"]]
                        ]
                    ),
                    "predicted_name": dict_pred["title"],
                    "predicted_cui": dict_pred["cui"],
                }
                failsucc["success" if tmp_eval_hits[i] else "failure"].append(
                    report_obj
                )

    results["filtered_length"] = nb_eval_examples
    normalized_eval_accuracy = 100 * eval_accuracy / nb_eval_examples
    results["normalized_accuracy"] = normalized_eval_accuracy

    print(f"Eval: Best accuracy: {normalized_eval_accuracy}%")

    if unfiltered_length is not None:
        results["unfiltered_length"] = unfiltered_length
        results["unnormalized_accuracy"] = eval_accuracy / unfiltered_length
    if compute_macro_avg:
        norm_macro, unnorm_macro = 0, 0
        for type in n_mentions_per_type:
            norm_macro += n_hits_per_type[type] / n_evaluated_per_type[type]
            unnorm_macro += n_hits_per_type[type] / n_mentions_per_type[type]
        norm_macro /= len(n_evaluated_per_type)
        unnorm_macro /= len(n_mentions_per_type)
        results["normalized_macro_avg_acc"] = norm_macro
        results["unnormalized_macro_avg_acc"] = unnorm_macro
    if store_failure_success:
        results["failure"] = failsucc["failure"]
        results["success"] = failsucc["success"]
    if not store_failure_success:
        logger.info(json.dumps(results))

    print("results :", results)
    return results