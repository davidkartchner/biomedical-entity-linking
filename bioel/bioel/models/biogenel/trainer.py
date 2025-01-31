from transformers import Seq2SeqTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.trainer import *
from dataclasses import dataclass
from torch.utils.data.distributed import DistributedSampler
from transformers import is_torch_tpu_available
from torch.cuda.amp import autocast
from tqdm import tqdm

# import ipdb

import collections

import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class modifiedLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    fairseq: bool = False
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        labels = labels.to(
            torch.int64
        )  # I had to add this line to cast to int64, originally not in the code

        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()

        ## fairseq implementation
        if self.fairseq:
            nll_loss = nll_loss.sum()
            eps_i = self.epsilon / (log_probs.size(-1) - 1)
            loss = (1.0 - self.epsilon - eps_i) * nll_loss + eps_i * smoothed_loss.sum()
            return loss / num_active_elements, nll_loss / num_active_elements

        else:
            # huggingface implementation
            # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
            nll_loss = nll_loss.sum() / num_active_elements
            smoothed_loss = smoothed_loss.sum() / (
                num_active_elements * log_probs.shape[-1]
            )
            return (
                1 - self.epsilon
            ) * nll_loss + self.epsilon * smoothed_loss, nll_loss


# @dataclass
class modifiedUNlikelihood:

    def __init__(self, alpha=0.1, prefix_allowed_tokens_fn=None):
        self.alpha = alpha
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def __call__(self, model_output, labels):
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
        unlikelihood = -torch.log(
            1 - nn.functional.softmax(logits, dim=-1).clamp(min=0.0, max=0.9999)
        )
        loss = 0
        count = 0

        for batch_i, sent in enumerate(labels):
            sent, anchor_idx = self.get_label(sent)
            sent = torch.tensor([16] + list(sent.cpu()))
            unlikhod_toks = self.prefix_allowed_tokens_fn(batch_i, torch.tensor([16]))
            i = 1
            while len(unlikhod_toks) > 1:
                unlikhod_toks.remove(sent[i])
                loss += torch.max(
                    unlikelihood[batch_i, anchor_idx + i - 1, unlikhod_toks]
                )
                i += 1
                count += 1  # len(unlikhod_toks)
                unlikhod_toks = self.prefix_allowed_tokens_fn(batch_i, sent[:i])
        loss = self.alpha * loss / count
        return loss

    def get_label(self, labels):
        valid = labels.ne(-100)
        return labels[valid], torch.min(torch.where(valid)[0])


def compute_kl_loss(p, q, pad_mask=None):
    log_p = p["logits"] if isinstance(p, dict) else p[0]
    log_q = q["logits"] if isinstance(q, dict) else q[0]

    p_loss = F.kl_div(
        F.log_softmax(log_p, dim=-1), F.softmax(log_q, dim=-1), reduction="none"
    ).sum(-1)
    q_loss = F.kl_div(
        F.log_softmax(log_q, dim=-1), F.softmax(log_p, dim=-1), reduction="none"
    ).sum(-1)

    # pad_mask is for seq-level tasks
    # import ipdb; ipdb.set_trace()
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.0)
        q_loss.masked_fill_(pad_mask, 0.0)

    if pad_mask is None:
        num_active = torch.ones_like(p_loss)
    else:
        num_active = torch.ones_like(pad_mask).sum() - pad_mask.float().sum()

    if num_active > 0.0:
        p_loss = p_loss.sum() / num_active
        q_loss = q_loss.sum() / num_active

        loss = (p_loss + q_loss) / 2
    else:
        loss = 0.0

    return loss


class modifiedSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        fairseq_loss=False,
        use_amp=False,
        enc_num=0,
        dec_num=0,
        prefix_allowed_tokens_fn=None,
        unlikelihood_weight=0.1,
        rdrop=0.0,
        generation_config=None,
        **kwargs,
    ):  # I had to add the use_amp arg and scaler

        super().__init__(**kwargs)
        self.label_smoother = modifiedLabelSmoother(
            epsilon=self.args.label_smoothing_factor, fairseq=fairseq_loss
        )
        if prefix_allowed_tokens_fn:
            self.unlikelihood_loss = modifiedUNlikelihood(
                alpha=unlikelihood_weight,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        else:
            self.unlikelihood_loss = None
        self.n_tokens_dec = dec_num
        self.n_tokens_enc = enc_num
        self.rdrop = rdrop
        self.use_amp = use_amp

    def reform_input(self, inputs, eos_token):

        enc_max_idx = torch.max(torch.where(inputs["input_ids"] == eos_token)[1]) + 1
        label_max_idx = torch.max(torch.where(inputs["labels"] == eos_token)[1]) + 1
        dec_max_idx = label_max_idx - self.n_tokens_dec
        inputs["input_ids"] = inputs["input_ids"][:, :enc_max_idx]
        inputs["decoder_input_ids"] = inputs["decoder_input_ids"][:, :dec_max_idx]
        inputs["attention_mask"] = inputs["attention_mask"][:, :enc_max_idx]
        inputs["decoder_attention_mask"] = inputs["decoder_attention_mask"][
            :, :dec_max_idx
        ]
        inputs["labels"] = inputs["labels"][:, :label_max_idx]

        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        # model = model.module if isinstance(model, torch.nn.DataParallel) else model
        inputs = self.reform_input(inputs, model.config.eos_token_id)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.rdrop > 0.0:
            outputs2 = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss, nll_loss = self.label_smoother(outputs, labels)
            if self.rdrop > 0.0:
                loss2, nll_loss2 = self.label_smoother(outputs2, labels)
                loss = (loss + loss2) * 0.5
                nll_loss = (nll_loss + nll_loss2) * 0.5
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            nll_loss = loss

        if self.unlikelihood_loss:
            # if not labels:
            #     labels = inputs.pop("labels")
            # loss += self.unlikelihood_loss(outputs, labels)
            if not self.rdrop > 0.0:
                loss += self.unlikelihood_loss(outputs, labels)
            else:
                loss += 0.5 * (
                    self.unlikelihood_loss(outputs, labels)
                    + self.unlikelihood_loss(outputs2, labels)
                )

        if self.rdrop > 0.0:
            if labels is None:
                label_mask = None
            else:
                label_mask = labels.eq(-100)
            rdrop_loss = compute_kl_loss(outputs, outputs2, label_mask)
            loss += self.rdrop * rdrop_loss

        return (loss, outputs) if return_outputs else loss, nll_loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps, scaler=scaler
            )
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss, nll_loss = self.compute_loss(model, inputs)
        else:
            loss, nll_loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # ipdb.set_trace()

        return loss.detach(), nll_loss.detach()

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval, nll_loss=0
    ):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            logs["nll_loss"] = round(float(nll_loss), 4)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):

        resume_from_checkpoint = (
            None if not resume_from_checkpoint else resume_from_checkpoint
        )

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(
                f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}."
            )
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(
                    f"Can't find a valid checkpoint at {resume_from_checkpoint}"
                )

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(
                    os.path.join(resume_from_checkpoint, CONFIG_NAME)
                )
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME),
                    map_location="cpu",
                )
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )
        if train_dataset_is_sized:
            num_update_steps_per_epoch = (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        delay_optimizer_creation = False
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self,
                num_training_steps=max_steps,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch
                    )
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches"
                    )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = (
            self.hp_name(trial) if self.hp_name is not None else None
        )
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(
                    train_dataloader, [args.device]
                ).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, nll_loss = self.training_step(model, inputs)
                else:
                    tr_loss_step, nll_loss = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if (
                        args.max_grad_norm is not None
                        and args.max_grad_norm > 0
                        and not self.deepspeed
                    ):
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                (
                                    amp.master_params(self.optimizer)
                                    if self.use_apex
                                    else model.parameters()
                                ),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    self.model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss,
                        self.model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        nll_loss=nll_loss,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, model, trial, epoch, ignore_keys_for_eval, nll_loss=nll_loss
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(
                self.state.best_model_checkpoint, WEIGHTS_NAME
            )
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)
