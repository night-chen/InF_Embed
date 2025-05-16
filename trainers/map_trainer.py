import os
import torch

from typing import Optional, Dict, Tuple
from transformers.trainer import Trainer

from logger_config import logger
from metrics import accuracy, batch_mrr
from models.basic_model import BasicOutput
from models.map_model import MapModel
from utils import AverageMeter
import wandb
from evals.mteb_eval import evaluate_fn
from tqdm import tqdm

def _unpack_iqp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    i_prefix, q_prefix, p_prefix = 'i_', 'q_', 'p_'
    instruction_batch_dict = {k[len(i_prefix):]: v for k, v in inputs.items() if k.startswith(i_prefix)}
    query_batch_dict = {k[len(q_prefix):]: v for k, v in inputs.items() if k.startswith(q_prefix)}
    passage_batch_dict = {k[len(p_prefix):]: v for k, v in inputs.items() if k.startswith(p_prefix)}
    if not instruction_batch_dict:
        instruction_batch_dict = None
    if not query_batch_dict:
        query_batch_dict = None
    if not passage_batch_dict:
        passage_batch_dict = None

    return instruction_batch_dict, query_batch_dict, passage_batch_dict


class MapTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(MapTrainer, self).__init__(*pargs, **kwargs)
        self.model: MapModel

        self.acc1_meter = AverageMeter('Acc@1', round_digits=2)
        self.acc3_meter = AverageMeter('Acc@3', round_digits=2)
        self.mrr_meter = AverageMeter('mrr', round_digits=2)
        self.last_epoch = 0
        print(self.accelerator)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            if self.model.uni_encoder:
                self.tokenizer.save_pretrained(output_dir)
            else:
                self.tokenizer.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(self.model.query_model_save_path)
                self.tokenizer.save_pretrained(self.model.passage_model_save_path)
        # Log model checkpoint to wandb
        if self.args.report_to and "wandb" in self.args.report_to:
            wandb.save(os.path.join(output_dir, "*"))

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        instruction, query, passage = _unpack_iqp(inputs)
        outputs: BasicOutput = model(instruction=instruction, query=query, passage=passage)
        loss = outputs.loss

        self.accelerator.wait_for_everyone()

        if self.model.training:
            step_acc1, step_acc3 = accuracy(output=outputs.scores.detach(), target=outputs.labels, topk=(1, 3))
            # print(outputs.scores.shape, outputs.labels.shape)
            step_mrr = batch_mrr(output=outputs.scores.detach(), target=outputs.labels)

            self.acc1_meter.update(step_acc1)
            self.acc3_meter.update(step_acc3)
            self.mrr_meter.update(step_mrr)

            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                log_info = ', '.join(map(str, [self.mrr_meter, self.acc1_meter, self.acc3_meter]))
                logger.info('step: {}, {}'.format(self.state.global_step, log_info))


                self.log({
                    "self/loss_self": loss.item(),
                    "mrr": self.mrr_meter.avg,
                    "acc@1": self.acc1_meter.avg,
                    "acc@3": self.acc3_meter.avg,
                    "self/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "self/epoch": self.state.epoch,
                    "self/global_step": self.state.global_step,
                })

            if self.state.global_step > 0 and self.state.global_step % self.args.eval_steps == 0:
                encode_kwargs = {
                    "batch_size": self.args.per_device_test_batch_size, 
                    "num_workers": self.args.num_workers,
                    "accelerator": self.accelerator
                }
                eval_results = {}
                unwrapped_model = self.accelerator.unwrap_model(self.model)

                eval_results = evaluate_fn(unwrapped_model, self.accelerator.project_dir, self.state.epoch, encode_kwargs)
                for key, value in eval_results.items():
                    self.log({"eval/" + key: value})
                self.log(eval_results)
                self.accelerator.print(
                    f"Evaluation results: {eval_results}"
                )
            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.acc1_meter.reset()
            self.acc3_meter.reset()
            self.mrr_meter.reset()