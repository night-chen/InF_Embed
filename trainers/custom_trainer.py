import os
import torch

from typing import Optional, Dict, Tuple
from transformers.trainer import Trainer

from logger_config import logger
from metrics import accuracy, batch_mrr
from models.basic_model import BasicOutput, BasicModel
from utils import AverageMeter
import wandb

def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    iq_prefix, p_prefix = 'iq_', 'p_'
    instruction_query_batch_dict = {k[len(iq_prefix):]: v for k, v in inputs.items() if k.startswith(iq_prefix)}
    passage_batch_dict = {k[len(p_prefix):]: v for k, v in inputs.items() if k.startswith(p_prefix)}
    # neg_doc1_batch_dict = {k[len(ndi_prefix):]: v for k, v in inputs.items() if k.startswith(ndi_prefix)}
    # neg_doc2_batch_dict = {k[len(ndii_prefix):]: v for k, v in inputs.items() if k.startswith(ndii_prefix)}

    if not instruction_query_batch_dict:
        instruction_query_batch_dict = None
    if not passage_batch_dict:
        passage_batch_dict = None
    # if not neg_doc1_batch_dict:
    #     neg_doc1_batch_dict = None
    # if not neg_doc2_batch_dict:
    #     neg_doc2_batch_dict = None

    return instruction_query_batch_dict, passage_batch_dict


class CustomTrainer:
    def __init__(self, args, model, accelerator):


        self.acc1_meter = AverageMeter('Acc@1', round_digits=2)
        self.acc3_meter = AverageMeter('Acc@3', round_digits=2)
        self.mrr_meter = AverageMeter('mrr', round_digits=2)
        self.last_epoch = 0

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
        instruction_query, passage = _unpack_qp(inputs)
        outputs: BiencoderOutput = model(instruction_query=instruction_query, passage=passage)
        loss = outputs.loss

        if self.model.training:
            step_acc1, step_acc3 = accuracy(output=outputs.scores.detach(), target=outputs.labels, topk=(1, 3))
            step_mrr = batch_mrr(output=outputs.scores.detach(), target=outputs.labels)

            self.acc1_meter.update(step_acc1)
            self.acc3_meter.update(step_acc3)
            self.mrr_meter.update(step_mrr)

            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                log_info = ', '.join(map(str, [self.mrr_meter, self.acc1_meter, self.acc3_meter]))
                logger.info('step: {}, {}'.format(self.state.global_step, log_info))

                # log metrics to wandb
                if self.args.report_to and "wandb" in self.args.report_to:
                    wandb.log({
                        "loss": loss.item(),
                        "mrr": self.mrr_meter.avg,
                        "acc@1": self.acc1_meter.avg,
                        "acc@3": self.acc3_meter.avg,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": self.state.epoch,
                        "global_step": self.state.global_step,
                    })

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.acc1_meter.reset()
            self.acc3_meter.reset()
            self.mrr_meter.reset()