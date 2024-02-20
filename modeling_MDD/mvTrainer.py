#%%
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer, is_apex_available
from typing import Any, Dict, List, Optional, Union, Tuple
from torch.nn import CrossEntropyLoss
from torch.nn import CTCLoss
import torch.nn.functional as F
from itertools import chain
import numpy as np


def compute_lengths(seq_tensor):
    """
    Compute sequence lengths.
    """
    lengths = []
    for seq in seq_tensor:
        neg_index = torch.nonzero(seq < 0)
        if neg_index.numel() > 0:
            neg_index = neg_index[0, 0].item()
            seq_len = neg_index
        else:
            seq_len = seq.numel()
        lengths.append(seq_len)
    return torch.tensor(lengths)


def backward_loss_logits(model, inputs, eval=False):
    """
    Get model outputs loss, logits during training or evaluation.
    """
    
    loss_fn = CTCLoss(
        blank=0
    )
    
    if eval:
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
    else:
        outputs = model(inputs)


    ## Logits from the model (only PR --Phoneme Recognition Logits)    
    logits = outputs.get("logits")
    logits_pr = logits["PR"]
    
    ## True Labels sequence
    labels_pr = inputs.get("labels_p")

    log_probs = F.log_softmax(logits_pr, dim=-1)


    input_lengths = torch.LongTensor([len(b) for b in log_probs])   
    log_probs = log_probs.permute(1, 0, 2)
    target_lengths = compute_lengths(labels_pr)
    mask = labels_pr >= 0
    
    labels_pr_masked = labels_pr[mask]
    loss = loss_fn(log_probs, labels_pr_masked, input_lengths, target_lengths) 

    return loss, log_probs.permute(1, 0, 2)


class ModelTrainerBase(Trainer):
    """
    Custom Trainer class for training a PyTorch model.
    """
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:        
        """
        Perform a training step on a batch of inputs.
        Args:
            model (:obj:`nn.Module`): The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.
        Return:
            :obj:`torch.Tensor`: The training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        return loss.detach() 


class ModelTrainerJoint(ModelTrainerBase):
    """
    Custom Trainer class for joint training.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss during training.
        """
        loss, logits = backward_loss_logits(model, inputs)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self ,model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],  prediction_loss_only: bool,
                    ignore_keys: Optional[List[str]] = None,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step on a batch of inputs.
        """
        inputs = self._prepare_inputs(inputs)
        loss, logits_ = backward_loss_logits(model, inputs, eval=True)
        
        labels = {}
        labels['logits'] = logits_
        labels['targets'] = inputs.get("labels_p")        
        labels['targets_ref'] = inputs.get("labels_ref")  
        return (loss, logits_, labels)
