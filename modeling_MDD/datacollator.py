#%%
import torch
from transformers import AutoTokenizer, Wav2Vec2Processor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class MVDataCollatorBatchWisePadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        
        
        ## input_features is the speech input to the model
        ## phoneme label features are the correct pronounced phonemes
        ## phoneme label ref are the given phonemes to the speaker
        ## LID are language identification features:
        ##                  1. just a simple ont-hot encoding
        ##                  2. extracted from a model trained for accent identification
        
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        phone_label_features = [{"input_ids": feature["labels_p"]} for feature in features]
        ref_label_features = [{"input_ids": feature["labels_p_ref"]} for feature in features]
        L_ids = [feature["LID"] for feature in features]


        ## pad the input_features with 0
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        ## pad phone labels with -100
        with self.processor.as_target_processor():
            labels_batch_p = self.processor.pad(
                phone_label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
        )
            labels_batch_p_ref = self.processor.pad(
                ref_label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels_batch_p = labels_batch_p["input_ids"].masked_fill(labels_batch_p.attention_mask.ne(1), -100)
        labels_batch_p_ref = labels_batch_p_ref["input_ids"].masked_fill(labels_batch_p_ref.attention_mask.ne(1), -100)

        batch["labels_p"] = labels_batch_p
        batch["labels_ref"] = labels_batch_p_ref
        batch["labels_LID"] = L_ids
        
        return batch 
