# trainer_module/trainer.py
from transformers import EarlyStoppingCallback, TrainingArguments
from datasets import load_dataset, load_metric
from modeling_MDD.create_dict_vocab import VocabDict
from modeling_MDD.tokenizer_extractor import TokenizerExtractor
from modeling_MDD.datacollator import MVDataCollatorBatchWisePadding
from modeling_MDD.mvModel import SSLforJointClassification
from transformers import AutoConfig
from modeling_MDD.mvTrainer import ModelTrainerJoint
import wandb
import json
import os
import numpy as np
import librosa
from typing import List, Dict, Union

# Preprocess datasets
def speech_file_to_array_fn(path):
    """
    Load an audio file at the specified path and return the audio array and sampling rate.
    Returns:
        tuple: A tuple containing the audio array and sampling rate.
    """
    speech_array, sampling_rate = librosa.load(path, sr=16_000)
    return speech_array

def prepare_dataset_(batch, tokenizer):
    """
    Prepare a dataset batch for training or evaluation.
    Returns:
        dict: Prepared batch containing input values, phoneme labels, and language ID.
    """
    audio = batch["path"]
    array = speech_file_to_array_fn(audio)
    batch["input_values"] = tokenizer.processor(array, sampling_rate=16000).input_values[0]
    
    with tokenizer.processor.as_target_processor():
        batch["labels_p"] = tokenizer.processor(batch['ref_anno']).input_ids
        batch["labels_p_ref"] = tokenizer.processor(batch['ref_ref']).input_ids

    if 'L2_ARCTIC' in audio:
        batch["LID"] = 0
    elif 'data_QU' in audio:
        batch["LID"] = 1
    else:
        batch["LID"] = 2

    return batch


def train_model(config: Dict[str, Union[str, int, float, bool, Dict]]):

    # Create vocab
    create_vocab = VocabDict(root_dir=config["root_dir"], out_dir=config["out_dir"], label=config["label"])
    create_vocab.process()

    # Initialize tokenizer extractor
    tokenizer_extractor = TokenizerExtractor(config["vocab_path"])

    # Load datasets
    data_files = {
        "train": os.path.join(config["data_dir"], "train.csv"),
        "eval": os.path.join(config["data_dir"], "eval.csv"),
        "test": os.path.join(config["data_dir"], "test.csv")
    }
    
    dataset = load_dataset('csv', data_files=data_files, delimiter=",", cache_dir="")
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    test_dataset = dataset["test"]

    ## Preprocess datasets and extract info
    train_dataset = train_dataset.map(
            lambda batch: prepare_dataset_(batch, tokenizer_extractor),
            num_proc=8)

    test_dataset = test_dataset.map(
            lambda batch: prepare_dataset_(batch, tokenizer_extractor),
            num_proc=8)
    
    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset_(batch, tokenizer_extractor),
        num_proc=8)
    
    
    # Data collator
    data_collator_ = MVDataCollatorBatchWisePadding(processor=tokenizer_extractor.processor, padding=True)

    # Model configuration
    config_model = AutoConfig.from_pretrained(
        config["model_name"],
        num_labels=len(tokenizer_extractor.tokenizer),
        finetuning_task="name_exp"
    )
    
    
    setattr(config_model, 'model_name_audio', config["model_name"])
    setattr(config_model, 'vocab_size', len(tokenizer_extractor.tokenizer))
    setattr(config_model, 'classifier_proj_size', config["classifier_proj_size"])
    setattr(config_model, 'pad_token_id', tokenizer_extractor.tokenizer.pad_token_id)

    ## Intialize the model
    model = SSLforJointClassification(config=config_model)
    ## Freeze CNN layers
    model.freeze_feature_extractor()

    # Training arguments
    training_args = TrainingArguments(**config["training_args"])

    # WandB initialization
    wandb.init(
        name=config["project_name"],
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        config=config["training_args"]
    )

    # Metric initialization
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.label_ids['logits'], axis=-1)
        targets_all = pred.label_ids['targets']
        targets_all[targets_all < 0] = tokenizer_extractor.processor.tokenizer.pad_token_id
        pred_str = tokenizer_extractor.processor.batch_decode(pred_ids, group_tokens=True)
        label_str = tokenizer_extractor.processor.batch_decode(targets_all, group_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        wandb.log({"wer": wer})
        return {"wer": wer}

    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
    )

    # Trainer initialization
    trainer = ModelTrainerJoint(
        model=model,
        data_collator=data_collator_,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer_extractor.feature_extractor,
        callbacks=[early_stopping],
    )

    # Model training
    trainer.train()
    wandb.finish()
