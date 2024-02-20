import os
import csv
import torch
import librosa
import argparse
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoConfig, TrainingArguments
from modeling_MDD.create_dict_vocab import VocabDict
from modeling_MDD.tokenizer_extractor import TokenizerExtractor
from modeling_MDD.datacollator import MVDataCollatorBatchWisePadding
from modeling_MDD.mvModel import SSLforJointClassification, ModelTrainerJoint


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



def test_model(dir_vocab, model_checkpoint, test_csv):
    # Load vocab
    vocab_path = os.path.join(dir_vocab,'vocab.json')
    tokenizer_extractor = TokenizerExtractor(vocab_path)

    # Load test dataset
    data_files = {"test": test_csv}
    dataset = load_dataset('csv', data_files=data_files, delimiter=",", cache_dir="")
    test_dataset = dataset["test"]

    # Load model
    model_name = 'facebook/wav2vec2-large-robust'
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(tokenizer_extractor.tokenizer),
        finetuning_task="name_exp"
    )
    setattr(config, 'model_name_audio', model_name)
    setattr(config, 'vocab_size', len(tokenizer_extractor.tokenizer))
    setattr(config, 'classifier_proj_size', 1024)
    setattr(config, 'pad_token_id', tokenizer_extractor.tokenizer.pad_token_id)
    model = SSLforJointClassification(config=config)


    # Load checkpoint
    checkpoint = torch.load(os.path.join(model_checkpoint, 'pytorch_model.bin'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=model_checkpoint,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        gradient_checkpointing=True,
        num_train_epochs=150,
        save_steps=100,
        eval_steps=100,
        learning_rate=5e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_wer',
        greater_is_better=False,
        remove_unused_columns=False
    )

    # Load metric
    wer_metric = load_metric("wer")

    # Define compute metrics function
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.label_ids['logits'], axis=-1)
        targets_all = pred.label_ids['targets']
        ref_all = pred.label_ids['targets_ref']

        targets_all[targets_all < 0] = tokenizer_extractor.processor.tokenizer.pad_token_id
        ref_all[ref_all < 0] = tokenizer_extractor.processor.tokenizer.pad_token_id
        pred_ids[pred_ids < 0] = tokenizer_extractor.processor.tokenizer.pad_token_id

        pred_str = tokenizer_extractor.processor.batch_decode(pred_ids, group_tokens=True)
        label_str = tokenizer_extractor.processor.batch_decode(targets_all, group_tokens=True)
        label_str_ref = tokenizer_extractor.processor.batch_decode(ref_all, group_tokens=True)

        ids = test_dataset['path']
        combined_data = zip(ids, label_str, label_str_ref, pred_str)
        with open(os.path.join(model_checkpoint,'results_model.csv'), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row
            csv_writer.writerow(['path', 'anno', 'ref', 'predicted'])

            # Write the data rows
            csv_writer.writerows(combined_data)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Set up trainer
    trainer = ModelTrainerJoint(
        model=model,
        data_collator=MVDataCollatorBatchWisePadding(processor=tokenizer_extractor.processor, padding=True),
        args=training_args,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer_extractor.feature_extractor,
    )

    # Prepare dataset
    test_dataset = test_dataset.map(
            lambda batch: prepare_dataset_(batch, tokenizer_extractor),
            num_proc=8)
    

    # Run prediction
    out = trainer.predict(test_dataset=test_dataset)

    # Print metrics
    print(compute_metrics(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a pre-trained model on a test dataset.")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the pre-trained model checkpoint.")
    parser.add_argument("--test_csv", required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--dir_vocab", required=True, help="Path to the Vocab Dict.")

    args = parser.parse_args()
    test_model(args.dir_vocab, args.model_checkpoint, args.test_csv)
