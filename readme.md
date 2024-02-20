# MDD Phoneme Recognition -- Multilingual wisth LID

## L1-AWARE MULTILINGUAL MISPRONUNCIATION DETECTION FRAMEWORK

L1-MultiMDD, enriched with L1-aware speech representation. An end-to-end speech encoder is trained on the input signal and corresponding reference phoneme sequence. First, an attention mechanism is deployed to align the input audio with the reference phoneme sequence. Afterward, the L1-L2-speech embedding is extracted from an auxiliary model, pretrained in a multi-task setup identifying L1 and L2 language, and infused with the primary network. Finally, the L1-MultiMDD is optimized for a unified multilingual phoneme recognition task using connectionist temporal classification (CTC) loss for the target languages: English, Arabic, and Mandarin.

For the current implementation, only the simple LID (language one-hot embedding) is proposed

Installation

1.**Clone the repository:**

```bash
   git clone git@github.com:Yaselley/multi-MDD-LID.git
   cd multi-MDD-LID
```

2.**Install Requirements:**

```
pip install -r requirements.txt
```

3.**Training The model:**
```
python3 main_train.py --config=config.json
```

4.**Testing The model**
```
python3 main_test.py --model_checkpoint=/path/to/model_checkpoint --test_csv=/path/to/test.csv
```
Replace /path/to/model_checkpoint and /path/to/test.csv with the appropriate paths.

5.**Change Experiments Set-up**
For easy interaction, config.json has many variables that control the model setup.

