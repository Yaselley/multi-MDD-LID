#%%
import sys
import os
import numpy as np
import torch
import logging
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.modeling_outputs import CausalLMOutput
#%%
import torch
import torch.nn as nn


# Import necessary libraries
# %%
import sys
import os
import numpy as np
import torch
import logging
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.modeling_outputs import CausalLMOutput

# %%
# Import necessary libraries (duplicate import statements removed)
import torch
import torch.nn as nn

# Define the SSLforJointClassification model
class SSLforJointClassification(nn.Module):
    
    """
    SSLforJointClassification is a PyTorch module for SSL (Self-Supervised Learning) with joint classification.
    => Attention Block for joint alignmnet text-audio
    => B-LSTM for aligned acoustic and textual information enhancement
    => LM-head encorporating LID features and joint embedding before classification

    Args:
        config: Configuration object containing model parameters.

    Attributes:
        pretrained (Wav2Vec2ForCTC): Pretrained Wav2Vec2 model for SSL.
        projection (nn.Linear): Linear layer for projection.
        config: Model configuration object.

    Example:
        ssl_model = SSLforJointClassification(config=my_config)
        output = ssl_model(input_values=my_inputs, labels=my_labels)
    """

    def __init__(self, config):
        super(SSLforJointClassification, self).__init__()

        # Load pretrained Wav2Vec2 model with specified configurations
        self.pretrained = Wav2Vec2ForCTC.from_pretrained(config.model_name_audio,
                                                            attention_dropout=0.0,
                                                            layerdrop=0.1,
                                                            feat_proj_dropout=0.01,
                                                            mask_time_prob=0.05, 
                                                            mask_time_length=10,
                                                            ctc_loss_reduction="mean",
                                                            ignore_mismatched_sizes=True,
                                                            vocab_size=config.num_labels
                                                        )

        # LSTM embeddings for phoneme labels
        rnn_hidden_size = 512
        self.embeds = nn.Embedding(83, 512)
        self.lstm_embeds = nn.LSTM(512, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.score = nn.Linear(rnn_hidden_size*2, rnn_hidden_size*2, bias=False)

        # LSTM embeddings for language ID labels
        # rnn_hidden_size_text = 24
        self.embeds_text = nn.Embedding(3, 3)
        # self.lstm_embeds_text = nn.LSTM(24, rnn_hidden_size_text, batch_first=True, bidirectional=True)

        # Final linear layer for classification
        self.fc = nn.Linear(2*rnn_hidden_size*2, config.num_labels, bias=False)
        self.lm_head = nn.Linear(2*rnn_hidden_size*2+3, config.num_labels, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.config = config

    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC
    """
    
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get hidden states from pretrained Wav2Vec2 model
        speech = input_values.get('input_values')
        hidden_states = self.pretrained.wav2vec2(
            speech,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = hidden_states[0]

        # Process phoneme labels with LSTM
        labels_pr = input_values.get('labels_ref')
        labels_pr[labels_pr < 0] = 82
        labels_pr = self.embeds(labels_pr)
        labels_pr, _ = self.lstm_embeds(labels_pr)
        key = self.score(labels_pr)

        # Attention mechanism
        attn_score = torch.bmm(hidden_states, key.transpose(1, 2))
        attn_max, _ = torch.max(attn_score, dim=-1, keepdim=True) 
        exp_score = torch.exp(attn_score - attn_max)

        attn_weights = exp_score
        weights_denom = torch.sum(attn_weights, dim=-1, keepdim=True)   
        attn_weights = attn_weights / (weights_denom + 1e-30)
        c = torch.bmm(attn_weights, labels_pr)
        
        out1 = torch.cat((hidden_states, c), -1)        
        
        # Process language ID labels with LSTM
        IDS = input_values.get('labels_LID')
        IDS = torch.tensor(IDS, device="cuda").unsqueeze(1)

        embs_LID = self.embeds_text(IDS)
        # embs_LID, _ = self.lstm_embeds_text(embs_LID)
        expanded_LID = embs_LID.expand(-1, out1.shape[1], -1)

        fused_tensor = torch.cat((out1, expanded_LID), dim=2)
        out = self.lm_head(fused_tensor)
        
        logits={}
        
        head = out
        
        ## Final Phoneme Recognition logits
        logits["PR"] = head
        
        # Loss calculation
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        return CausalLMOutput(
            logits=logits, hidden_states=hidden_states, attentions=hidden_states
        )
        
   
    # Functions to freeze different parts of the model during training
    def freeze_feature_extractor(self):
        self.pretrained.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False 

    def gradient_checkpointing_enable(self):
        self.pretrained.gradient_checkpointing_enable()