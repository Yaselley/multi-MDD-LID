from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor

class TokenizerExtractor:
    def __init__(self, vocab_path):
        # Initialize TokenizerExtractor with the provided vocabulary path
        self.vocab = vocab_path
        
        # Add special tokens to Wav2Vec2CTCTokenizer 
        self.tokenizer = Wav2Vec2CTCTokenizer(
                                                self.vocab,
                                                unk_token="[UNK]",
                                                pad_token="[PAD]",
                                                word_delimiter_token="|"
                                            )

        # Ensure the Sampling Rate is 16K
        self.feature_extractor = Wav2Vec2FeatureExtractor(
                                                            feature_size=1,
                                                            sampling_rate=16000,
                                                            padding_value=0.0,
                                                            do_normalize=True,
                                                            return_attention_mask=True
                                                        )

        # Initialize Wav2Vec2Processor with feature_extractor and tokenizer
        self.processor = Wav2Vec2Processor(
                                            feature_extractor=self.feature_extractor,
                                            tokenizer=self.tokenizer
                                        )
