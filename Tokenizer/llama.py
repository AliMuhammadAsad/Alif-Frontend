# Making and training a sentence piece based tokenzier
import sentencepiece as spm
import os


from .base import Tokenizer

import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # original bechara gpt ka

GPT4_SPLIT_PATTERN_URDU = r"""'(?i:[sdmt]|ll|ve|re)|[\p{N}]{1,3}|[\p{P}\p{S}]+|\p{Arabic}+|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]""" # modifed for urdu

class LlamaTokenizer(Tokenizer):
    def __init__(self, type):
        self.spm = spm.SentencePieceProcessor()
        self.vocab = {}
        if type == "llama":
            self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        elif type == "llama_newsplit":
            self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN_URDU)

    def train(self, text_file, vocab_size, model_prefix, verbose = False):
        options = dict(
        # input spec
        input=text_file,
        input_format="text",
        # output spec
        model_prefix=model_prefix, # output filename prefix
        # algorithm spec
        # BPE alg
        model_type="bpe",
        vocab_size=vocab_size,
        # normalization
        normalization_rule_name="identity", # ew, turn off normalization
        remove_extra_whitespaces=False,
        input_sentence_size=200000000, # max number of training sentences
        max_sentence_length=4192, # max number of bytes per sentence
        seed_sentencepiece_size=1000000,
        shuffle_input_sentence=True,
        # rare word treatment
        character_coverage=0.99995,
        byte_fallback=True,
        # merge rules
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0, # the UNK token MUST exist
        bos_id=1, # the others are optional, set to -1 to turn off
        eos_id=2,
        pad_id=-1,
        # systems
        num_threads=os.cpu_count(), # use ~all system resources
        )
        spm.SentencePieceTrainer.train(**options)

    def encode(self, text, allowed_special = "none_raise"):

        return self.spm.encode(text)

    def decode(self, ids):
        return self.spm.decode(ids)

    def _encode_chunk(self, text_bytes):
        '''
        Encodes a chunk of bytes directly using sentence piece -> decode the bytes to utf-8 string and then encode it since SPM works on strings
        '''
        text = text_bytes.decode("utf-8", errors="replace")
        return self.spm.encode(text)

    def encode_ordinary(self, text):
        '''
        Encodes text using SPM but skips special tokens
        '''
        encoded_ids = self.spm.encode(text, out_type=int)
        return encoded_ids
        # return [id for id in encoded_ids if id not in self.vocab.values()]
        # return [id for id in encoded_ids if id not in self.special_tokens]

    def register_special_tokens(self, special_tokens):
        try:
            self.spm.encode("test")
        except:
            raise ValueError("Special tokens can only be registered after training the tokenizer")
        
        for token, idx in special_tokens.items():
            self.vocab[token] = idx

    def load(self, model_file):
        assert model_file.endswith(".model")
        self.spm.load(model_file)
