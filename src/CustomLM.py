# Look at my pretty imports
import os
import torch

from typing import *

from allennlp.modules.seq2seq_encoders import *
from allennlp.training.metrics import Perplexity
from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import *
from allennlp.data.tokenizers import *
from allennlp.data.data_loaders import *
from allennlp.modules.text_field_embedders import *
from allennlp.modules.token_embedders import *

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.data.token_indexers import *
from allennlp.data.tokenizers import *
from allennlp.nn import util

from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

from allennlp.predictors import *

import glob
import os

import numpy as np



START_TOKEN = "@@START@@"

@DatasetReader.register("my-data")
class TextReader(DatasetReader):
    def __init__(self, data_root: str, tokenizer_in: Tokenizer = None, tokenizer_out: Tokenizer = None, 
                 token_indexer_in = None, token_indexer_out = None, 
                 max_tokens: int = None, truncate_last_in: bool = True, 
                 include_labels: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.tokenizer_in = tokenizer_in or CharacterTokenizer(start_tokens=[START_TOKEN])
        self.tokenizer_out = tokenizer_out or CharacterTokenizer()
        self.token_indexers_in = {"tokens": token_indexer_in or SingleIdTokenIndexer('tokens')}
        self.token_indexers_out = {"labels": token_indexer_in or SingleIdTokenIndexer('labels')}
        self.max_tokens = max_tokens
        self.truncate_last_in = truncate_last_in
        self.include_labels = include_labels
    
    def text_to_instance(self, text: str) -> Instance:  # type: ignore
        tokens_in = self.tokenizer_in.tokenize(text)
        if self.truncate_last_in:
            tokens_in = tokens_in[:-1]
        tokens_out = self.tokenizer_out.tokenize(text)
        if self.max_tokens:
            tokens_in = tokens_in[: self.max_tokens]
            tokens_out = tokens_out[: self.max_tokens]
        text_field = TextField(tokens_in, self.token_indexers_in)
        fields: Dict[str, Field] = {"text": text_field}
        if self.include_labels:
            fields["labels"] = TextField(tokens_out, self.token_indexers_out)
        return Instance(fields)
    
    def _read(self, file_root: str) -> Iterable[Instance]:
        filenames = glob.glob(os.path.join(self.data_root, file_root, '**/*'), recursive=True)
        for filename in filenames:
            with open(filename) as file:
                for line in file:
                    line = line.strip()
                    if not len(line):
                        continue
                    yield self.text_to_instance(line)

@Model.register('custom_lm')
class CustomLM(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2SeqEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.vocab_size = vocab.get_vocab_size('labels')
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), self.vocab_size)
        self.perplexity = Perplexity()
    def forward(self, text: TextFieldTensors, labels: TextFieldTensors = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, num_tokens, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_tokens, vocab_size)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_tokens, vocab_size)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = {}
        output['probs'] = probs
        if labels is not None:
            labels = labels['labels']['tokens']
            labels[~mask] = -100
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
            output["loss"] = loss
            self.perplexity(loss)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"perplexity": self.perplexity.get_metric(reset)}



@Predictor.register('my_predictor')
class MyPredictor(Predictor):
    def predict(self, sentence):
        return self.predict_json({'sentence': sentence})
    
    def _json_to_instance(self, json_dict):
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)
