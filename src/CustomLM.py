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

@DatasetReader.register("my_data")
class TextReader(DatasetReader):
    def __init__(self, data_root: str, tokenizer_in: Tokenizer = None, tokenizer_out: Tokenizer = None, 
                 token_indexer_in = None, token_indexer_out = None, 
                 max_tokens: int = None, truncate_last_in: bool = True, 
                 include_labels: bool = True, black_list: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.tokenizer_in = tokenizer_in or CharacterTokenizer(start_tokens=[START_TOKEN])
        self.tokenizer_out = tokenizer_out or CharacterTokenizer()
        self.token_indexers_in = {"tokens": token_indexer_in or SingleIdTokenIndexer('tokens')}
        self.token_indexers_out = {"labels": token_indexer_in or SingleIdTokenIndexer('labels')}
        self.max_tokens = max_tokens
        self.truncate_last_in = truncate_last_in
        self.include_labels = include_labels
        self.black_list = black_list
    
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
            if self.black_list is not None and any((os.path.basename(filename).startswith(black) for black in self.black_list)):
                continue
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
        self.only_last = False

        self.bad_for_pred = [idx for token, idx in vocab.get_token_to_index_vocabulary('labels').items() if len(token) > 1]
        self.accuracy = MyCategoricalAccuracy(top_k=3)

    def forward(self, text: TextFieldTensors, labels: TextFieldTensors = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, num_tokens, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        if self.only_last:
            #encoded_text = torch.gather(encoded_text, 1, torch.sum(mask, axis=-1) - 1)
            last_index = torch.sum(mask, axis=-1) - 1
            encoded_text = encoded_text[torch.arange(len(encoded_text)), last_index, :]
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

            if not self.training:
                probs[:, :, self.bad_for_pred] = -1
                self.accuracy(probs, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"perplexity": self.perplexity.get_metric(reset), "accuracy": self.accuracy.get_metric(reset)}



@Predictor.register('my_predictor')
class MyPredictor(Predictor):
    def predict(self, sentence):
        return self.predict_json({'sentence': sentence})
    
    def predict_batch(self, sentences):
        return self.predict_batch_json([{'sentence': sentence} for sentence in sentences])
    
    def _json_to_instance(self, json_dict):
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)
    
    def predict_batch_instance(self, instances):
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instances(instances)
        return outputs



from typing import Optional

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


class MyCategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """

    supports_distributed = True

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise ConfigurationError(
                "Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)"
            )
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = torch.topk(predictions, min(self._top_k, predictions.shape[-1]))[1]
                #_, sorted_indices = predictions.sort(dim=-1, descending=True)
                #top_k = sorted_indices[..., : min(self._top_k, predictions.shape[-1])]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel(), device=gold_labels.device).long(), gold_labels
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1)
            _total_count = mask.sum()
        else:
            _total_count = torch.tensor(gold_labels.numel())
        _correct_count = correct.sum()

        if is_distributed():
            device = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")
            _correct_count = _correct_count.to(device)
            _total_count = _total_count.to(device)
            dist.all_reduce(_correct_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(_total_count, op=dist.ReduceOp.SUM)

        self.correct_count += _correct_count.item()
        self.total_count += _total_count.item()

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0

        if reset:
            self.reset()

        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0