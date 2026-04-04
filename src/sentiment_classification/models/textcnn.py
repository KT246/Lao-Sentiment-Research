from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class TextCNNConfig(PretrainedConfig):
    model_type = "textcnn"

    def __init__(
        self,
        vocab_size: int = 250002,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: Iterable[int] = (3, 4, 5),
        dropout: float = 0.3,
        pad_token_id: int = 0,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, num_labels=num_labels, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = list(kernel_sizes)
        self.dropout = dropout


class TextCNNForSequenceClassification(PreTrainedModel):
    config_class = TextCNNConfig
    base_model_prefix = "textcnn"

    def __init__(self, config: TextCNNConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.embedding_dim,
                    out_channels=config.num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in config.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.num_filters * len(config.kernel_sizes), config.num_labels)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            return

        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _ensure_minimum_sequence_length(self, embeddings: torch.Tensor) -> torch.Tensor:
        required_length = max(self.config.kernel_sizes)
        current_length = embeddings.size(-1)
        if current_length >= required_length:
            return embeddings

        pad_amount = required_length - current_length
        return F.pad(embeddings, (0, pad_amount))

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> List[torch.Tensor]:
        embeddings = self.embedding(input_ids)
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).to(embeddings.dtype)

        embeddings = embeddings.transpose(1, 2)
        embeddings = self._ensure_minimum_sequence_length(embeddings)

        pooled_outputs = []
        for conv in self.convs:
            features = torch.relu(conv(embeddings))
            pooled = torch.max(features, dim=-1).values
            pooled_outputs.append(pooled)
        return pooled_outputs

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("TextCNN requires `input_ids`.")

        pooled_outputs = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        concatenated = torch.cat(pooled_outputs, dim=1)
        logits = self.classifier(self.dropout(concatenated))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
