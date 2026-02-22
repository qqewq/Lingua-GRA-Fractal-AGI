"""
neural_encoders.py – нейросетевые блоки для уровней Lingua GRA

🇷🇺
Здесь находятся базовые encoder/decoder-модули для разных уровней языка:
- TokenEncoder / TokenDecoder: символьный уровень;
- SentenceEncoder: семантический уровень (предложение/документ);
- PragmaticPolicy: прагматический уровень (policy-сеть для агента);
- утилиты для сборки простых моделей.

🇬🇧
This module provides basic encoder/decoder modules for different language levels:
- TokenEncoder / TokenDecoder: symbolic level;
- SentenceEncoder: semantic level (sentence/document);
- PragmaticPolicy: pragmatic level (agent policy network);
- utilities to assemble simple models.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Символьный / токенный уровень
# ---------------------------------------------------------------------------


class TokenEncoder(nn.Module):
    """
    🇷🇺 Encoder для последовательности токенов: embedding + Transformer encoder.

    🇬🇧 Encoder for token sequences: embedding + Transformer encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, T] – индексы токенов.

        Возвращает:
        - h_cls: [B, D] – агрегированный эмбеддинг последовательности
          (по аналогии с [CLS]-токеном в Transformer-моделях).[web:162][web:166]
        """
        bsz, seq_len = token_ids.shape
        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, -1)

        x = self.embedding(token_ids) + self.pos_embedding(pos_ids)  # [B, T, D]
        h = self.encoder(x)  # [B, T, D]

        # простая агрегация: берём первый токен как "CLS"
        h_cls = h[:, 0, :]
        return h_cls


class TokenDecoder(nn.Module):
    """
    🇷🇺 Decoder: из эмбеддинга последовательности обратно в распределение по словарю.

    🇬🇧 Decoder: from sequence embedding back to vocabulary logits.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, D] → logits: [B, V]

        Можно использовать для реконструкции или генерации с помощью softmax/сэмплинга.
        """
        return self.linear(h)


# ---------------------------------------------------------------------------
# Семантический уровень (предложение/документ)
# ---------------------------------------------------------------------------


class SentenceEncoder(nn.Module):
    """
    🇷ГУ Семантический encoder для предложений/документов.

    В простейшем виде:
    - принимает уже готовые sentence-эмбеддинги (например, среднее по токенам),
      и дообрабатывает их MLP-слоем;
    или
    - можно встроить TokenEncoder внутри и агрегировать.

    🇬🇧 Semantic encoder for sentences/documents.

    In the simplest form:
    - takes pre-computed sentence embeddings (e.g., mean-pooled token embeddings),
      and refines them with an MLP;
    or
    - can include a TokenEncoder and aggregate internally.
    """

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D_in] → [B, D_model]
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Прагматический уровень: policy-сеть
# ---------------------------------------------------------------------------


class PragmaticPolicy(nn.Module):
    """
    🇷ГУ Policy-сеть для прагматического уровня.

    Вход:
    - obs: наблюдение среды (например, состояние grid-world),
    - msg: эмбеддинг сообщения от другого агента (семантический уровень).

    Выход:
    - logits действий.

    🇬🇧 Policy network for the pragmatic level.

    Input:
    - obs: environment observation (e.g., grid-world state),
    - msg: message embedding from another agent (semantic level).

    Output:
    - action logits.
    """

    def __init__(self, obs_dim: int, msg_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, D_obs]
        msg: [B, D_msg]
        returns: logits [B, n_actions]
        """
        x = torch.cat([obs, msg], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Вспомогательные высокоуровневые модели
# ---------------------------------------------------------------------------


class SemanticLevelModel(nn.Module):
    """
    🇷ГУ Упрощённая модель семантического уровня:
    - sentence encoder + projector (задаётся снаружи) + (опционально) decoder.

    🇬🇧 Simplified semantic level model:
    - sentence encoder + external projector + optional decoder.
    """

    def __init__(
        self,
        encoder: SentenceEncoder,
        projector: nn.Module,
        decoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder

    def forward(self, x: torch.Tensor):
        """
        x: [B, D_in] – sentence-level inputs (e.g., pooled token embeddings).

        Returns:
        - h: [B, D] – raw embeddings,
        - h_proj: [B, D] – projected embeddings,
        - x_rec: [B, D_out] or None – reconstructed outputs if decoder is present.
        """
        h = self.encoder(x)
        h_proj = self.projector(h)
        x_rec = self.decoder(h_proj) if self.decoder is not None else None
        return h, h_proj, x_rec


class TokenLevelModel(nn.Module):
    """
    🇷ГУ Упрощённая модель символьного уровня:
    - TokenEncoder + projector + TokenDecoder.

    🇬🇧 Simplified token-level model:
    - TokenEncoder + projector + TokenDecoder.
    """

    def __init__(
        self,
        encoder: TokenEncoder,
        projector: nn.Module,
        decoder: TokenDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder

    def forward(self, token_ids: torch.Tensor):
        """
        token_ids: [B, T]

        Returns:
        - h: [B, D],
        - h_proj: [B, D],
        - logits: [B, V]
        """
        h = self.encoder(token_ids)
        h_proj = self.projector(h)
        logits = self.decoder(h_proj)
        return h, h_proj, logits


# ---------------------------------------------------------------------------
# Пример потерь
# ---------------------------------------------------------------------------


def reconstruction_loss_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    🇷ГУ Кросс-энтропийный loss для реконструкции токенов по logits.

    🇬🇧 Cross-entropy loss for token reconstruction from logits.
    """
    # logits: [B, V], target_ids: [B]
    return F.cross_entropy(logits, target_ids)
