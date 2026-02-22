"""
language_levels.py – уровни языка в Lingua GRA

🇷🇺
Здесь описаны основные уровни языка:
- SYMBOLIC: токены, базовый синтаксис;
- SYNTACTIC: деревья/структуры;
- SEMANTIC: эмбеддинги смыслов;
- PRAGMATIC: агент-в-среде, цели и планы;
- META: язык о языке и архитектурах.

Каждый уровень связывает:
- HilbertSpace,
- нейросетевой encoder/decoder,
- проектор P_G.

🇬🇧
This module defines the main language levels:
- SYMBOLIC: tokens, basic syntax;
- SYNTACTIC: trees/structures;
- SEMANTIC: meaning embeddings;
- PRAGMATIC: agent-in-environment, goals and plans;
- META: language about language and architectures.

Each level ties together:
- a HilbertSpace,
- neural encoder/decoder,
- a projector P_G.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .gra_core import HilbertSpace, Projector


class Level(IntEnum):
    SYMBOLIC = 0
    SYNTACTIC = 1
    SEMANTIC = 2
    PRAGMATIC = 3
    META = 4


@dataclass
class LevelConfig:
    """
    🇷🇺 Конфигурация уровня языка.

    🇬🇧 Configuration structure for a language level.
    """

    dim: int
    lambda_weight: float = 1.0
    gamma_fractal: float = 0.0
    target_fractal_dim: Optional[float] = None


class BaseLevel(nn.Module):
    """
    🇷🇺 Базовый класс уровня языка Lingua GRA.

    Содержит:
    - HilbertSpace,
    - encoder,
    - decoder (опционально),
    - projector.

    🇬🇧 Base class for a Lingua GRA language level.

    Holds:
    - HilbertSpace,
    - encoder,
    - optional decoder,
    - projector.
    """

    def __init__(
        self,
        level: Level,
        config: LevelConfig,
        encoder: nn.Module,
        projector: Projector,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.level = level
        self.config = config
        self.hilbert = HilbertSpace(dim=config.dim)
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.decoder is None:
            return None
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        🇷🇺
        Возвращает:
        - h: эмбеддинг уровня,
        - h_proj: проецированное состояние,
        - x_rec: реконструкция (если есть decoder).

        🇬🇧
        Returns:
        - h: level embedding,
        - h_proj: projected state,
        - x_rec: reconstruction (if decoder is present).
        """
        h = self.encode(x)
        h_proj = self.projector(h)
        x_rec = self.decode(h_proj)
        return h, h_proj, x_rec


# --- Конкретные уровни -----------------------------------------------------


class SymbolicLevel(BaseLevel):
    """
    🇷🇺 Символьный уровень (токены, базовый синтаксис).

    🇬🇧 Symbolic level (tokens, basic syntax).
    """

    def __init__(self, vocab_size: int, config: LevelConfig):
        encoder = SymbolicEncoder(vocab_size, config.dim)
        decoder = SymbolicDecoder(vocab_size, config.dim)
        projector = Projector(dim=config.dim)
        super().__init__(Level.SYMBOLIC, config, encoder, projector, decoder)


class SyntacticLevel(BaseLevel):
    """
    🇷🇺 Синтаксический уровень (деревья/структуры).

    🇬🇧 Syntactic level (trees/structures).
    """

    def __init__(self, config: LevelConfig):
        encoder = SyntacticEncoder(config.dim)
        projector = Projector(dim=config.dim)
        # decoder не обязателен, но можно добавить реконструкцию дерева
        super().__init__(Level.SYNTACTIC, config, encoder, projector, decoder=None)


class SemanticLevel(BaseLevel):
    """
    🇷🇺 Семантический уровень (эмбеддинги предложений/документов).

    🇬🇧 Semantic level (embeddings of sentences/documents).
    """

    def __init__(self, config: LevelConfig):
        encoder = SemanticEncoder(config.dim)
        projector = Projector(dim=config.dim)
        super().__init__(Level.SEMANTIC, config, encoder, projector, decoder=None)


class PragmaticLevel(BaseLevel):
    """
    🇷🇺 Прагматический уровень (агент-в-среде, планы).

    🇬🇧 Pragmatic level (agent-in-environment, plans).
    """

    def __init__(self, obs_dim: int, config: LevelConfig):
        encoder = PragmaticEncoder(obs_dim, config.dim)
        projector = Projector(dim=config.dim)
        super().__init__(Level.PRAGMATIC, config, encoder, projector, decoder=None)


class MetaLevel(BaseLevel):
    """
    🇷🇺 Мета-уровень (описание правил, архитектур).

    🇬🇧 Meta level (description of rules, architectures).
    """

    def __init__(self, config: LevelConfig):
        encoder = MetaEncoder(config.dim)
        projector = Projector(dim=config.dim)
        super().__init__(Level.META, config, encoder, projector, decoder=None)


# --- Простейшие заглушки энкодеров/декодеров -------------------------------


class SymbolicEncoder(nn.Module):
    """
    🇷🇺 Простой encoder для токенов: embedding + Transformer-encoder.

    🇬🇧 Simple encoder for tokens: embedding + Transformer encoder.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4),
            num_layers=2,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        x = self.embedding(token_ids)  # [B, T, D]
        x = x.transpose(0, 1)         # [T, B, D]
        h = self.encoder(x)           # [T, B, D]
        return h.mean(dim=0)          # [B, D] – усреднение по времени


class SymbolicDecoder(nn.Module):
    """
    🇷🇺 Простой decoder: проекция из эмбеддинга в распределение по словарю.

    🇬🇧 Simple decoder: projection from embedding to vocabulary distribution.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, D]
        logits = self.linear(h)  # [B, V]
        return logits


class SyntacticEncoder(nn.Module):
    """
    🇷🇺 Заглушка синтаксического encoder’а.

    В реальной реализации сюда может прийти парсер или graph NN,
    кодирующий дерево разбора в вектор.

    🇬🇧 Placeholder for syntactic encoder.

    In a real implementation this could be a parser or graph NN
    encoding a parse tree into a vector.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticEncoder(nn.Module):
    """
    🇷🇺 Семантический encoder: небольшой Transformer/MLP как заглушка.

    В реальном проекте можно заменить на LLM-энкодер из transformers.

    🇬🇧 Semantic encoder: small Transformer/MLP as a placeholder.

    In a real project, this can be replaced by a LLM encoder.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PragmaticEncoder(nn.Module):
    """
    🇷ГУ Прагматический encoder: из наблюдения среды в состояние уровня.

    🇬🇧 Pragmatic encoder: from environment observation to level state.
    """

    def __init__(self, obs_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MetaEncoder(nn.Module):
    """
    🇷ГУ Мета-encoder: кодирует описания правил, конфигураций, архитектур.

    🇬🇧 Meta encoder: encodes descriptions of rules, configurations, architectures.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Вспомогательная сборка уровней ----------------------------------------


def build_default_levels(
    vocab_size: int,
    obs_dim: int,
) -> Dict[Level, BaseLevel]:
    """
    🇷ГУ Собрать дефолтный набор уровней для экспериментов.

    🇬🇧 Build a default set of levels for experiments.
    """
    levels: Dict[Level, BaseLevel] = {}

    levels[Level.SYMBOLIC] = SymbolicLevel(
        vocab_size=vocab_size,
        config=LevelConfig(dim=128, lambda_weight=1.0),
    )
    levels[Level.SYNTACTIC] = SyntacticLevel(
        config=LevelConfig(dim=128, lambda_weight=0.5),
    )
    levels[Level.SEMANTIC] = SemanticLevel(
        config=LevelConfig(
            dim=256,
            lambda_weight=1.0,
            gamma_fractal=0.1,
            target_fractal_dim=None,  # можно установить после эксперимента
        ),
    )
    levels[Level.PRAGMATIC] = PragmaticLevel(
        obs_dim=obs_dim,
        config=LevelConfig(dim=128, lambda_weight=0.5),
    )
    levels[Level.META] = MetaLevel(
        config=LevelConfig(dim=128, lambda_weight=0.1),
    )

    return levels
