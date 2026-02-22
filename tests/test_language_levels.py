"""
test_language_levels.py – юнит-тесты для language_levels.py

🇷ГУ
Проверяем:
- базовую работу BaseLevel (encode → projector → decode);
- корректную инициализацию конкретных уровней (Symbolic, Semantic, Pragmatic, Meta);
- сборку уровней через build_default_levels.

🇬🇧
Unit tests for language_levels.py:

- basic BaseLevel flow (encode → projector → decode);
- correct initialization of concrete levels (Symbolic, Semantic, Pragmatic, Meta);
- assembling levels via build_default_levels.
"""

from __future__ import annotations

import torch

from lingua_gra.language_levels import (
    Level,
    LevelConfig,
    BaseLevel,
    SymbolicLevel,
    SyntacticLevel,
    SemanticLevel,
    PragmaticLevel,
    MetaLevel,
    build_default_levels,
)
from lingua_gra.gra_core import Projector


class DummyEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyDecoder(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def test_base_level_forward():
    dim = 8
    in_dim = 4

    encoder = DummyEncoder(in_dim, dim)
    decoder = DummyDecoder(dim, in_dim)
    projector = Projector(dim=dim)

    lvl = BaseLevel(
        level=Level.SEMANTIC,
        config=LevelConfig(dim=dim),
        encoder=encoder,
        projector=projector,
        decoder=decoder,
    )

    x = torch.randn(5, in_dim)
    h, h_proj, x_rec = lvl(x)

    assert h.shape == (5, dim)
    assert h_proj.shape == (5, dim)
    assert x_rec is not None
    assert x_rec.shape == (5, in_dim)


def test_symbolic_level_shapes():
    vocab_size = 100
    cfg = LevelConfig(dim=16)
    lvl = SymbolicLevel(vocab_size=vocab_size, config=cfg)

    token_ids = torch.randint(0, vocab_size, (4, 10))
    h, h_proj, logits = lvl(token_ids)

    assert h.shape == (4, cfg.dim)
    assert h_proj.shape == (4, cfg.dim)
    assert logits.shape == (4, vocab_size)


def test_semantic_level_shapes():
    cfg = LevelConfig(dim=32)
    lvl = SemanticLevel(config=cfg)

    x = torch.randn(6, cfg.dim)
    h, h_proj, x_rec = lvl(x)

    assert h.shape == (6, cfg.dim)
    assert h_proj.shape == (6, cfg.dim)
    assert x_rec is None


def test_pragmatic_level_shapes():
    obs_dim = 5
    cfg = LevelConfig(dim=16)
    lvl = PragmaticLevel(obs_dim=obs_dim, config=cfg)

    obs = torch.randn(7, obs_dim)
    h, h_proj, x_rec = lvl(obs)

    assert h.shape == (7, cfg.dim)
    assert h_proj.shape == (7, cfg.dim)
    assert x_rec is None


def test_meta_level_shapes():
    cfg = LevelConfig(dim=12)
    lvl = MetaLevel(config=cfg)

    x = torch.randn(3, cfg.dim)
    h, h_proj, x_rec = lvl(x)

    assert h.shape == (3, cfg.dim)
    assert h_proj.shape == (3, cfg.dim)
    assert x_rec is None


def test_build_default_levels():
    vocab_size = 5000
    obs_dim = 16
    levels = build_default_levels(vocab_size=vocab_size, obs_dim=obs_dim)

    assert Level.SYMBOLIC in levels
    assert Level.SEMANTIC in levels
    assert Level.PRAGMATIC in levels
    assert Level.META in levels

    sym = levels[Level.SYMBOLIC]
    assert sym.config.dim == 128

    sem = levels[Level.SEMANTIC]
    assert sem.config.dim == 256
    assert sem.config.gamma_fractal == 0.1
