"""
test_neural.py – юнит-тесты для neural_encoders.py и высокоуровневых моделей

🇷ГУ
Проверяем:
- формы выходов TokenEncoder/TokenDecoder;
- SentenceEncoder и PragmaticPolicy;
- высокоуровневые TokenLevelModel и SemanticLevelModel.

🇬🇧
Unit tests for neural_encoders.py and high-level models:

- output shapes of TokenEncoder/TokenDecoder;
- SentenceEncoder and PragmaticPolicy;
- TokenLevelModel and SemanticLevelModel wrappers.
"""

from __future__ import annotations

import torch

from lingua_gra.neural_encoders import (
    TokenEncoder,
    TokenDecoder,
    SentenceEncoder,
    PragmaticPolicy,
    TokenLevelModel,
    SemanticLevelModel,
)
from lingua_gra.gra_core import Projector


def test_token_encoder_decoder_shapes():
    vocab_size = 100
    d_model = 32
    encoder = TokenEncoder(vocab_size=vocab_size, d_model=d_model, max_len=32)
    decoder = TokenDecoder(vocab_size=vocab_size, d_model=d_model)

    token_ids = torch.randint(0, vocab_size, (4, 10))  # [B, T]
    h = encoder(token_ids)
    logits = decoder(h)

    assert h.shape == (4, d_model)
    assert logits.shape == (4, vocab_size)


def test_sentence_encoder_shape():
    in_dim = 64
    d_model = 32
    encoder = SentenceEncoder(in_dim=in_dim, d_model=d_model)

    x = torch.randn(5, in_dim)
    h = encoder(x)

    assert h.shape == (5, d_model)


def test_pragmatic_policy_shapes():
    obs_dim = 7
    msg_dim = 16
    hidden_dim = 32
    n_actions = 5

    policy = PragmaticPolicy(
        obs_dim=obs_dim,
        msg_dim=msg_dim,
        hidden_dim=hidden_dim,
        n_actions=n_actions,
    )

    obs = torch.randn(6, obs_dim)
    msg = torch.randn(6, msg_dim)
    logits = policy(obs, msg)

    assert logits.shape == (6, n_actions)


def test_token_level_model_wrapper():
    vocab_size = 50
    d_model = 16

    encoder = TokenEncoder(vocab_size=vocab_size, d_model=d_model, max_len=16)
    projector = Projector(dim=d_model)
    decoder = TokenDecoder(vocab_size=vocab_size, d_model=d_model)

    model = TokenLevelModel(encoder=encoder, projector=projector, decoder=decoder)

    token_ids = torch.randint(0, vocab_size, (3, 8))
    h, h_proj, logits = model(token_ids)

    assert h.shape == (3, d_model)
    assert h_proj.shape == (3, d_model)
    assert logits.shape == (3, vocab_size)


def test_semantic_level_model_wrapper():
    in_dim = 20
    d_model = 10

    encoder = SentenceEncoder(in_dim=in_dim, d_model=d_model)
    projector = Projector(dim=d_model)

    model = SemanticLevelModel(encoder=encoder, projector=projector, decoder=None)

    x = torch.randn(4, in_dim)
    h, h_proj, x_rec = model(x)

    assert h.shape == (4, d_model)
    assert h_proj.shape == (4, d_model)
    assert x_rec is None
