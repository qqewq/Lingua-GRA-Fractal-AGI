"""
simple_gra_demo.py – простой демонстрационный скрипт Lingua GRA

🇷🇺
Демонстрация:
- создаём игрушечную модель с символьным и семантическим уровнями;
- считаем GRA-пену;
- (опционально) добавляем фрактальный регуляризатор;
- запускаем несколько шагов обучения и печатаем метрики.

🇬🇧
Demo:
- build a toy model with symbolic and semantic levels;
- compute GRA foam;
- (optionally) add a fractal regularizer;
- run a few training steps and print metrics.
"""

from __future__ import annotations

import logging

import torch
import torch.optim as optim

from lingua_gra.gra_core import GRAFunctional, Projector
from lingua_gra.language_levels import (
    Level,
    LevelConfig,
    SymbolicLevel,
    SemanticLevel,
)
from lingua_gra.fractal_utils import fractal_regularizer
from lingua_gra.neural_encoders import reconstruction_loss_logits
from lingua_gra.training import LinguaGRAModel
from lingua_gra.utils import setup_logging


def build_toy_model(vocab_size: int = 1000) -> LinguaGRAModel:
    # --- Уровни -------------------------------------------------------------
    sym_cfg = LevelConfig(dim=64, lambda_weight=1.0)
    sem_cfg = LevelConfig(dim=128, lambda_weight=1.0, gamma_fractal=0.1)

    sym_level = SymbolicLevel(vocab_size=vocab_size, config=sym_cfg)
    sem_level = SemanticLevel(config=sem_cfg)

    levels = {
        Level.SYMBOLIC: sym_level,
        Level.SEMANTIC: sem_level,
    }

    # --- GRA-функционал -----------------------------------------------------
    projectors = {
        int(Level.SYMBOLIC): Projector(dim=sym_cfg.dim),
        int(Level.SEMANTIC): Projector(dim=sem_cfg.dim),
    }
    lambdas = {
        int(Level.SYMBOLIC): sym_cfg.lambda_weight,
        int(Level.SEMANTIC): sem_cfg.lambda_weight,
    }

    gra = GRAFunctional(projectors=projectors, lambdas=lambdas)

    model = LinguaGRAModel(
        levels=levels,
        policy=None,
        gra=gra,
    )
    return model


def main():
    setup_logging()
    logger = logging.getLogger("simple_gra_demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_toy_model(vocab_size=5000).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    B, T = 32, 16  # batch size, sequence length
    vocab_size = 5000
    target_D2 = 7.0  # произвольное целевое значение для демонстрации

    for step in range(1, 11):
        # Синтетические данные
        token_ids = torch.randint(0, vocab_size, (B, T), device=device)
        target_ids = token_ids[:, 0]  # пример: пытаемся предсказать первый токен

        # Семантический вход: усреднённые one-hot’ы (очень грубый пример)
        one_hot = torch.nn.functional.one_hot(token_ids, num_classes=vocab_size).float()
        sent_emb = one_hot.mean(dim=1)  # [B, V]

        # Прямой проход
        model.train()
        optimizer.zero_grad()

        sym_level = model.levels[str(Level.SYMBOLIC)]
        sem_level = model.levels[str(Level.SEMANTIC)]

        h_sym, h_sym_proj, logits = sym_level(token_ids)
        h_sem, h_sem_proj, _ = sem_level(sent_emb)

        # Потери
        recon_loss = reconstruction_loss_logits(logits, target_ids)
        foam_sym = ((h_sym - h_sym_proj) ** 2).mean()
        foam_sem = ((h_sem - h_sem_proj) ** 2).mean()

        # Фрактальный регуляризатор на семантическом уровне
        fract_loss, D2_est = fractal_regularizer(
            h_sem, target_dim=target_D2, weight=0.1
        )

        # GRA-функционал
        level_embeddings = {
            int(Level.SYMBOLIC): h_sym,
            int(Level.SEMANTIC): h_sem,
        }
        gra_loss, gra_logs = model.gra(level_embeddings)

        total_loss = recon_loss + foam_sym + foam_sem + fract_loss + gra_loss

        total_loss.backward()
        optimizer.step()

        logger.info(
            f"step={step} "
            f"loss={total_loss.item():.4f} "
            f"recon={recon_loss.item():.4f} "
            f"foam_sym={foam_sym.item():.4f} "
            f"foam_sem={foam_sem.item():.4f} "
            f"fract={fract_loss.item():.4f} "
            f"D2_sem={D2_est.item():.4f}"
        )


if __name__ == "__main__":
    main()
