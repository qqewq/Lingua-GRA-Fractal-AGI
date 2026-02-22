"""
training.py – обучение Lingua GRA с GRA-пеной и фрактальным регуляризатором

🇷🇺
Здесь описан базовый тренировочный цикл:
- объединение потерь разных уровней (символьный, семантический, прагматический);
- добавление GRA-пены (foam) от проекторов;
- добавление фрактального регуляризатора по D2 (correlation dimension).

🇬🇧
This module implements a basic training loop:
- combining losses from different levels (symbolic, semantic, pragmatic);
- adding GRA foam from projectors;
- adding a fractal regularizer based on D2 (correlation dimension).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .gra_core import GRAFunctional
from .fractal_utils import fractal_regularizer
from .language_levels import BaseLevel, Level
from .neural_encoders import reconstruction_loss_logits, PragmaticPolicy


class LinguaGRAModel(nn.Module):
    """
    🇷ГУ Высокоуровневая модель, объединяющая уровни Lingua GRA.

    Содержит:
    - словарь уровней (BaseLevel),
    - прагматический policy-модуль (если нужен),
    - GRAFunctional для пены + экстра-термов.

    🇬🇧 High-level model that glues together Lingua GRA levels.

    Holds:
    - dict of levels (BaseLevel),
    - pragmatic policy module (optionally),
    - GRAFunctional for foam + extra terms.
    """

    def __init__(
        self,
        levels: Dict[Level, BaseLevel],
        policy: PragmaticPolicy | None,
        gra: GRAFunctional,
    ):
        super().__init__()
        self.levels = nn.ModuleDict({str(l): lvl for l, lvl in levels.items()})
        self.policy = policy
        self.gra = gra

    def forward_token_level(self, token_ids: torch.Tensor):
        """
        🇷ГУ Прямой проход через символьный уровень.

        🇬🇧 Forward pass through the symbolic level.
        """
        lvl = self.levels[str(Level.SYMBOLIC)]
        h, h_proj, logits = lvl(token_ids)
        return h, h_proj, logits

    def forward_semantic_level(self, sent_emb: torch.Tensor):
        """
        🇷ГУ Прямой проход через семантический уровень.

        🇬🇧 Forward pass through the semantic level.
        """
        lvl = self.levels[str(Level.SEMANTIC)]
        h, h_proj, _ = lvl(sent_emb)
        return h, h_proj

    def act_pragmatic(self, obs: torch.Tensor, msg_emb: torch.Tensor) -> torch.Tensor:
        """
        🇷ГУ Вычислить logits действий на прагматическом уровне.

        🇬🇧 Compute action logits at the pragmatic level.
        """
        if self.policy is None:
            raise RuntimeError("Pragmatic policy is not set.")
        return self.policy(obs, msg_emb)


# ---------------------------------------------------------------------------
# Пример одного шага обучения
# ---------------------------------------------------------------------------


def train_step_supervised_token_semantic(
    model: LinguaGRAModel,
    optimizer: optim.Optimizer,
    batch_tokens: torch.Tensor,
    batch_target_tokens: torch.Tensor,
    semantic_inputs: torch.Tensor,
    semantic_target_dim: float | None = None,
    gamma_fract_semantic: float = 0.0,
) -> Dict[str, float]:
    """
    🇷ГУ
    Одиночный шаг обучения для упрощённого сценария:
    - символьный уровень: реконструкция токенов + GRA-пена;
    - семантический уровень: регуляризация GRA + (опционально) фрактальный регуляризатор.

    Параметры:
    - batch_tokens: [B, T] – входные токены;
    - batch_target_tokens: [B] или [B, T] – цели реконструкции;
    - semantic_inputs: [B, D_in] – входы семантического уровня (например, pooled embeddings);
    - semantic_target_dim: целевая D2; если None, фрактальный loss не добавляется;
    - gamma_fract_semantic: вес фрактального регуляризатора.

    🇬🇧
    Single training step for a simplified scenario:
    - symbolic level: token reconstruction + GRA foam;
    - semantic level: GRA regularization + optional fractal regularizer.
    """
    model.train()
    optimizer.zero_grad()

    logs: Dict[str, float] = {}

    # --- Символьный уровень -------------------------------------------------
    h_sym, h_sym_proj, logits = model.forward_token_level(batch_tokens)

    # Пример: если цель — предсказать один токен (например, next token),
    # то batch_target_tokens: [B]
    recon_loss_sym = reconstruction_loss_logits(logits, batch_target_tokens)
    logs["recon_sym"] = float(recon_loss_sym.detach().cpu())

    foam_sym = ((h_sym - h_sym_proj) ** 2).mean()
    logs["foam_sym"] = float(foam_sym.detach().cpu())

    # --- Семантический уровень ---------------------------------------------
    h_sem, h_sem_proj = model.forward_semantic_level(semantic_inputs)
    foam_sem = ((h_sem - h_sem_proj) ** 2).mean()
    logs["foam_sem"] = float(foam_sem.detach().cpu())

    fract_loss_sem = 0.0
    D2_sem_val = 0.0
    if semantic_target_dim is not None and gamma_fract_semantic > 0.0:
        fract_loss_tensor, D2_sem = fractal_regularizer(
            h_sem, target_dim=semantic_target_dim, weight=gamma_fract_semantic
        )
        fract_loss_sem = fract_loss_tensor
        D2_sem_val = float(D2_sem.detach().cpu())
        logs["fract_sem"] = float(fract_loss_tensor.detach().cpu())
        logs["D2_sem"] = D2_sem_val

    # --- GRA-функционал по уровням -----------------------------------------
    level_embeddings = {
        int(Level.SYMBOLIC): h_sym,
        int(Level.SEMANTIC): h_sem,
    }
    gra_loss, gra_logs = model.gra(level_embeddings)
    for k, v in gra_logs.items():
        logs[f"gra_{k}"] = v

    # --- Суммарный loss -----------------------------------------------------
    total_loss = recon_loss_sym + foam_sym + foam_sem + gra_loss
    if semantic_target_dim is not None and gamma_fract_semantic > 0.0:
        total_loss = total_loss + fract_loss_sem

    logs["total_loss"] = float(total_loss.detach().cpu())

    total_loss.backward()
    optimizer.step()

    return logs


# ---------------------------------------------------------------------------
# Пример цикла RL для прагматического уровня (набросок)
# ---------------------------------------------------------------------------


def policy_gradient_update(
    model: LinguaGRAModel,
    optimizer: optim.Optimizer,
    obs_batch: torch.Tensor,
    msg_emb_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    returns_batch: torch.Tensor,
) -> Dict[str, float]:
    """
    🇷ГУ
    Простейшая REINFORCE-обновление для прагматического уровня.[web:177][web:180]

    Параметры:
    - obs_batch: [B, D_obs] – наблюдения,
    - msg_emb_batch: [B, D_msg] – семантические эмбеддинги сообщений,
    - actions_batch: [B] – индексы действий,
    - returns_batch: [B] – кумулятивные вознаграждения.

    🇬🇧
    Simple REINFORCE-style update for the pragmatic level.
    """
    model.train()
    optimizer.zero_grad()

    logits = model.act_pragmatic(obs_batch, msg_emb_batch)  # [B, A]
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen_log_probs = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

    # REINFORCE: loss = -E[ return * log pi(a|s) ]
    loss = -(returns_batch * chosen_log_probs).mean()

    loss.backward()
    optimizer.step()

    return {"pg_loss": float(loss.detach().cpu())}
