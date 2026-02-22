"""
gra_core.py – GRA ядро для Lingua GRA

🇷🇺
Здесь определяются абстракции:
- HilbertSpace: оболочка над векторным пространством представлений;
- Projector: аппроксимация проектора на подпространство целей G_l;
- GRAFunctional: многоуровневый функционал «пены» для обнулёнки.

🇬🇧
This module defines:
- HilbertSpace: a thin wrapper over the representation vector space;
- Projector: an approximation of the goal projector P_{G_l};
- GRAFunctional: the multi-level foam functional for annihilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class HilbertSpace:
    """
    🇷🇺 Абстракция гильбертового пространства представлений.

    На практике это просто:
    - размерность d,
    - тип носителя (в данный момент: torch.Tensor),
    - функция для вычисления нормы и скалярного произведения.

    🇬🇧 Abstraction of a Hilbert space of representations.

    In practice this is:
    - dimension d,
    - carrier type (currently: torch.Tensor),
    - functions for norm and inner product.
    """

    dim: int

    def inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Скалярное произведение / Inner product."""
        return (x * y).sum(dim=-1)

    def norm2(self, x: torch.Tensor) -> torch.Tensor:
        """Квадрат нормы / Squared norm."""
        return self.inner(x, x)


class Projector(nn.Module):
    """
    🇷🇺 Нейросетевой проектор, аппроксимирующий оператор P_G.

    Идея:
    - forward(x) ≈ P_G x
    - foam(x) = ||(1 - P_G) x||^2

    🇬🇧 Neural projector approximating the operator P_G.

    Idea:
    - forward(x) ≈ P_G x
    - foam(x) = ||(1 - P_G) x||^2
    """

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        h = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(),
            nn.Linear(h, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def foam(self, x: torch.Tensor) -> torch.Tensor:
        """
        🇷🇺 Пену считаем как средний квадрат отклонения (x - P_G x).
        🇬🇧 Foam is the mean squared deviation (x - P_G x).
        """
        x_proj = self.forward(x)
        return ((x - x_proj) ** 2).mean()


class GRAFunctional(nn.Module):
    """
    🇷🇺 GRA-функционал для набора уровней.

    Хранит:
    - словарь проекторов по уровням,
    - веса Lambda_l,
    - при необходимости дополнительные регуляризаторы (например, фрактальный).

    foam_terms[l](x_l) ожидается как scalar-тензор (loss).

    🇬🇧 GRA functional over multiple levels.

    Stores:
    - a dict of projectors per level,
    - weights Lambda_l,
    - optional extra regularizers (e.g. fractal term).

    foam_terms[l](x_l) is expected to be a scalar tensor (loss).
    """

    def __init__(
        self,
        projectors: Dict[int, Projector],
        lambdas: Dict[int, float],
        extra_terms: Optional[
            Dict[int, List[Callable[[torch.Tensor], torch.Tensor]]]
        ] = None,
    ):
        super().__init__()
        self.projectors = nn.ModuleDict(
            {str(l): p for l, p in projectors.items()}
        )
        self.lambdas = lambdas
        self.extra_terms = extra_terms or {}

    def forward(
        self,
        level_embeddings: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        🇷🇺
        level_embeddings: словарь {l: x_l}, где x_l — батч представлений уровня l.

        Возвращает:
        - суммарный loss J_GRA,
        - лог-метрики по уровням.

        🇬🇧
        level_embeddings: dict {l: x_l}, where x_l is the batch of representations at level l.

        Returns:
        - total loss J_GRA,
        - per-level metrics for logging.
        """
        total_loss = 0.0
        logs: Dict[str, float] = {}

        for l, x_l in level_embeddings.items():
            key = str(l)
            if key not in self.projectors:
                continue

            proj = self.projectors[key]
            foam_l = proj.foam(x_l)
            lam = self.lambdas.get(l, 1.0)

            loss_l = lam * foam_l
            logs[f"foam_l{l}"] = float(foam_l.detach().cpu())

            # Дополнительные термины, например фрактальный регуляризатор
            if l in self.extra_terms:
                for i, term in enumerate(self.extra_terms[l]):
                    term_val = term(x_l)
                    loss_l = loss_l + term_val
                    logs[f"extra_l{l}_{i}"] = float(term_val.detach().cpu())

            total_loss = total_loss + loss_l

        return total_loss, logs
