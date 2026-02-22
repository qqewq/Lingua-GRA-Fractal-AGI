"""
meta_evolution.py – мета-эволюция Lingua GRA

🇷🇺
Этот модуль описывает простую схему мета-эволюции:
- мета-контроллер предлагает изменения гиперпараметров (Lambda_l, gamma_l, target_dim);
- мета-оценщик оценивает изменение целевого функционала (например, J_fractal на валидации);
- принимаем изменения, если они улучшают метрику.

Это ближе к простому стохастическому search/мета-RL для гиперпараметров,[web:182][web:186]
но адаптированному под фрактальный профиль и GRA.

🇬🇧
This module implements a simple meta-evolution scheme:
- a meta-controller proposes changes to hyperparameters (Lambda_l, gamma_l, target_dim);
- a meta-evaluator measures change in the target functional (e.g., J_fractal on validation);
- changes are accepted if they improve the metric.

This is similar to basic stochastic search / meta-RL for hyperparameters,[web:182][web:186]
adapted to the fractal profile and GRA.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Callable, Tuple

import torch

from .language_levels import Level
from .fractal_utils import correlation_dimension


@dataclass
class MetaState:
    """
    🇷ГУ Хранит текущее состояние гиперпараметров Lingua GRA.

    🇬🇧 Stores current Lingua GRA hyperparameter state.
    """

    lambda_weights: Dict[Level, float]
    gamma_fract: Dict[Level, float]
    target_fractal_dim: Dict[Level, float]


class MetaController:
    """
    🇷ГУ Мета-контроллер, предлагающий изменения гиперпараметров.

    В простейшем варианте:
    - стохастические шаги (гауссовский шум в лог-пространстве),
    - отдельные коэффициенты для разных уровней.

    🇬🇧 Meta-controller proposing hyperparameter changes.

    In the simplest variant:
    - stochastic steps (Gaussian noise in log-space),
    - separate coefficients per level.
    """

    def __init__(
        self,
        step_scale_lambda: float = 0.3,
        step_scale_gamma: float = 0.3,
        step_scale_target_dim: float = 0.1,
        seed: int | None = None,
    ):
        self.step_scale_lambda = step_scale_lambda
        self.step_scale_gamma = step_scale_gamma
        self.step_scale_target_dim = step_scale_target_dim
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def _log_step(self, value: float, scale: float) -> float:
        log_v = math.log(max(value, 1e-8))
        noise = torch.normal(0.0, scale, generator=self.rng).item()
        return float(math.exp(log_v + noise))

    def propose(self, state: MetaState) -> MetaState:
        """
        🇷ГУ Сгенерировать новое состояние гиперпараметров.

        🇬🇧 Propose a new hyperparameter state.
        """
        new_state = copy.deepcopy(state)

        # Обновляем веса Lambda_l
        for lvl, v in state.lambda_weights.items():
            new_state.lambda_weights[lvl] = self._log_step(v, self.step_scale_lambda)

        # Обновляем gamma_l (веса фрактального регуляризатора)
        for lvl, v in state.gamma_fract.items():
            new_state.gamma_fract[lvl] = self._log_step(v, self.step_scale_gamma)

        # Обновляем целевые фрактальные размерности (умеренные аддитивные шаги)
        for lvl, v in state.target_fractal_dim.items():
            noise = torch.normal(0.0, self.step_scale_target_dim, generator=self.rng).item()
            new_state.target_fractal_dim[lvl] = float(v + noise)

        return new_state


class MetaEvaluator:
    """
    🇷ГУ Мета-оценщик, вычисляющий метрику для мета-эволюции.

    В качестве метрики можно взять:
    - J_fractal (если реализован),
    - или более простой surrogate: насколько D2-распределение близко к целевым значениям.

    🇬🇧 Meta-evaluator computing a metric for meta-evolution.

    The metric can be:
    - J_fractal (if implemented),
    - or a simpler surrogate: how close the D2 distribution is to target values.
    """

    def __init__(
        self,
        level_to_embeddings_fn: Callable[[Level], torch.Tensor],
    ):
        """
        🇷ГУ
        level_to_embeddings_fn(level) должен возвращать облако эмбеддингов уровня
        на валидационном наборе (torch.Tensor [N, d]).

        🇬🇧
        level_to_embeddings_fn(level) must return a validation embedding cloud
        for the given level (torch.Tensor [N, d]).
        """
        self.level_to_embeddings_fn = level_to_embeddings_fn

    def evaluate_fractal_alignment(self, state: MetaState) -> float:
        """
        🇷ГУ
        Оценка "фрактального выравнивания":
        - для каждого уровня, где задан target_fractal_dim,
        - оцениваем D2 на валидационных эмбеддингах,
        - считаем усреднённый MSE между D2 и target.

        Метрический score ниже → лучше.

        🇬🇧
        Evaluate “fractal alignment”:
        - for each level with target_fractal_dim,
        - estimate D2 on validation embeddings,
        - compute average MSE between D2 and target.

        Lower score → better.
        """
        errors = []
        for lvl, target in state.target_fractal_dim.items():
            emb = self.level_to_embeddings_fn(lvl)
            if emb is None or emb.numel() == 0:
                continue
            D2, _ = correlation_dimension(emb)
            diff = float((D2 - emb.new_tensor(target)).abs().item())
            errors.append(diff ** 2)

        if not errors:
            return float("inf")
        return sum(errors) / len(errors)


def meta_evolution_step(
    controller: MetaController,
    evaluator: MetaEvaluator,
    current_state: MetaState,
    current_score: float,
) -> Tuple[MetaState, float, bool]:
    """
    🇷ГУ
    Один шаг мета-эволюции:
    - мета-контроллер предлагает новое состояние;
    - мета-оценщик считает новую метрику;
    - если новая метрика лучше (меньше), принимаем.

    Возвращает:
    - new_state: новое (возможно принятое) состояние;
    - new_score: метрика для new_state;
    - accepted: был ли шаг принят.

    🇬🇧
    One meta-evolution step:
    - controller proposes a new state;
    - evaluator computes a new metric;
    - accept if new metric is better (lower).

    Returns:
    - new_state: new (possibly accepted) state;
    - new_score: metric for new_state;
    - accepted: whether the step was accepted.
    """
    candidate_state = controller.propose(current_state)
    candidate_score = evaluator.evaluate_fractal_alignment(candidate_state)

    if candidate_score < current_score:
        return candidate_state, candidate_score, True
    else:
        return current_state, current_score, False
