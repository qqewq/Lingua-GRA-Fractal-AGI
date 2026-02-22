"""
fractal_utils.py – фрактальные метрики и регуляризатор для Lingua GRA

🇷🇺
В этом модуле:
- вычисление корреляционной размерности D2 (алгоритм типа Grassberger–Procaccia)
  для облака эмбеддингов;
- фрактальный регуляризатор, штрафующий отклонение D2 от целевого значения.

🇬🇧
This module provides:
- estimation of the correlation dimension D2 (Grassberger–Procaccia-style)
  for a cloud of embeddings;
- a fractal regularizer penalizing deviation of D2 from a target value.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """
    🇷🇺 Попарные евклидовы расстояния для набора точек x: [N, d].

    🇬🇧 Pairwise Euclidean distances for points x: [N, d].
    """
    # x: [N, d]
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, d]
    dist = torch.linalg.norm(diff, dim=-1)  # [N, N]
    return dist


def correlation_integral(
    dists: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """
    🇷🇺
    Вычислить корреляционный интеграл C(r) для набора радиусов.

    C(r) = 2 / (N (N-1)) * sum_{i<j} I(||x_i - x_j|| < r)

    🇬🇧
    Compute the correlation integral C(r) for a set of radii.

    C(r) = 2 / (N (N-1)) * sum_{i<j} I(||x_i - x_j|| < r)
    """
    N = dists.shape[0]
    # маскируем диагональ, чтобы не считать пары (i, i)
    tri_mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    d_ij = dists[tri_mask]  # [N*(N-1)/2]

    C_vals = []
    for r in radii:
        C_r = (d_ij < r).float().mean()
        C_vals.append(C_r + 1e-12)  # защита от log(0)
    return torch.stack(C_vals, dim=0)


def correlation_dimension(
    points: torch.Tensor,
    n_bins: int = 12,
    min_frac: float = 0.05,
    max_frac: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    🇷🇺
    Оценка корреляционной размерности D2 для облака точек в R^d.

    Алгоритм:
    1) считаем попарные расстояния;
    2) берём диапазон r от min_frac*max_dist до max_frac*max_dist;
    3) считаем C(r) и строим log C(r) vs log r;
    4) проводим линейную регрессию, наклон ≈ D2.

    Параметры:
    - n_bins: число радиусов;
    - min_frac, max_frac: доля от максимального расстояния,
      задающая рабочий диапазон масштабов.

    Возвращает:
    - D2: скалярный тензор (оценка корреляционной размерности);
    - aux: словарь с радиусами и логами для отладки/визуализации.

    🇬🇧
    Estimate the correlation dimension D2 for a point cloud in R^d.

    Steps:
    1) compute pairwise distances;
    2) take radii in [min_frac*max_dist, max_frac*max_dist];
    3) compute C(r) and log C(r) vs log r;
    4) linear regression, slope ≈ D2.

    Returns:
    - D2: scalar tensor (estimated correlation dimension);
    - aux: dict with radii and logs for debugging/visualization.
    """
    device = points.device
    dists = pairwise_distances(points)  # [N, N]

    # игнорируем нули на диагонали
    max_dist = dists.max()
    r_min = max_dist * min_frac
    r_max = max_dist * max_frac

    # логарифмическая сетка по r
    radii = torch.logspace(
        torch.log10(r_min + 1e-8),
        torch.log10(r_max + 1e-8),
        steps=n_bins,
        device=device,
    )

    C = correlation_integral(dists, radii)  # [n_bins]
    log_r = torch.log(radii)
    log_C = torch.log(C)

    # простая линейная регрессия: log_C = D2 * log_r + b
    # D2 = cov(log_r, log_C) / var(log_r)
    log_r_mean = log_r.mean()
    log_C_mean = log_C.mean()
    cov = ((log_r - log_r_mean) * (log_C - log_C_mean)).mean()
    var = ((log_r - log_r_mean) ** 2).mean()
    D2 = cov / (var + 1e-12)

    aux = {
        "radii": radii.detach(),
        "log_r": log_r.detach(),
        "log_C": log_C.detach(),
    }
    return D2, aux


def fractal_regularizer(
    embeddings: torch.Tensor,
    target_dim: float,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    🇷ГУ
    Фрактальный регуляризатор для Lingua GRA.

    Параметры:
    - embeddings: [N, d] – облако эмбеддингов (например, семантический уровень);
    - target_dim: целевая D2 (из экспериментов или внешних корпусов);
    - weight: вес регуляризатора.

    Возвращает:
    - loss: weight * (D2 - target_dim)^2,
    - D2: оценённая корреляционная размерность.

    🇬🇧
    Fractal regularizer for Lingua GRA.

    Parameters:
    - embeddings: [N, d] – embedding cloud (e.g. semantic level);
    - target_dim: target D2 (from experiments or external corpora);
    - weight: regularizer weight.

    Returns:
    - loss: weight * (D2 - target_dim)^2,
    - D2: estimated correlation dimension.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be of shape [N, d]")

    D2, _ = correlation_dimension(embeddings)
    loss = weight * (D2 - embeddings.new_tensor(target_dim)) ** 2
    return loss, D2
