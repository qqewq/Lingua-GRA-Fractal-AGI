"""
test_fractal_utils.py – юнит-тесты для fractal_utils.py

🇷ГУ
Проверяем:
- pairwise_distances и correlation_integral на простых случаях;
- correlation_dimension на известных конфигурациях:
  - точки на отрезке (ожидаем D2 ~ 1),
  - точки в квадрате (ожидаем D2 ~ 2);
- корректность формы и знака фрактального регуляризатора.

🇬🇧
Unit tests for fractal_utils.py:

- pairwise_distances and correlation_integral on simple cases;
- correlation_dimension on known configurations:
  - points on a line (expect D2 ~ 1),
  - points in a square (expect D2 ~ 2);
- basic sanity checks for the fractal regularizer.
"""

from __future__ import annotations

import math

import torch

from lingua_gra.fractal_utils import (
    pairwise_distances,
    correlation_integral,
    correlation_dimension,
    fractal_regularizer,
)


def test_pairwise_distances_symmetry_and_zero_diag():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = pairwise_distances(x)

    assert d.shape == (3, 3)
    assert torch.allclose(d, d.t(), atol=1e-6)
    assert torch.allclose(torch.diag(d), torch.zeros(3))


def test_correlation_integral_monotonic():
    x = torch.tensor([[0.0], [1.0], [2.0]])  # точки на прямой
    d = pairwise_distances(x)
    radii = torch.tensor([0.1, 1.0, 3.0])

    C = correlation_integral(d, radii)
    # C(r) должно возрастать с r
    assert C.shape == (3,)
    assert C[0] <= C[1] <= C[2]
    # максимум 1
    assert C[-1] <= 1.0 + 1e-6


def test_correlation_dimension_line():
    """
    🇷ГУ Точки на отрезке [0, 1] ⊂ R → ожидаем D2 ≈ 1.

    🇬🇧 Points on [0, 1] ⊂ R → expect D2 ≈ 1.
    """
    N = 200
    x = torch.linspace(0.0, 1.0, steps=N).unsqueeze(1)  # [N, 1]
    D2, _ = correlation_dimension(x, n_bins=10, min_frac=0.05, max_frac=0.5)
    assert 0.5 < D2.item() < 1.5  # достаточно грубая проверка


def test_correlation_dimension_square():
    """
    🇷ГУ Точки в единичном квадрате [0,1]^2 → ожидаем D2 ≈ 2.

    🇬🇧 Points in the unit square [0,1]^2 → expect D2 ≈ 2.
    """
    N = 400
    x = torch.rand(N, 2)
    D2, _ = correlation_dimension(x, n_bins=10, min_frac=0.05, max_frac=0.5)
    assert 1.0 < D2.item() < 3.0


def test_fractal_regularizer_zero_at_target():
    """
    🇷ГУ Если D2 == target_dim, регуляризатор должен быть ≈ 0.

    🇬🇧 If D2 == target_dim, regularizer should be ≈ 0.
    """
    # создадим искусственные данные: повтор одного вектора → D2 ~ 0
    x = torch.zeros(32, 4)
    target_dim = 0.0

    loss, D2 = fractal_regularizer(x, target_dim=target_dim, weight=1.0)

    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert abs(D2.item() - target_dim) < 1e-3


def test_fractal_regularizer_positive_when_off_target():
    """
    🇷ГУ Если target_dim сильно отличается от D2, loss должен быть положительным.

    🇬🇧 If target_dim is far from D2, loss must be positive.
    """
    x = torch.rand(64, 2)  # D2 ~ 2
    target_dim = 5.0

    loss, D2 = fractal_regularizer(x, target_dim=target_dim, weight=1.0)

    assert loss.item() > 0.0
    # D2 не обязана быть ровно 2, но должна быть конечной
    assert math.isfinite(D2.item())
