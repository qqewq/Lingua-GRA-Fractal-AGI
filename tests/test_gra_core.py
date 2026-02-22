"""
test_gra_core.py – юнит-тесты для gra_core.py

🇷🇺
Набор простых тестов:
- HilbertSpace.inner и norm2;
- Projector.foam для линейного проектора;
- GRAFunctional для двух уровней.

🇬🇧
A small unit test suite for gra_core.py:
- HilbertSpace.inner and norm2;
- Projector.foam for a linear projector;
- GRAFunctional with two levels.
"""

from __future__ import annotations

import math

import torch

from lingua_gra.gra_core import HilbertSpace, Projector, GRAFunctional


def test_hilbert_space_basic():
    H = HilbertSpace(dim=3)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[4.0, 5.0, 6.0]])

    inner = H.inner(x, y)  # [1]
    norm2_x = H.norm2(x)

    assert inner.shape == (1,)
    assert math.isclose(inner.item(), 1 * 4 + 2 * 5 + 3 * 6, rel_tol=1e-6)
    assert math.isclose(norm2_x.item(), 1**2 + 2**2 + 3**2, rel_tol=1e-6)


def test_projector_foam_zero_if_identity():
    dim = 4
    proj = Projector(dim=dim)
    # Сделаем из проектора тождественное отображение
    with torch.no_grad():
        for layer in proj.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        # Вручную зададим последний слой как Identity: y = x
        last = proj.net[-1]
        torch.nn.init.eye_(last.weight)
        torch.nn.init.zeros_(last.bias)

    x = torch.randn(8, dim)
    foam_val = proj.foam(x)
    assert foam_val.item() < 1e-6


def test_gra_functional_two_levels():
    dim1, dim2 = 3, 5
    p1 = Projector(dim=dim1)
    p2 = Projector(dim=dim2)

    projectors = {
        0: p1,
        1: p2,
    }
    lambdas = {
        0: 1.0,
        1: 0.5,
    }

    gra = GRAFunctional(projectors=projectors, lambdas=lambdas)

    x0 = torch.randn(4, dim1)
    x1 = torch.randn(4, dim2)
    level_embeddings = {0: x0, 1: x1}

    loss, logs = gra(level_embeddings)

    assert loss.dim() == 0
    assert "foam_l0" in logs
    assert "foam_l1" in logs
    assert logs["foam_l0"] >= 0.0
    assert logs["foam_l1"] >= 0.0
