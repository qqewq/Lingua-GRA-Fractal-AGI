"""
src/domains/math_domain.py

Lingua‑GRA‑Math: доменный язык на стыке лингвистики и математики
Lingua‑GRA‑Math: a domain-specific language at the intersection of linguistics and mathematics
"""

from __future__ import annotations

from .domain_spec import DomainSpec, LanguageConfig


def default_math_domain(data_path: str) -> DomainSpec:
    """
    RU:
    Описание математического домена: откуда берём данные и какие задачи решаем.

    EN:
    Default specification of the math domain: where data comes from and which tasks we target.
    """
    return DomainSpec(
        name="math",
        data_path=data_path,
        tasks=["generation", "explanation"],  # текст + формулы, объяснения, переформулировки
        extra={
            "description_ru": "Математические тексты (определения, теоремы, доказательства).",
            "description_en": "Mathematical texts (definitions, theorems, proofs).",
        },
    )


def default_math_config(d2_target: float | None = None) -> LanguageConfig:
    """
    RU:
    Конфигурация языка Lingua‑GRA‑Math:
    - уровни: символический и семантический (доменный уровень можно добавить позже);
    - веса GRA-пены по уровням;
    - фрактальный таргет D2 для семантических эмбеддингов (если уже измерен).

    EN:
    Configuration for the Lingua‑GRA‑Math language:
    - levels: symbolic and semantic (a dedicated domain_math level can be added later);
    - GRA foam weights per level;
    - fractal target D2 for semantic embeddings (if already estimated).
    """
    return LanguageConfig(
        name="Lingua-GRA-Math",
        levels=["symbolic", "semantic"],
        lambda_levels={
            "symbolic": 1.0,
            "semantic": 1.0,
        },
        gamma_fractal={
            # фрактальный регуляризатор включаем только на семантическом уровне
            # fractal regularizer is applied only at the semantic level
            "semantic": 0.1,
        },
        d2_target=d2_target,
    )
