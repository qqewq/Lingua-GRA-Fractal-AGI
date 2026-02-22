"""
utils.py – утилиты для логирования, сохранения метрик и воспроизводимости

🇷🇺
В этом модуле:
- настройка логирования;
- фиксация random seed;
- сохранение метрик обучения в JSONL;
- простая прогресс-обёртка вокруг tqdm.

🇬🇧
This module provides:
- logging setup;
- random seed fixing;
- saving training metrics to JSONL;
- a simple tqdm-based progress wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import torch
from tqdm import tqdm  # type: ignore


# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    🇷ГУ Настроить базовое логирование (в консоль и, опционально, в файл).[web:192][web:195]

    🇬🇧 Configure basic logging (to console and optionally to file).
    """
    # сбрасываем возможные старые хендлеры
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """
    🇷ГУ Зафиксировать random seed для random, numpy и torch.

    🇬🇧 Fix random seed for random, numpy and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Метрики и JSONL
# ---------------------------------------------------------------------------


@dataclass
class MetricRecord:
    """
    🇷ГУ Одна запись метрик (например, один шаг/эпоха).

    🇬🇧 Single metrics record (e.g., per step/epoch).
    """

    step: int
    split: str  # "train", "val", "test"
    metrics: Dict[str, float]


def append_metrics_jsonl(path: str, record: MetricRecord) -> None:
    """
    🇷ГУ Добавить одну запись метрик в JSONL-файл.[web:198][web:201]

    🇬🇧 Append a single metrics record to a JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Прогресс-бар
# ---------------------------------------------------------------------------


def tqdm_wrap(iterable: Iterable[Any], desc: str = "", total: Optional[int] = None) -> Iterator[Any]:
    """
    🇷ГУ Обёртка над tqdm для удобного отображения прогресса.[web:197][web:200]

    🇬🇧 Thin wrapper around tqdm for progress display.
    """
    return tqdm(iterable, desc=desc, total=total)
