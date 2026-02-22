"""
compute_dimension.py – оценка фрактальной размерности эмбеддингов

🇷🇺
Скрипт для запуска эксперимента:
- загрузка word2vec-эмбеддингов (или любых других);
- выбор подмножества слов;
- вычисление корреляционной размерности D2 для этого облака;
- вывод результатов в консоль и сохранение в JSON.

🇬🇧
Script to run the experiment:
- load word2vec embeddings (or any others);
- select a subset of words;
- compute the correlation dimension D2 for this cloud;
- print results to console and save to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from gensim.models import KeyedVectors  # type: ignore

from .fractal_utils import correlation_dimension


def load_embeddings_from_word2vec(path: str, limit: int | None = None):
    """
    🇷ГУ Загрузить word2vec-совместимую модель и извлечь матрицу эмбеддингов.

    🇬🇧 Load a word2vec-compatible model and extract the embedding matrix.
    """
    kv = KeyedVectors.load_word2vec_format(path, binary=False)
    words = kv.index_to_key[:limit] if limit is not None else kv.index_to_key
    vectors = np.vstack([kv[w] for w in words])
    return words, torch.from_numpy(vectors).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--w2v-path",
        type=str,
        required=True,
        help="Path to word2vec-compatible embeddings (text or binary).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Max number of words to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dimension_result.json",
        help="Where to save the result.",
    )
    args = parser.parse_args()

    words, embeddings = load_embeddings_from_word2vec(args.w2v_path, limit=args.limit)

    # Переводим на устройство (если нужно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embeddings.to(device)

    # Оценка D2
    D2, aux = correlation_dimension(embeddings)

    print(f"Estimated correlation dimension D2: {D2.item():.4f}")
    print(f"Used {len(words)} words, embedding dim = {embeddings.shape[1]}.")

    # Сохраняем результат
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "D2": float(D2.cpu().item()),
        "n_words": len(words),
        "dim": int(embeddings.shape[1]),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
