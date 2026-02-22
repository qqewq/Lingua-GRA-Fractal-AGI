"""
generate.py – простая генерация текста на базе Lingua GRA

🇷🇺
Скрипт демонстрирует:
- загрузку обученной модели Lingua GRA (символьный + семантический уровни);
- пошаговую генерацию последовательности токенов;
- логирование семантического эмбеддинга и оценку его D2 по мини-батчу.

🇬🇧
This script demonstrates:
- loading a trained Lingua GRA model (symbolic + semantic levels);
- step-by-step token sequence generation;
- logging semantic embeddings and estimating their D2 on a mini-batch.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import torch
import torch.nn.functional as F

from lingua_gra.fractal_utils import correlation_dimension
from lingua_gra.language_levels import Level
from lingua_gra.training import LinguaGRAModel
from lingua_gra.utils import setup_logging, set_seed


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """
    🇷ГУ Сэмплирование следующего токена из logits.

    🇬🇧 Sample next token from logits.
    """
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())
    probs = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_sequence(
    model: LinguaGRAModel,
    start_tokens: List[int],
    max_len: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
):
    """
    🇷ГУ Сгенерировать последовательность токенов, начиная с start_tokens.

    🇬🇧 Generate a token sequence starting from start_tokens.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    tokens = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, T0]
    generated = start_tokens[:]

    with torch.no_grad():
        for _ in range(max_len - len(start_tokens)):
            h_sym, h_sym_proj, logits = model.forward_token_level(tokens)
            # logits: [B, V] – интерпретируем как следующего токена для всего контекста
            next_id = sample_next_token(logits[0], temperature=temperature)
            generated.append(next_id)

            next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            tokens = torch.cat([tokens, next_token], dim=1)

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to a trained Lingua GRA checkpoint (.pt).")
    parser.add_argument("--start-ids", type=str, default="1,2,3", help="Comma-separated list of start token ids.")
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("generate")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Здесь предполагается, что вы где-то определили функцию build_model_from_config
    # или грузите уже готовый LinguaGRAModel из чекпоинта.
    #
    # Пример (псевдокод, зависит от вашего проекта):
    #
    # from lingua_gra.config import load_config_and_build_model
    # config, model = load_config_and_build_model("config.yaml")
    #
    # Здесь оставим заглушку:
    if args.checkpoint is None:
        raise RuntimeError("Checkpoint path must be provided to load a trained model.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model: LinguaGRAModel = checkpoint["model"]  # предполагаем, что так сохранено
    model.to(device)

    start_ids = [int(x) for x in args.start_ids.split(",") if x.strip()]
    gen_ids = generate_sequence(
        model=model,
        start_tokens=start_ids,
        max_len=args.max_len,
        temperature=args.temperature,
        device=device,
    )

    logger.info(f"Generated token ids: {gen_ids}")

    # Дополнительно: оценка D2 для семантических эмбеддингов сгенерированных последовательностей
    # (игрушечный пример: один батч, усреднение токенов → семантический вход).
    tokens = torch.tensor(gen_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        h_sym, h_sym_proj, _ = model.forward_token_level(tokens)
        # допустим, семантический уровень берёт h_sym как вход
        h_sem, h_sem_proj = model.forward_semantic_level(h_sym)

    # чтобы иметь облако точек, дублируем вектор или используем несколько генераций;
    # здесь просто делаем вид, что у нас N копий
    embeddings = h_sem_proj.repeat(16, 1)  # [16, D]
    D2, _ = correlation_dimension(embeddings)
    logger.info(f"Estimated D2 for generated semantic embeddings (toy): {D2.item():.4f}")


if __name__ == "__main__":
    main()
