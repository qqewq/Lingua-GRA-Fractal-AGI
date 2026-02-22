"""
simulate.py – симуляция Box World Language с Lingua GRA

🇷🇺
Этот скрипт:
- создаёт простую среду Box World (агент Speaker и агент Mover);
- генерирует сообщения, проходящие через семантический уровень Lingua GRA;
- обучает Mover’а с помощью policy gradient (упрощённый REINFORCE);
- логирует фрактальную размерность эмбеддингов сообщений по мере обучения.

🇬🇧
This script:
- creates a simple Box World environment (Speaker and Mover agents);
- generates messages passing through Lingua GRA’s semantic level;
- trains the Mover using policy gradient (simplified REINFORCE);
- logs the fractal dimension of message embeddings during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lingua_gra.fractal_utils import fractal_regularizer
from lingua_gra.language_levels import LevelConfig, SemanticLevel
from lingua_gra.neural_encoders import PragmaticPolicy
from lingua_gra.utils import setup_logging, set_seed


# ---------------------------------------------------------------------------
# Простейшая среда Box World
# ---------------------------------------------------------------------------


@dataclass
class BoxWorldState:
    agent_pos: np.ndarray  # [2]
    box_pos: np.ndarray    # [2]
    goal_pos: np.ndarray   # [2]
    done: bool = False


class BoxWorldEnv:
    """
    🇷ГУ Очень упрощённый grid-world: агент должен подойти к коробке и затем к цели.

    🇬🇧 Very simplified grid-world: agent must go to the box and then to the goal.
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.reset()

    def reset(self) -> BoxWorldState:
        self.agent_pos = np.array([0, 0], dtype=np.int64)
        self.box_pos = np.array([self.size - 2, 1], dtype=np.int64)
        self.goal_pos = np.array([self.size - 1, self.size - 1], dtype=np.int64)
        self.phase = 0  # 0: к коробке, 1: к цели
        self.done = False
        return self._get_state()

    def _get_state(self) -> BoxWorldState:
        return BoxWorldState(
            agent_pos=self.agent_pos.copy(),
            box_pos=self.box_pos.copy(),
            goal_pos=self.goal_pos.copy(),
            done=self.done,
        )

    def step(self, action: int) -> Tuple[BoxWorldState, float, bool, dict]:
        """
        actions: 0=up, 1=down, 2=left, 3=right, 4=stay
        """
        if self.done:
            return self._get_state(), 0.0, True, {}

        move = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
            4: np.array([0, 0]),
        }.get(action, np.array([0, 0]))

        new_pos = self.agent_pos + move
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos

        reward = -0.01  # маленький штраф за шаг

        if self.phase == 0 and np.array_equal(self.agent_pos, self.box_pos):
            self.phase = 1
            reward += 0.5

        if self.phase == 1 and np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0
            self.done = True

        return self._get_state(), reward, self.done, {}


# ---------------------------------------------------------------------------
# Speaker и Mover
# ---------------------------------------------------------------------------


class Speaker(nn.Module):
    """
    🇷ГУ Speaker получает полное состояние мира и выдаёт эмбеддинг сообщения.

    🇬🇧 Speaker takes full world state and outputs a message embedding.
    """

    def __init__(self, obs_dim: int, msg_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, msg_dim * 2),
            nn.ReLU(),
            nn.Linear(msg_dim * 2, msg_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def encode_full_state(state: BoxWorldState, size: int) -> np.ndarray:
    """
    🇷ГУ One-hot/float-представление состояния для Speaker/Mover.

    🇬🇧 One-hot/float representation of the state for Speaker/Mover.
    """
    vec = np.concatenate(
        [
            state.agent_pos / (size - 1),
            state.box_pos / (size - 1),
            state.goal_pos / (size - 1),
            np.array([float(state.done)], dtype=np.float32),
        ],
        axis=0,
    )
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Основной цикл симуляции
# ---------------------------------------------------------------------------


def run_simulation(
    num_episodes: int = 200,
    max_steps: int = 50,
    grid_size: int = 5,
    msg_dim: int = 32,
    hidden_dim: int = 64,
    n_actions: int = 5,
    target_D2: float = 5.0,
):
    logger = logging.getLogger("simulate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = BoxWorldEnv(size=grid_size)

    obs_dim = 7  # agent(2) + box(2) + goal(2) + done(1)
    speaker = Speaker(obs_dim=obs_dim, msg_dim=msg_dim).to(device)

    # Семантический уровень для сообщений
    sem_cfg = LevelConfig(dim=msg_dim, lambda_weight=1.0, gamma_fractal=0.1)
    semantic_level = SemanticLevel(config=sem_cfg).to(device)

    # Прагматический policy (Mover)
    policy = PragmaticPolicy(
        obs_dim=obs_dim,
        msg_dim=msg_dim,
        hidden_dim=hidden_dim,
        n_actions=n_actions,
    ).to(device)

    params = list(speaker.parameters()) + list(semantic_level.parameters()) + list(
        policy.parameters()
    )
    optimizer = optim.Adam(params, lr=1e-3)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0

        log_probs = []
        rewards = []
        msg_embs = []

        for t in range(max_steps):
            obs_vec = encode_full_state(state, grid_size)
            obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(device)  # [1, obs_dim]

            # Speaker генерирует сообщение (эмбеддинг)
            raw_msg = speaker(obs_t)  # [1, msg_dim]
            # прогоняем через семантический уровень
            h_sem, h_sem_proj, _ = semantic_level(raw_msg)
            msg_emb = h_sem_proj.detach()  # фиксируем семантическую проекцию как сообщение
            msg_embs.append(msg_emb)

            # Mover выбирает действие
            logits = policy(obs_t, msg_emb)  # [1, n_actions]
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            # шаг среды
            next_state, reward, done, _ = env.step(int(action.item()))
            rewards.append(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        # Политика: REINFORCE
        returns = []
        G = 0.0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        log_probs_t = torch.stack(log_probs)  # [T]
        pg_loss = -(log_probs_t * returns_t).mean()

        # Фрактальный регуляризатор по эмбеддингам сообщений в эпизоде
        msg_batch = torch.cat(msg_embs, dim=0)  # [T, msg_dim]
        fract_loss, D2_est = fractal_regularizer(
            msg_batch, target_dim=target_D2, weight=0.1
        )

        loss = pg_loss + fract_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(
            f"episode={ep:04d} "
            f"reward={total_reward:.3f} "
            f"len={len(rewards):02d} "
            f"pg_loss={pg_loss.item():.4f} "
            f"fract_loss={fract_loss.item():.4f} "
            f"D2_msg={D2_est.item():.4f}"
        )


def main():
    setup_logging()
    set_seed(42)
    run_simulation()


if __name__ == "__main__":
    main()
