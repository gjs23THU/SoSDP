import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from SoS_env import SoSEnv


def check_interface() -> None:
    env = SoSEnv(num_samples=128, seed=2026)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def check_reproducibility() -> None:
    env = SoSEnv(num_samples=128, seed=2026)

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)

    np.testing.assert_allclose(obs1["static"], obs2["static"])
    np.testing.assert_allclose(obs1["dynamic"], obs2["dynamic"])
    np.testing.assert_allclose(obs1["mask"], obs2["mask"])
    np.testing.assert_allclose(obs1["t"], obs2["t"])
    assert obs1["step_count"] == obs2["step_count"], "step_count not reproducible"

    env.close()


def rollout_stress_test(episodes: int = 1000) -> dict:
    env = SoSEnv(num_samples=512, seed=2026)

    total_rewards = []
    total_steps = []
    invalid_terminations = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done:
            valid_actions = np.flatnonzero(obs["mask"] > 0.5)
            if valid_actions.size > 0:
                action = int(env.np_random.choice(valid_actions))
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            if not np.isfinite(reward):
                raise AssertionError("Reward contains NaN/Inf")
            if not np.isfinite(obs["static"]).all():
                raise AssertionError("Observation static contains NaN/Inf")
            if not np.isfinite(obs["dynamic"]).all():
                raise AssertionError("Observation dynamic contains NaN/Inf")
            if not np.isfinite(obs["mask"]).all():
                raise AssertionError("Observation mask contains NaN/Inf")
            if not np.isfinite(obs["t"]).all():
                raise AssertionError("Observation t contains NaN/Inf")
            if (obs["t"] < 0).any():
                raise AssertionError("Task demand became negative")

            ep_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

            if done and info.get("invalid", False):
                invalid_terminations += 1

            if steps > env.max_steps + 1:
                raise AssertionError("Episode exceeded expected max steps")

        total_rewards.append(ep_reward)
        total_steps.append(steps)

    env.close()

    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "min_reward": float(np.min(total_rewards)),
        "max_reward": float(np.max(total_rewards)),
        "mean_steps": float(np.mean(total_steps)),
        "max_steps_observed": int(np.max(total_steps)),
        "invalid_terminations": int(invalid_terminations),
    }


def main() -> None:
    print("[1/3] Running Gymnasium interface checks...")
    check_interface()
    print("PASS: interface check")

    print("[2/3] Running reproducibility checks...")
    check_reproducibility()
    print("PASS: reproducibility check")

    print("[3/3] Running random rollout stress test...")
    stats = rollout_stress_test(episodes=1000)
    print("PASS: stress test")

    print("\n===== SoSEnv self-check summary =====")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
