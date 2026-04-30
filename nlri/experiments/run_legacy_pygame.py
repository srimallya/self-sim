import importlib

from nlri.envs.maze_reservoir_env import MazeReservoirEnv
from nlri.models.nlri_agent import NLRIAgent


def _fallback_runtime(max_steps: int = 1000):
    env = MazeReservoirEnv(render_mode="human", max_steps=max_steps)
    observations, _ = env.reset(seed=11)
    agents = [NLRIAgent(use_legacy_fallback=True) for _ in observations]
    for agent in agents:
        agent.reset_state()

    step = 0
    try:
        while step < max_steps:
            if env.viewer is not None and env.viewer.closed:
                break
            actions = [agent.act(obs, deterministic=False)[0] for agent, obs in zip(agents, observations)]
            observations, _rewards, terminated, truncated, _info = env.step(actions)
            step += 1
            if step % 100 == 0:
                print(f"legacy-compat step={step}")
            if terminated or truncated:
                break
    finally:
        env.close()


def main():
    try:
        legacy = importlib.import_module("legacy.Simulator_v25")
    except Exception as exc:
        print(f"Legacy TensorFlow simulator unavailable ({exc}). Running pygame compatibility fallback instead.")
        _fallback_runtime()
        return

    legacy.main()


if __name__ == "__main__":
    main()
