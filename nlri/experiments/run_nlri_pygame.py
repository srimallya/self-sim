from nlri.envs.maze_reservoir_env import MazeReservoirEnv
from nlri.models.nlri_agent import NLRIAgent
from nlri.training.metrics import RunningMetrics


def main(max_steps: int = 1000, seed: int = 7):
    env = MazeReservoirEnv(render_mode="human", max_steps=max_steps)
    observations, info = env.reset(seed=seed)
    agents = [NLRIAgent(use_legacy_fallback=True) for _ in observations]
    for agent in agents:
        agent.reset_state()

    metrics = RunningMetrics()
    step = 0
    try:
        while step < max_steps:
            if env.viewer is not None and env.viewer.closed:
                break

            actions = []
            debug_rows = []
            for agent, obs in zip(agents, observations):
                action, debug = agent.act(obs)
                actions.append(action)
                debug_rows.append(debug)

            observations, rewards, terminated, truncated, info = env.step(actions)
            metrics.update(info, debug_rows)
            step += 1

            if step % 100 == 0:
                print(metrics.format_status(step))

            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
