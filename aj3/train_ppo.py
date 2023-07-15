import numpy as np
from mltoolkit import parse_config

from aj3.agents.ppo import PPOAgent
from aj3.configs import MazeArguments
from aj3.eval import evaluate_and_store_mp4
from aj3.net import MazeNet
from aj3.train import setup_seed, setup_gym, compute_stats


def setup():
    cfg = MazeArguments(**parse_config('configs/maze.yaml'))
    setup_seed(cfg)
    env = setup_gym()
    return cfg, env


def main():
    cfg, env = setup()

    network = MazeNet(cfg)
    agent = PPOAgent(env, network)

    # Training loop
    num_episodes = cfg.train.num_episodes
    results = []
    for episode in range(num_episodes):
        state = env.reset()

        states = []
        actions = []
        log_probs = []
        rewards = []

        for i in range(env._max_episode_steps):
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

            if done:
                results.append((i + 1, i < env._max_episode_steps - 1))
                break

        # Compute advantages and returns
        returns = []
        advantages = []
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + 0.99 * G  # Discount factor (gamma)
            returns.insert(0, G)
            advantages.insert(0, G - agent.net(states[t])[1])

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Update the policy
        agent.update_policy(states, actions, log_probs, returns, advantages)

        if (episode + 1) % 100 == 0:
            avg_num_steps, avg_success_rate = compute_stats(results)
            print(f'Episode: {episode + 1}, Avg. num steps: {avg_num_steps}, Avg. success rate: {avg_success_rate}')
            results = []
            evaluate_and_store_mp4(env, agent, f'videos/{episode + 1}.mp4')

        evaluate_and_store_mp4(env, agent, f'videos/final.mp4')

    # Evaluate the agent
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _, _ = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")


if __name__ == '__main__':
    main()
