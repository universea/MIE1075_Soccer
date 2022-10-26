import soccer_twos

env = soccer_twos.make(render=True,flatten_branched=True)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)


print(env.observation_space)
print(env.action_space)

team0_reward = 0
team1_reward = 0
env.reset()

obs, reward, done, info = env.step(
    {
        0: env.action_space.sample(),
        1: env.action_space.sample(),
        2: env.action_space.sample(),
        3: env.action_space.sample(),
    }
)



import os
import time
import argparse
import numpy as np
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from parl.env.multiagent_env import MAenv
from parl.utils import logger, summary
from gym import spaces

CRITIC_LR = 0.01  # learning rate for the critic model
ACTOR_LR = 0.01  # learning rate of the actor model
GAMMA = 0.95  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 1024
MAX_STEP_PER_EPISODE = 10240  # maximum step per episode
EVAL_EPISODES = 3


# Runs policy and returns episodes' rewards and steps for evaluation
def run_evaluate_episodes(env, agents, eval_episodes):
    eval_episode_rewards = []
    eval_episode_steps = []
    while len(eval_episode_rewards) < eval_episodes:
        obs_n = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = {}

            for agent, obs in zip(agents, obs_n):
                action_n[obs] = np.argmax(agent.sample(obs_n[obs]))
                
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            for i, reward in enumerate(reward_n):
                total_reward += reward_n[i]

        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
    return eval_episode_rewards, eval_episode_steps


def run_episode(env, agents):
    obs_n = env.reset()
    done = False
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1

        action_n = {}

        for agent, obs in zip(agents, obs_n):
            action_n[obs] = np.argmax(agent.sample(obs_n[obs]))

        next_obs_n, reward_n, done_n, _ = env.step(action_n)

        done = done_n['__all__']

        # store experience
        for i, agent in enumerate(agents):

            agent.add_experience(obs_n[i], action_n[i], reward_n[i],next_obs_n[i], done_n['__all__'])

        # compute reward of every agent
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward_n[i]
            agents_reward[i] += reward_n[i]

        # show model effect without training
        if False:#args.restore
            continue

        # learn policy
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)

    return total_reward, agents_reward, steps


def main():

    # build agents
    agents = []
    
    players_num = 4

    multi_action_space=[env.action_space,env.action_space,env.action_space,env.action_space]
    multi_obs_shape_n=[336, 336, 336, 336]
    multi_act_shape_n=[27,27,27,27]
    critic_in_dim = sum(multi_obs_shape_n) + sum(multi_act_shape_n)
    
    env.n = 4

    for i in range(players_num):
        
        model = MAModel(multi_obs_shape_n[i], multi_act_shape_n[i], critic_in_dim, False)
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=multi_action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=multi_obs_shape_n,
            act_dim_n=multi_act_shape_n,
            batch_size=BATCH_SIZE)
        
        agents.append(agent)
        


        
    if False:#args.restore:
        # restore modle
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    total_steps = 0
    total_episodes = 0
    while total_episodes <= 3000:
        # run an episode
        ep_reward, ep_agent_rewards, steps = run_episode(env, agents)
        summary.add_scalar('train/episode_reward_wrt_episode', ep_reward,
                           total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', ep_reward,
                           total_steps)
        logger.info(
            'total_steps {}, episode {}, reward {}, agents rewards {}, episode steps {}'
            .format(total_steps, total_episodes, ep_reward, ep_agent_rewards,
                    steps))

        total_steps += steps
        total_episodes += 1

        # evaluste agents
        if total_episodes % 20 == 0:

            eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(
                env, agents, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward',
                               np.mean(eval_episode_rewards), total_episodes)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, np.mean(eval_episode_rewards)))

            # save model
            if True:#not args.restore:
                model_dir = './model'
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)
                    agents[i].save(model_dir + model_name)

main()