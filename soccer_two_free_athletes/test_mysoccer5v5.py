from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.base_env import ActionTuple, BaseEnv, DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from soccer_gym import TransUnity2Gym
engine_channel = EngineConfigurationChannel()

engine_channel.set_configuration_parameters(time_scale=3,quality_level=0)

#origin_env = UnityEnvironment(file_name="C://Users//raman//Documents//Pengsong//MIE1075_Soccer//buildmysoccer//SoccerTwos", seed=1, side_channels=[engine_channel])
origin_env = UnityEnvironment(file_name="C://Users//raman//Documents//Pengsong//MIE1075_Soccer//osoccer//UnityEnvironment", seed=1, no_graphics=False,side_channels=[engine_channel])


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
MAX_STEP_PER_EPISODE = 2000  # maximum step per episode
EVAL_EPISODES = 3

def translate_actions(action):
    if action == 0:
        actions=np.array([0,0,0])
    elif action == 1:
        actions=np.array([0,0,1])
    elif action == 2:
        actions=np.array([0,0,2])
    elif action == 3:
        actions=np.array([0,1,0])
    elif action == 4:
        actions=np.array([0,1,1])
    elif action == 5:
        actions=np.array([0,1,2])
    elif action == 6:
        actions=np.array([0,2,0])
    elif action == 7:
        actions=np.array([0,2,1])
    elif action == 8:
        actions=np.array([0,2,2])
    elif action == 9:
        actions=np.array([1,0,0])
    elif action == 10:
        actions=np.array([1,0,1])
    elif action == 11:
        actions=np.array([1,0,2])
    elif action == 12:
        actions=np.array([1,1,0])
    elif action == 13:
        actions=np.array([1,1,1])
    elif action == 14:
        actions=np.array([1,1,2])
    elif action == 15:
        actions=np.array([1,2,0])
    elif action == 16:
        actions=np.array([1,2,1])
    elif action == 17:
        actions=np.array([1,2,2])
    elif action == 18:
        actions=np.array([2,0,0])
    elif action == 19:
        actions=np.array([2,0,1])
    elif action == 20:
        actions=np.array([2,0,2])
    elif action == 21:
        actions=np.array([2,1,0])
    elif action == 22:
        actions=np.array([2,1,1])
    elif action == 23:
        actions=np.array([2,1,2])
    elif action == 24:
        actions=np.array([2,2,0])
    elif action == 25:
        actions=np.array([2,2,1])
    elif action == 26:
        actions=np.array([2,2,2])

    return actions


# Runs policy and returns episodes' rewards and steps for evaluation
def run_evaluate_episodes(env, agents, eval_episodes):
    eval_episode_rewards = []
    eval_episode_steps = []
    while len(eval_episode_rewards) < eval_episodes:
        obs_n = env.reset()
        done = False
        total_reward = 0
        agents_reward = [0 for _ in range(env.n)]
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1

            action_n = []
            action_transform_n = []

            for i in range(env.n):

                # ???????????????action??????
                action = np.argmax(agents[i].sample(obs_n[i]))
                action_n.append(action)

                # ?????????action??????
                action_transform = translate_actions(action)
                action_transform_n.append(action_transform)            

            obs_n, reward_n, done_n, _ = env.step(action_transform_n)
          
            done = False
            if True in done_n:
                done = True
                print('done all = ', done)

            for i in range(env.n):
                total_reward += reward_n[i]
                agents_reward[i] += reward_n[i]

        eval_episode_rewards.append(agents_reward)
        eval_episode_steps.append(steps)
    return eval_episode_rewards, eval_episode_steps




def main():

    env = TransUnity2Gym(origin_env)

    agents = []
    agents_sum = 0
    behavior_agents_num = 0

    multi_action_space = [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
    multi_obs_shape_n = [336, 336, 336, 336, 336, 336, 336, 336, 336, 336]
    multi_act_shape_n = [27,27,27,27,27,27,27,27,27,27]
    critic_in_dim = sum(multi_obs_shape_n) + sum(multi_act_shape_n)

    # build agents

    for i in range(env.n):
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
        


        
    if True:#args.restore:
        # restore modle
        for i in range(len(agents)):
            model_file = 'model/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)
        print('restor model success!')

    total_steps = 0
    total_episodes = 0

    # evaluste agents
    for total_episodes in range(50):

        eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(
            env, agents, EVAL_EPISODES)
        summary.add_scalar('eval/episode_reward',
                            np.mean(eval_episode_rewards), total_episodes)
        logger.info('Evaluation over: {} episodes, Reward: {}'.format(
            EVAL_EPISODES, eval_episode_rewards))


main()