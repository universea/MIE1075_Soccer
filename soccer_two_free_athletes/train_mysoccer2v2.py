from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.base_env import ActionTuple, BaseEnv, DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from soccer_gym import TransUnity2Gym

origin_env = UnityEnvironment(file_name="C://Users//raman//Documents//Pengsong//MIE1075_Soccer//buildmysoccer//SoccerTwos", seed=1, side_channels=[])




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
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = []

            for i in range(env.n):
                argm = np.argmax(agents[i].sample(obs_n[i]))
                print('np.argmax(agents[i].sample(obs_n[i]))====',argm)
                action = translate_actions(argm)
                action_n.append(action)
          
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

        action_n = []
        action_transform_n = []

        for i in range(env.n):

            # 模型输出的action类型
            action = np.argmax(agents[i].sample(obs_n[i]))
            action_n.append(action)

            # 使用的action类型
            action_transform = translate_actions(action)
            action_transform_n.append(action_transform)            

        next_obs_n, reward_n, done_n, _ = env.step(action_transform_n)

        done = False
        if True in done_n:
            done = True
            print('done all = ', done)

        # store experience
        for i, agent in enumerate(agents):

            agent.add_experience(obs_n[i], action_n[i], reward_n[i],next_obs_n[i], done)

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

    env = TransUnity2Gym(origin_env)

    agents = []
    agents_sum = 0
    behavior_agents_num = 0

    multi_action_space = [[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
    multi_obs_shape_n = [336, 336, 336, 336]
    multi_act_shape_n = [27,27,27,27]
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