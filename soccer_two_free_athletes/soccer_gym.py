import numpy as np
from mlagents_envs.base_env import ActionTuple, BaseEnv, DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class TransUnity2Gym():
    def __init__(self,env):
        env.reset()
        self.environment  = env
        self.calculate_agent_num()
        
    def calculate_agent_num(self):
        agent_sum = 0
        behavior_names = list(self.environment.behavior_specs)
        for behavior_name in behavior_names:
            decision_steps, terminal_steps = self.environment.get_steps(behavior_name)
            agent_sum += len(decision_steps)
        self.n = agent_sum
        print('self.agent_sum = ',self.n)
        
    def reset(self):
        
        env = self.environment
        env.reset()
        behavior_names = list(env.behavior_specs)
        obs_n=[0 for _ in range(self.n)]
        
        for behavior_name in behavior_names:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if len(terminal_steps.agent_id) > 0:
                for agent_id_terminated in terminal_steps:
                    obs  = terminal_steps[agent_id_terminated].obs
                    obs_s = np.concatenate([obs[0],obs[1]])
                    obs_n[agent_id_terminated] = obs_s

                    print(' reset terminal_steps obs_n = ',obs_s.shape)
                
            if len(decision_steps.agent_id) > 0:
                for agent_id_decision in decision_steps:
                     
                    obs  = decision_steps[agent_id_decision].obs
                    obs_s = np.concatenate([obs[0],obs[1]])
                    obs_n[agent_id_decision] = obs_s

                    print(' reset decision_steps obs_n = ',obs_s.shape)
                
        print('reset = ',obs_n[0].shape)
        return obs_n
    
    def step(self,action_n):
        
        env = self.environment
        behavior_names = list(env.behavior_specs)   

        next_obs_n = [0 for _ in range(self.n)]
        reward_n = [0 for _ in range(self.n)]
        done_n = [False for _ in range(self.n)]
        info = [0 for _ in range(self.n)]        
        
        for behavior_name in behavior_names:
            
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            # print('pp_terminal_steps.agent_id',terminal_steps.agent_id)
            # print('pp_decision_steps.agent_id',decision_steps.agent_id)
            
            if(len(terminal_steps.agent_id)==0):
                action = []
                for i_d in decision_steps.agent_id:
                    action.append(action_n[i_d])

                action = np.array(action)
                action_tuple = ActionTuple()
                action_tuple.add_discrete(action)
                env.set_actions(behavior_name,action_tuple)
                env.step()
            else:
                for agent_id_terminated in terminal_steps:
                    
                    done = terminal_steps[agent_id_terminated].interrupted
                    obs  = terminal_steps[agent_id_terminated].obs
                    reward = terminal_steps[agent_id_terminated].reward
                    obs_s = np.concatenate([obs[0],obs[1]])
                    
                    done_n[agent_id_terminated] = done
                    next_obs_n[agent_id_terminated] = obs_s
                    reward_n[agent_id_terminated] = reward

                return next_obs_n, reward_n, done_n, info


            
        for behavior_name in behavior_names:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            #print('decision_steps.agent_id',decision_steps.agent_id,len(decision_steps.agent_id))
            #print('terminal_steps.agent_id',terminal_steps.agent_id,len(terminal_steps.agent_id))

            #print('decision_steps.obs',decision_steps.obs[0].shape)        
            #print('terminal_steps.obs',terminal_steps.obs[0].shape)
            
            #print('decision_steps.reward',decision_steps.reward)        
            #print('terminal_steps.reward',terminal_steps.reward)
            
            obs_s = np.concatenate([decision_steps.obs[0],decision_steps.obs[1]],axis=1)
            #print(obs_s.shape)
            #print('decision_steps.reward',decision_steps.reward)
            
            #print('terminal_steps.reward',terminal_steps.reward)
            #print('terminal_steps.interrupted=',terminal_steps.interrupted) 
            
            local_done = False
            
            if len(terminal_steps.agent_id) > 0:
                for agent_id_terminated in terminal_steps:
                    
                    done = terminal_steps[agent_id_terminated].interrupted
                    obs  = terminal_steps[agent_id_terminated].obs
                    reward = terminal_steps[agent_id_terminated].reward
                    obs_s = np.concatenate([obs[0],obs[1]])
                    
                    done_n[agent_id_terminated] = done
                    next_obs_n[agent_id_terminated] = obs_s
                    reward_n[agent_id_terminated] = reward
                    
                    # print('dddone=',done,'agent_id_terminated=',agent_id_terminated) 
                
            if len(decision_steps.agent_id) > 0:
                # print('decision_steps.agent_id=',decision_steps.agent_id)
                for agent_id_decision in decision_steps:
                     
                    obs  = decision_steps[agent_id_decision].obs
                    reward = decision_steps[agent_id_decision].reward
                    obs_s = np.concatenate([obs[0],obs[1]])
                    
                    next_obs_n[agent_id_decision] = obs_s
                    reward_n[agent_id_decision] = reward
                    
                    #print('agent_id_decision=',agent_id_decision) 

                

                

        if done_n[0] == True:  
            print(len(next_obs_n),next_obs_n[0].shape,reward_n,done_n)
        return next_obs_n, reward_n, done_n, info
        