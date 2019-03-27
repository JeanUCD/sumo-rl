import gym

import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
import numpy as np
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.util.gen_route import write_route_file
import traci

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

metric = []
step = 1

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global step, metric
    r = _locals['self'].env.get_attr('last_reward')[0]['t']
    metric.append({'step': step, 'reward': r})

    step +=1
    return True


if __name__ == '__main__':

    # multiprocess environment
    n_cpu = 2
    env = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                        route_file='nets/2way-single-intersection/single-intersection-gen-hmhm.rou.xml',
                                        out_csv_name='outputs/2way-single-intersection/a2c-contexts-hmhm-400k',
                                        single_agent=True,
                                        use_gui=False,
                                        num_seconds=400000,
                                        min_green=5,
                                        time_to_load_vehicles=0,  
                                        max_depart_delay=10,
                                        phases=[
                                            traci.trafficlight.Phase(32000, "GGrrrrGGrrrr"),  
                                            traci.trafficlight.Phase(2000, "yyrrrryyrrrr"),
                                            traci.trafficlight.Phase(32000, "rrGrrrrrGrrr"),   
                                            traci.trafficlight.Phase(2000, "rryrrrrryrrr"),
                                            traci.trafficlight.Phase(32000, "rrrGGrrrrGGr"),   
                                            traci.trafficlight.Phase(2000, "rrryyrrrryyr"),
                                            traci.trafficlight.Phase(32000, "rrrrrGrrrrrG"), 
                                            traci.trafficlight.Phase(2000, "rrrrryrrrrry")
                                            ]) for i in range(n_cpu)])
    envh = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                        route_file='nets/2way-single-intersection/single-intersection-gen-h.rou.xml',
                                        out_csv_name='outputs/2way-single-intersection/a2c-contexts-h-50k',
                                        single_agent=True,
                                        use_gui=False,
                                        num_seconds=50000,
                                        min_green=5,
                                        time_to_load_vehicles=0,  
                                        max_depart_delay=10,
                                        phases=[
                                            traci.trafficlight.Phase(32000, "GGrrrrGGrrrr"),  
                                            traci.trafficlight.Phase(2000, "yyrrrryyrrrr"),
                                            traci.trafficlight.Phase(32000, "rrGrrrrrGrrr"),   
                                            traci.trafficlight.Phase(2000, "rryrrrrryrrr"),
                                            traci.trafficlight.Phase(32000, "rrrGGrrrrGGr"),   
                                            traci.trafficlight.Phase(2000, "rrryyrrrryyr"),
                                            traci.trafficlight.Phase(32000, "rrrrrGrrrrrG"), 
                                            traci.trafficlight.Phase(2000, "rrrrryrrrrry")
                                            ]) for i in range(n_cpu)])
    envm = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                    route_file='nets/2way-single-intersection/single-intersection-gen-m.rou.xml',
                                    out_csv_name='outputs/2way-single-intersection/a2c-contexts-m-50k',
                                    single_agent=True,
                                    use_gui=False,
                                    num_seconds=50000,
                                    min_green=5,
                                    time_to_load_vehicles=0,  
                                    max_depart_delay=10,
                                    phases=[
                                        traci.trafficlight.Phase(32000, "GGrrrrGGrrrr"),  
                                        traci.trafficlight.Phase(2000, "yyrrrryyrrrr"),
                                        traci.trafficlight.Phase(32000, "rrGrrrrrGrrr"),   
                                        traci.trafficlight.Phase(2000, "rryrrrrryrrr"),
                                        traci.trafficlight.Phase(32000, "rrrGGrrrrGGr"),   
                                        traci.trafficlight.Phase(2000, "rrryyrrrryyr"),
                                        traci.trafficlight.Phase(32000, "rrrrrGrrrrrG"), 
                                        traci.trafficlight.Phase(2000, "rrrrryrrrrry")
                                        ]) for i in range(n_cpu)])        

    # HMHM
    model = A2C(MlpPolicy, env, verbose=1, learning_rate=0.0001, lr_schedule='constant',  tensorboard_log="./a2c/")
    model.learn(total_timesteps=1000000, tb_log_name="hmhm")
    model.save('a2c_hmhm')

    
    model = A2C.load('a2c_hmhm', envh, verbose=1, learning_rate=0.0001, lr_schedule='constant',  tensorboard_log="./a2c/")
    model.learn(total_timesteps=50000, tb_log_name="h")
    
    model.set_env(envm)
    model.learn(total_timesteps=50000, tb_log_name="m")


    #envh.env_method('save_csv')
    
    #model.set_env(envm)
    #model = A2C.load('a2c_m', envm, verbose=1, learning_rate=0.0001, lr_schedule='constant',  tensorboard_log="./a2c/")
    #model.learn(total_timesteps=20000, tb_log_name='m')
    #envm.env_method('save_csv')





    """     df = pd.DataFrame(metric)
    df.to_csv('teste.csv', index=False)

    plt.figure()
    plt.plot(df['step'], df['reward'])
    plt.show() """

    # Enjoy trained agent
    """ obs = env.reset()
    for i in range(100000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action) """


    #model.save("a2c_m")


