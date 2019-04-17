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

from stable_baselines2.common.policies import MlpPolicy
from stable_baselines2.common.vec_env import SubprocVecEnv
from stable_baselines2 import A2C

metric = []

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global metric
    t = _locals['self'].env.get_attr('sim_step')[0]
   
    if t == 100000 or t == 300000:
        data, params = A2C._load_from_file('a2c_m')

        model = _locals['self']
        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        print('trocou os pesos!')

    return True


if __name__ == '__main__':

    # multiprocess environment
    n_cpu = 2
    env = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                        route_file='nets/2way-single-intersection/single-intersection-gen-hmhm.rou.xml',
                                        out_csv_name='outputs/2way-single-intersection/a2c-contexts-hmhm-400k-semoraculo',
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
    """envh = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                        route_file='nets/2way-single-intersection/single-intersection-gen-h.rou.xml',
                                        out_csv_name='outputs/2way-single-intersection/a2c-contexts-h-50khmhm',
                                        single_agent=True,
                                        use_gui=True,
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
                                    out_csv_name='outputs/2way-single-intersection/a2c-contexts-m-50k-teste',
                                    single_agent=True,
                                    use_gui=True,
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
                                        ]) for i in range(n_cpu)])  """ 


    model = A2C.load('a2c_hmhm', env, verbose=1, learning_rate=0.0001, lr_schedule='constant',  tensorboard_log="./a2c/")
    model.learn(total_timesteps=1000000, tb_log_name="hmhm-semoraculo", reset_num_timesteps=False)


    """     # HMHM
    model = A2C(MlpPolicy, env, verbose=1, learning_rate=0.0001, lr_schedule='constant',  tensorboard_log="./a2c/")
    model.learn(total_timesteps=650000, tb_log_name="hmhm")
    model.save('a2c_hmhm')

    
    model = A2C.load('a2c_hmhm', envh, verbose=1, learning_rate=0.0001, lr_schedule='constant',  tensorboard_log="./a2c/")
    model.learn(total_timesteps=50000, tb_log_name="h")
    
    model.set_env(envm)
    model.learn(total_timesteps=50000, tb_log_name="m") """


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


