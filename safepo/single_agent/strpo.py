from sTRPO.logger import Logger
from sTRPO.agent import Agent
# from models import AdvPolicy
# from models import AdvValue
# from graph import Graph
# from env import Env

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
# import safety_gym
import argparse
import pickle
import random
import torch
import wandb
import copy
import time
import gym
import os

# try:
#     import safety_gym.envs
# except ImportError:
#     print("can not find safety gym...")

try:
    import safety_gymnasium
except ImportError:
    print("can not find safety gym...")

torch.set_default_tensor_type('torch.DoubleTensor')

def train(main_args):
    algo_idx = 0  #101
    agent_name = 'ExTRPO'
    env_name = main_args.env_name
    max_ep_len = 10000
    max_steps = 15000
    epochs = 50000
    save_freq = 10
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    save_name = os.path.join("data", "{}_{}".format(algo, save_name), time.strftime("%Y-%m-%d_%H-%M-%S"))
    args = {
        'agent_name':agent_name,
        'save_name': save_name,
        'discount_factor':0.99,
        'hidden1':512,
        'hidden2':512,
        'v_lr':2e-4,
        'cost_v_lr':2e-4,
        'value_epochs':200,
        'batch_size':10000,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.3,  # 0.8
        'max_kl': 0.0001, #0.0001,   #0.001
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        # 'cost_d':25.0/1000.0,
        'cost_d':10.0/1000.0,
        'unsafe_agent_path': main_args.unsafe_agent_path,
        'l2_reg': main_args.l2_reg,
    }
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    # env = Env(env_name, seed, max_ep_len)
    env = safety_gymnasium.make(env_name)
    
    agent = Agent(env, device, args)

    # for wandb
    wandb.init(project='sTRPO-Distributional', name=f"{agent_name}_{env_name}",
            config={
               "algorithm": "ExTRPO",
               "env_name": main_args.env_name,
               "type": "Original Value function of STRPO"
           })
    # if main_args.graph: graph = Graph(1000, "TRPO", ['score', 'cv', 'policy objective', 'value loss', 'kl divergence', 'entropy'])

    for epoch in range(epochs):
        trajectories = []
        ep_step = 0
        scores = []
        penalty_costs = []
        cvs = []
        while ep_step < max_steps:
            state, _ = env.reset()
            score = 0
            cost_score = 0
            cv = 0
            step = 0
            while True:
                ep_step += 1
                step += 1
                state_tensor = torch.tensor(state, device=device, dtype=torch.double)
                action_tensor, clipped_action_tensor = agent.getAction(state_tensor, is_train=True)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                # next_state, reward, done, info = env.step(clipped_action)
                # cost = info['cost']
                # print(clipped_action)
                next_state, reward, cost, done, truncated, info  = env.step(clipped_action.squeeze())
                done = True if step >= max_ep_len else done
                done = done or truncated
                # done = True if step >= max_ep_len else done
                fail = True if step < max_ep_len and done else False
                trajectories.append([state, action, reward, cost, done, fail, next_state])

                state = next_state
                score += reward
                cost_score += cost
                # cv += info['num_cv']

                if done or step >= max_ep_len:
                    break

            scores.append(score)
            penalty_costs.append(cost_score)
            cvs.append(cv)

        # v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajs=trajectories)
        v_loss, objective, kl, entropy, cv_loss = agent.train(trajs=trajectories)
        score = np.mean(scores)
        safety_cost = np.mean(penalty_costs)
        # cvs = np.mean(cvs)
        log_data = {"Reward":score, "safety_cost": safety_cost,"value loss":v_loss, "objective":objective, "kl":kl, "entropy":entropy,
                    "cost value loss": cv_loss}
        print(log_data)
        # if main_args.graph: graph.update([score, objective, v_loss, kl, entropy])

        # store logs in memory
        # train_test_log_dir = os.path.join(os.getcwd(), 'train_test_logs', 'ExTRPO_' + env_name)
        # use_path(train_test_log_dir)
        # train_test_log_file = os.path.join(train_test_log_dir, "epoch" + str(epoch) + ".pickle")
        # with open(train_test_log_file, 'wb') as f:
        #     pickle.dump(log_data, f)

        wandb.log(log_data)
        if (epoch + 1)%save_freq == 0:
            agent.save()

    if main_args.graph: graph.update(None, finished=True)

def use_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sTRPO')
    parser.add_argument('--env-name', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--test', action='store_true', help='For test.')
    parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--graph', action='store_true', help='For graph.')
    parser.add_argument('--unsafe-agent-path', type=str, help='Trained unsafe agent path.')
    # Distributional RL specific arguments
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
    args = parser.parse_args()
    dict_args = vars(args)
    if args.test:
        test(args)
    else:
        train(args)
