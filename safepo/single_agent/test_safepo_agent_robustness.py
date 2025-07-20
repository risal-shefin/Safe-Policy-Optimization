import argparse
import os

import yaml
import torch
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params
from safepo.common.model import ActorVCritic
import pandas as pd

default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 40,
    'max_grad_norm': 40.0,
}

isaac_gym_specific_cfg = {
    'total_steps': 100000000,
    'steps_per_epoch': 32768,
    'hidden_sizes': [1024, 1024, 512],
    'gamma': 0.96,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
}

def test_eval(env,
              policy,
              device,
              attack_type=None,
              attack_frequency=0,
             ):
    
    reward_list = []
    cost_list = []
    epochs = 100
    for epoch in range(epochs):
        seed_value = random.randint(0, 10000000)
        state, _ = env.reset(seed = seed_value)
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        eval_rew = 0
        eval_cost = 0
        step = 0
        eval_done = False

        while not eval_done:
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(state, deterministic=True)

            act = act.detach().squeeze().cpu().numpy()

            if attack_type=="random_atk" and random.random()<attack_frequency:
                act = env.action_space.sample()

            next_state, reward, cost, terminated, truncated, info = env.step(act)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            eval_rew += reward[0]
            eval_cost += cost[0]
            step += 1
            eval_done = terminated[0] or truncated[0]
            state = next_state

        reward_list.append(eval_rew)
        cost_list.append(eval_cost)

    print("test_eval completed with attack rate:", attack_frequency)
    return reward_list, cost_list

def test(main_args):
    env_name = main_args.env_name
    algo_name = main_args.algo_name
    saved_data_path = main_args.saved_data_path

    print("---- Algorithm: {}, Environment: {} ----".format(algo_name, env_name))

    if not os.path.exists(saved_data_path):
        raise FileNotFoundError("Invalid Saved Data Path! Saved Data doesn't exist!")
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    cfg_env={}
    base_path = os.path.dirname(os.path.abspath(__file__)).replace("utils", "multi_agent")

    if env_name not in isaac_gym_map.keys():
        env, obs_space, act_space = make_sa_mujoco_env(
            num_envs=1, env_id=env_name, seed=seed
        )
        eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=env_name, seed=None)
        config = default_cfg
    else:
        cfg_env_path = "marl_cfg/{}.yaml".format(isaac_gym_map[env_name])
        with open(os.path.join(base_path, cfg_env_path), 'r') as f:
            cfg_env = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_env["name"] = env_name
            if "task" in cfg_env:
                if "randomize" not in cfg_env["task"]:
                    cfg_env["task"]["randomize"] = main_args.randomize
                else:
                    cfg_env["task"]["randomize"] = False

        sim_params = parse_sim_params(args, cfg_env, None)
        env = make_sa_isaac_env(args=args, cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        config = isaac_gym_specific_cfg

    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)

    # load the saved data
    saved_data = joblib.load(saved_data_path)
    policy.actor.load_state_dict(saved_data["actor_state_dict"])
    policy.reward_critic.load_state_dict(saved_data["reward_critic_state_dict"])
    policy.cost_critic.load_state_dict(saved_data["cost_critic_state_dict"])
    eval_env.obs_rms = saved_data["Normalizer"]
    print("Environment & Models loaded.", flush=True)


    atk_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # atk_rates = [0.0, 0.25, 0.5, 0.75, 1.0]  # Reduced for faster testing
    score_vec = []
    score_std_vec = []
    score_min_vec = []
    score_max_vec = []
    cost_vec = []
    cost_std_vec = []
    cost_min_vec = []
    cost_max_vec = []
    for rate in atk_rates:
        print("Test_eval with attack rate: {} starts: ".format(rate), flush=True)
        r_list, c_list = test_eval(eval_env,
                                  policy,
                                  device,
                                  attack_type = "random_atk",
                                  attack_frequency = rate                                   
                                 )
        numpy_score_array = np.array(r_list)
        mean_score_value = np.mean(numpy_score_array)
        score_std_deviation = np.std(numpy_score_array)
        score_vec.append(mean_score_value)
        score_std_vec.append(score_std_deviation)
        score_min_vec.append(np.min(numpy_score_array))
        score_max_vec.append(np.max(numpy_score_array))

        numpy_cost_array = np.array(c_list)
        mean_cost_value = np.mean(numpy_cost_array)
        cost_std_deviation = np.std(numpy_cost_array)
        cost_vec.append(mean_cost_value)
        cost_std_vec.append(cost_std_deviation)
        cost_min_vec.append(np.min(numpy_cost_array))
        cost_max_vec.append(np.max(numpy_cost_array))

    print("Reward vec:", score_vec)
    print("Reward std vec:", score_std_vec)
    print("Reward min vec:", score_min_vec)
    print("Reward max vec:", score_max_vec)
    print("Cost vec:", cost_vec)
    print("Cost std vec:", cost_std_vec)
    print("Cost min vec:", cost_min_vec)
    print("Cost max vec:", cost_max_vec)
    return {
        "atk_rates": atk_rates,
        "reward_mean": score_vec,
        "reward_std": score_std_vec,
        "reward_min": score_min_vec,
        "reward_max": score_max_vec,
        "cost_mean": cost_vec,
        "cost_std": cost_std_vec,
        "cost_min": cost_min_vec,
        "cost_max": cost_max_vec,
    }

def plot_results(x_values, x_label, y_values, y_min, y_max, y_label, logdir):
    """
    Plot results with standard deviation highlighted in background
    """
    plt.figure(figsize=(10, 6))
    
    # Plot main line
    plt.plot(x_values, y_values, 'b-', linewidth=2, marker='o', markersize=6)
    
    # Fill area for standard deviation
    plt.fill_between(x_values, y_min, y_max, alpha=0.2, color='blue')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(logdir, exist_ok=True)
    
    # Save plot
    filename = f"{y_label.lower().replace(' ', '_')}_vs_{x_label.lower().replace(' ', '_')}.png"
    filepath = os.path.join(logdir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {filepath}")

def save_results_to_csv(results, logdir):
    """
    Save results dictionary to CSV file
    """
    os.makedirs(logdir, exist_ok=True)
    filepath = os.path.join(logdir, "results.csv")
    # Convert dictionary to DataFrame
    df = pd.DataFrame(results)
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}", flush=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Safepo')
    parser.add_argument('--env-name', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--algo-name', default='none', help='Set your algorithm name')
    parser.add_argument('--saved-data-path', default='none', help='Directory of your saved model')
    parser.add_argument("--randomize", type=bool, default=False, help="Wheather to randomize the environments' initial states")
    args = parser.parse_args()
    dict_args = vars(args)
    results = test(args)

    curdir = os.getcwd()
    logdir = os.path.join(curdir, "robust_test", args.env_name, args.algo_name)
    save_results_to_csv(results, logdir)
    plot_results(results["atk_rates"], "Attack Rate", results["reward_mean"], results["reward_min"], results["reward_max"], "Reward", logdir)
    plot_results(results["atk_rates"], "Attack Rate", results["cost_mean"], results["cost_min"], results["cost_max"], "Cost", logdir)
