import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from utilss.env import launch_env
from utilss.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def extract_path(control_path):
    path = []
    print("Given control_path is ", control_path)
    with open('C://Users//e0425530//Project-Materials//testcases//milestone1_paths//map1_0_seed1_start_0,1_goal_15,1.txt', 'r') as file:
        for line in file:
            line = line.strip()
            values = line.split(',')
            path.append((int(values[0][1:]), int(values[1].strip()[0])))
    print(path)    
    return path

def prepare_intention_allignment_reward(path, max_reward=0.3):
    intention_reward_table = {}
    for i in range(len(path)):
        cell = path[i]
        intention_reward_table[cell] = i * max_reward
    return intention_reward_table

def get_intention_allignment_reward(intention_reward_table, cur_pos):
    print(f'Current position is {cur_pos}')
    if cur_pos in intention_reward_table:
        print(f'Additional reward of {intention_reward_table[cur_pos]}')
        return intention_reward_table[cur_pos]
    else:
        return -1

def launch_env_with_wrappers():
    env = launch_env()
        # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    return env


def _train(args):   
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Extract the instructed path and create intention_allignment_reward_table table
    path = extract_path(args.control_path)
    intention_allignment_reward_table = prepare_intention_allignment_reward(path)
        
    # # Launch the env with our helper function
    # env = launch_env()
    # print("Initialized environment")

    # # Wrappers
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    # env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    # env = ActionWrapper(env)
    # env = DtRewardWrapper(env)
    # print("Initialized Wrappers")

    # Launch the env with wrappers
    env = launch_env_with_wrappers()
    
    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy, load from pretrained if specified
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    if args.load_pretrained:
        policy.load(filename='ddpg', directory=args.model_dir)
        print(f"Successully loaded the model from {args.model_dir}")

    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    print("Initialized DDPG")
    
    # Evaluate untrained policy
    print("before policy evaluations")
    evaluations= [evaluate_policy(env, policy)]
    print("after policy evaluations")
   
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    
    print("Starting training")
    while total_timesteps < args.max_timesteps:
        
        print("timestep: {} | reward: {}".format(total_timesteps, reward))
            
        if done:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy))
                    print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))

                    if args.save_models:
                        policy.save(filename='ddpg'+str(total_timesteps), directory=args.model_dir)
                    np.savez("./results/rewards.npz",evaluations)

            # Reset environment
            env_counter += 1
            obs = env.reset()
            # env = launch_env_with_wrappers()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            print("action from the policy")
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    args.expl_noise,
                    size=env.action_space.shape[0])
                          ).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, info = env.step(action)
        
        # calculate intention_allignment_reward
        curr_pos = info['curr_pos']
        intention_allignment_reward = get_intention_allignment_reward(intention_allignment_reward_table, curr_pos)
        reward += intention_allignment_reward

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
    
    print("Training done, about to save..")
    policy.save(filename='ddpg', directory=args.model_dir)
    print("Finished saving..should return now!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e3, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument('--model-dir', type=str, default='reinforcement/pytorch/models/')
    parser.add_argument('--control_path', type=str, default='C://Users//e0425530//Project-Materials//testcases//milestone1_paths//map1_0_seed1_start_0,1_goal_15,1.txt')
    parser.add_argument('--load_pretrained', action="store_true", default=False)


    _train(parser.parse_args())
