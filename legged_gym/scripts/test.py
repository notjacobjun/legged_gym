# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import logging as log
import os
import statistics

from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pylab as plt

def initialize_dataframes():
    velocity_range = np.arange(0.25, 3.25, 0.25)
    gap_range = np.arange(3, 7, 1)
    height_df = pd.DataFrame(columns=gap_range, index=velocity_range)
    distance_df = pd.DataFrame(columns=gap_range, index=velocity_range)
    height_df.to_pickle('height_data.pkl')
    distance_df.to_pickle('distance_data.pkl')

def test(args, curr_velocity):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    x_distances_traveled, episode_heights = [], []

    # prepare environment
    env_cfg.commands.ranges.lin_vel_x = [curr_velocity, curr_velocity]
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # NOTE since env.root_pos initially has an extra dimension we are going to squeeze it
    env.root_pos = torch.squeeze(env.root_pos)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    num_episodes = 0
    init_position = env.root_pos[0]
    print(f"initial pos: {init_position}")
    curr_height = env.root_pos[2]
    i = 0
    avg_base_heights = []

    while num_episodes < 20:
        reset_flag = False

        # get the positioning of the robot from Isaacgym tensor api
        rb_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)
        # wrap the tensor in pytorch tensor 
        rb_states = gymtorch.wrap_tensor(rb_tensor)
        # access the first rigid body in the tensor and parse the position values from the rigid body state 
        rb_positions = rb_states[0, 0:3]

        # log the base height of the robot
        curr_height = rb_positions[2]
        # NOTE make sure that the robot always restarts at height greater than or equal to 0.5 for experiments or else this will give us empty height logs
        # check against logging the height of robot when it is still resetting (therefore skewing the values)
        if not episode_heights and curr_height >= 0.5:
            episode_heights.append(float(curr_height))
        elif episode_heights: # normal height logging case
            episode_heights.append(float(curr_height))

        # check if the robot needs to be reset (has fallen or episode timed out)
        if env.reset_buf.any() and i != 0:
            # calculate the distance traveled from the spawn during this episode
            x_distance_from_spawn = abs(rb_positions[0] - init_position)
            # convert the single element tensor into a scalar
            x_distance_from_spawn = x_distance_from_spawn.view(-1)[0].item()
            x_distances_traveled.append(x_distance_from_spawn)

            # calculate the average height of the robot throughout the episode
            avg_base_height = (sum(episode_heights) / float(len(episode_heights)))
            avg_base_heights.append(avg_base_height)

            num_episodes += 1
            episode_heights = []
            reset_flag = True

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # increment i counter for logging purposes
        i += 1
        # recalibrate the init_pos to new spawn point after restart
        if reset_flag:
            init_position = env.root_pos[0]

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

    if i < stop_state_log:
        logger.log_states(
            {
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            }
        )
    elif i==stop_state_log:
        logger.plot_states()
    if  0 < i < stop_rew_log:
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
    elif i==stop_rew_log:
        logger.print_rewards()
    
    # close the resources for next iteration
    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)

    print(f"distances_traveled: {x_distances_traveled} for velocity: {curr_velocity}")
    print(f"average height for all episodes: {sum(avg_base_heights) / len(avg_base_heights)}")
    return [curr_velocity, x_distances_traveled, avg_base_heights]

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    distances, heights = {}, {}
    x_data, y_data = [], []
    curr_velocity = 2.75
    # NOTE that the gap size here is just used for logging but we have to change the hardcoded gap_size value in the terrain.py script
    curr_gap_size = 4
    while curr_velocity <= 3.00:
        print(f"testing with speed: {curr_velocity}")
        velocity, x_distances_traveled, avg_base_heights = test(args, curr_velocity)
        print(f"heights: {avg_base_heights}")
        print(f"distances: {x_distances_traveled}")
        # store the distances traveled for this velocity setting
        distances[velocity] = x_distances_traveled
        heights[velocity] = avg_base_heights

        # create the pandas dataframe to load into
        if os.path.exists("height_data.pkl"):
            height_df = pd.read_pickle('height_data.pkl')
            height_df[curr_gap_size][curr_velocity] = statistics.fmean(avg_base_heights)
            height_df.to_pickle('height_data.pkl')
            print(height_df)
        else:
            log.error("There is no height data dataframe")
            initialize_dataframes()
            height_df = pd.read_pickle('height_data.pkl')
            height_df[curr_gap_size][curr_velocity] = statistics.fmean(avg_base_heights)
            height_df.to_pickle('height_data.pkl')

        if os.path.exists("distance_data.pkl"):
            distance_df = pd.read_pickle('distance_data.pkl')
            distance_df[curr_gap_size][curr_velocity] = statistics.fmean(x_distances_traveled)
            distance_df.to_pickle('distance_data.pkl')
            print(distance_df)
        else:
            log.error("There is no distance data dataframe")
            initialize_dataframes()
            distance_df = pd.read_pickle('distance_data.pkl')
            distance_df[curr_gap_size][curr_velocity] = statistics.fmean(x_distances_traveled)
            distance_df.to_pickle('distance_data.pkl')
        curr_velocity += 0.25

