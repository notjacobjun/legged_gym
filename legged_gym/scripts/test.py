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
import os

from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import seaborn as sns
import matplotlib.pylab as plt


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

    # list to store the velocities for our experiments (to plot later on our heatmap)
    x_distances_traveled = []

    # prepare environment
    env_cfg.commands.ranges.lin_vel_x = [curr_velocity, curr_velocity]
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # set the target position for this experiment
    # TODO add the targets
    # isaacgym.gymapi.Gym.set_actor_dof_position_targets(env, isaacgym.gymapi.Gym.get_actor_handle(env, 0), target_positions)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
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

    i = steps_alive_after_crossing = avg_base_height = 0
    while num_episodes < 20:
        reset_flag = False
        # check if the robot needs to be reset (has fallen or episode timed out)
        if env.reset_buf and i != 0:
            # calculate the distance traveled and log it
            rb_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)
            # wrap the tensor in pytorch tensor 
            rb_states = gymtorch.wrap_tensor(rb_tensor)
            # access the first rigid body in the list and parse the position values from the rigid body state 
            rb_positions = rb_states[0, 0:3]
            # calculate the distance traveled from the spawn during this episode
            x_distance_from_spawn = (rb_positions[0] - init_position)
            # convert the single element tensor into a scalar
            x_distance_from_spawn = x_distance_from_spawn.view(-1)[0].item()
            x_distances_traveled.append(x_distance_from_spawn)

            # calculate the average height of the robot throughout the episode
            num_episodes += 1
            reset_flag = True
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # increment i counter for logging purposes
        i += 1
        # recalibrate the init_pos to new spawn point after restart
        if reset_flag:
            init_position = env.root_pos[0][0]

        # NOTE that obs[2] gives us out of index bounds so look at docs
        # avg_base_height = obs[2] # TODO perform running average formula
        # TODO This is another possible metric
        # if obs[0] > length_threshold:
        #     steps_alive_after_crossing += 1
        # if obs[2] < 0: 
            # this means the base height is less than 0
            # break
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

    # experiments[curr_velocity] = base_pos_x  + steps_alive_after_crossing # most likely obs[0]
    # TODO check if the robot has passed the length of the gap (somewhere around 3-5 meters)
    # if obs[0] > length_threshold:
        # TODO consider using distance travelled instead of num_successes, then log it with 
        # the current experiment list
        # num_successful_crosses += 1
    # else: 
        # leave it as such
        # continue

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
    
    print(f"distances_traveled: {x_distances_traveled} for velocity: {curr_velocity}")
    return [curr_velocity, x_distances_traveled]


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()

    current_velocity = 0.25
    distances = {}
    x_data, y_data = [], []
    while current_velocity <= 3.0:
        velocity, distances_traveled = test(args, current_velocity)
        # store the distances traveled for this velocity setting
        distances[velocity] = distances_traveled
        # increase the current velocity
        current_velocity += 0.25

    # create data for scatter plot
    for velocity in distances:
        for i, distance in enumerate(distances[velocity]):
            x_data.append(float(velocity))
            y_data.append(distance)

    # create scatter plot using matplotlib
    plt.scatter(x_data, y_data, s=50)
    plt.xlabel('Velocity')
    plt.ylabel('Distance Traveled')
    plt.title('Distances Traveled at Different Velocities')
    plt.show()
