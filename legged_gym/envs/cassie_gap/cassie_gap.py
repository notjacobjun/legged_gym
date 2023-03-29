from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import Terrain

class CassieGap(LeggedRobot):
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # zero out the measured heights so that the robot doesn't have ability to crouch during low velocity training examples (ablating vision)
        self.measured_heights = torch.zeros_like(self.measured_heights)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _init_buffers(self):
        super()._init_buffers()
        # TODO parse the positions (first 2 indices) from the actor root state tensor
        self.root_pos = self.root_states[:, 0:3]
 
    
    def create_sim(self):
        super().create_sim()
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)
    