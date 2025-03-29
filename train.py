import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import roboschool

from PPO import PPO



################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######