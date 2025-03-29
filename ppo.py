import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical







class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        
        
        




class ActorCritic(nn.Module):
    pass