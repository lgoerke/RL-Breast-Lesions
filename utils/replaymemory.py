from collections import namedtuple
import random
import numpy as np
import psutil

###########################################################
# Replay memory                                           #
###########################################################

Transition = namedtuple('Transition',
                        ('resized_im_state', 'action', 'reward', 'resized_im_next_state','action_hist'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
#         print('-----',flush=True)
#         print('Current memory has length {}'.format(len(self.memory)),flush=True)
#         print('-----',flush=True)
#         print(psutil.virtual_memory(),flush=True)
#         print('-----',flush=True)
#         print(psutil.swap_memory(), flush=True)
#         print('-----',flush=True)
    
    def push_list(self, transition_list):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition_list
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_seq(self, batch_size, seq_len):
        positions = np.random.randint(0, self.position, batch_size)
        output = []
        for p in positions:
            half_len = seq_len // 2
            rest = seq_len % 2
            if p - half_len < 0:
                p = half_len
            output.append(self.memory[p - half_len:p + half_len + rest])
        return output

    def __len__(self):
        return len(self.memory)
