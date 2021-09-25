import numpy as np
import math
import random
import config as cfg
from LRU import LRUCache

t_observation = cfg.sys_observation_window_days
rth_thesis = cfg.rth_thesis
snm_window = cfg.train_min_window_threshold


class BaseStationThesis:

    def __init__(self, req_num, bs_repository_size, observation_window_temp):
        self.bs_repository_size = bs_repository_size
        self.req_num = req_num
        self.observation_window = np.zeros((self.req_num, t_observation))
        col = observation_window_temp.shape[1]
        self.init_observation_window = observation_window_temp[:, col - t_observation:col]
        self.cache = LRUCache(self.bs_repository_size)
        self.observation_window[:, 0:t_observation] = self.init_observation_window
        self.next_slot_observation()

    def restart(self, bs_repository_size):
        self.bs_repository_size = bs_repository_size
        self.cache = LRUCache(self.bs_repository_size)
        self.observation_window[:, 0:t_observation] = self.init_observation_window
        self.next_slot_observation()

    def is_cached(self, f):
        return self.cache.get(f) != -1

    def cache_f(self, f, size):
        self.cache.put(f, size)

    def is_snm(self, f):
        self.observation_window[f, t_observation - 1] = self.observation_window[f, t_observation - 1] + 1
        # return f >= 5000
        return len(np.where(self.observation_window[f] != 0)[0]) < snm_window and np.sum(self.observation_window[f]) <= rth_thesis

    def next_slot_observation(self):
        self.observation_window = np.roll(self.observation_window, -1)
        self.observation_window[:, t_observation - 1] = 0


class BaseStationPaper:

    def __init__(self, req_num, bs_repository_size, fraction):
        self.req_num = req_num
        self.bs_repository_size = bs_repository_size
        self.fraction = fraction
        self.lru_cache = LRUCache(self.bs_repository_size * self.fraction)
        self.irm_cache = {}

    def is_cached(self, f):
        return self.lru_cache.get(f) != -1 or self.irm_cache.get(f) is not None

    def cache_f(self, f, size):
        self.lru_cache.put(f, size)

    def cache_irm_f(self, contents, size):
        for content in contents:
            self.irm_cache[content] = size

    def update_fraction(self, fraction):
        self.fraction = fraction
        self.lru_cache = LRUCache(self.bs_repository_size * self.fraction)
        self.irm_cache = {}
