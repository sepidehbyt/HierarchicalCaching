import numpy as np
import math
import random
import config as cfg
from MLP import MultilayerPerceptron

t_observation = cfg.sys_observation_window_days
rth_thesis = cfg.rth_thesis
rth_paper = cfg.rth_paper
snm_window = cfg.train_min_window_threshold


# content_sizes still not figured out completely
class CloudUnit:

    def __init__(self, cu_repository_size, req_num, data, observation_window_temp):
        self.cu_repository_size = cu_repository_size
        self.mlp = MultilayerPerceptron(data, False, 'irm')
        self.cache = {}
        self.req_num = req_num
        self.observation_window = np.zeros((req_num, t_observation))
        self.contents_sizes = np.ones(req_num)
        col = observation_window_temp.shape[1]
        self.init_observation_window = observation_window_temp[:, col - t_observation:col]
        self.observation_window[:, 0:t_observation] = self.init_observation_window
        self.update_cache()
        self.next_slot_observation()

    def restart(self, cu_repository_size):
        self.cu_repository_size = cu_repository_size
        self.cache = {}
        self.observation_window[:, 0:t_observation] = self.init_observation_window
        self.update_cache()
        self.next_slot_observation()

    def is_cached(self, f, size):
        self.observation_window[f, t_observation - 1] = self.observation_window[f, t_observation - 1] + 1
        return self.just_is_cached(f, size)

    def just_is_cached(self, f, size):
        self.contents_sizes[f] = size
        return self.cache.get(f) is not None

    def cache_f(self, f, size):
        self.cache[f] = size
        self.contents_sizes[f] = size

    def update_cache(self):
        self.cache = {}
        predict = self.mlp.predict(self.observation_window)
        # to calculate cost
        cost = np.zeros((self.req_num, 2))
        for i in range(self.req_num):
            cost[i, 0] = i
            cost[i, 1] = predict[i] / self.contents_sizes[i]
        # to sort cost desc
        cost = cost[(-1 * cost[:, 1]).argsort()]
        # starting to greedy cache
        storage = 0
        for i in range(self.req_num):
            if storage == self.cu_repository_size:
                break
            f = int(cost[i, 0])
            size = self.contents_sizes[f]
            if storage + size <= self.cu_repository_size \
                    and len(np.where(self.observation_window[f] != 0)) > snm_window:
                self.cache_f(f, size)
                storage = storage + size
        pass

    def next_slot_observation(self):
        self.observation_window = np.roll(self.observation_window, -1)
        self.observation_window[:, t_observation - 1] = 0


class ControlUnit:

    def __init__(self, bs_num, bs_size, req_num, data_w_s, data_irm, observation_window_temp):
        self.irm_mlp = MultilayerPerceptron(data_irm, False, 'irm')
        self.ws_mlp = MultilayerPerceptron(data_w_s, False, 'w_s')
        self.bs_num = bs_num
        self.bs_size = bs_size
        self.req_num = req_num
        col_ws = data_w_s.shape[0]
        self.ws_window = np.zeros((1, t_observation))
        self.init_ws_window = np.zeros((1, t_observation))
        self.init_ws_window[0] = data_w_s[col_ws-1, 1:t_observation + 1]
        self.ws_window[0] = self.init_ws_window[0]
        col_irm = observation_window_temp.shape[1]
        self.init_observation_window = observation_window_temp[:, col_irm - t_observation - 1:col_irm]
        self.observation_window = np.zeros((req_num, t_observation + 1))
        self.observation_window[:, 0:t_observation + 1] = self.init_observation_window
        pass

    def reset(self, bs_size):
        self.ws_window[0] = self.init_ws_window[0]
        self.observation_window[:, 0:t_observation + 1] = self.init_observation_window
        self.bs_size = bs_size
        pass

    def next_fraction_contents(self):
        # calculate new fraction
        self.ws_window = np.roll(self.ws_window, -1)
        self.ws_window[0, t_observation - 1] = self.calculate_ws()
        fraction = float(np.round(self.ws_mlp.predict(self.ws_window), 2))
        print(fraction)
        popularity = self.irm_mlp.predict(self.observation_window[:, 1:t_observation + 1])
        best_num = round(self.bs_num * (self.bs_size * (1 - fraction)))
        cost = np.zeros((self.req_num, 2))
        for i in range(self.req_num):
            cost[i, 0] = i
            cost[i, 1] = popularity[i]
        return fraction, cost[(-1 * cost[:, 1]).argsort()][:best_num, 0]

    def calculate_ws(self):
        calc_window = self.observation_window[:, t_observation]
        observation_window_i = self.observation_window[:, :t_observation]
        contents = np.where(calc_window.astype(int) >= rth_paper)[0]
        temp = 0
        for content in contents:
            if (np.where(observation_window_i[content] != 0)[0].shape[0]) == 0:
                temp = temp + calc_window[content]
        return np.round(temp / (np.sum(calc_window[contents])), 2)

    def is_snm(self, f):
        self.observation_window[f, t_observation] = self.observation_window[f, t_observation] + 1
        # return f >= 5000
        return len(np.where(self.observation_window[f, 1:t_observation] != 0)) < snm_window or np.sum(self.observation_window[f, 1:t_observation+1]) <= rth_thesis

    def next_slot_observation(self):
        self.observation_window = np.roll(self.observation_window, -1)
        self.observation_window[:, t_observation] = 0
