import numpy as np
import math
import random
import config as cfg
from MLP import MultilayerPerceptron
from GenerateRequests import generate_sim_requests
from GenerateTrainData import turn_requests_into_observation_window
from GenerateTrainData import turn_observation_into_w_s
from GenerateTrainData import turn_observation_into_irm
from Graphs import plot_stuff
from Graphs import plot_stuff_
from BS import BaseStationThesis
from BS import BaseStationPaper
from CU import ControlUnit
from CU import CloudUnit
import random
import heapq as hq


t_observation = cfg.sys_observation_window_days
sim_days = cfg.sys_simulation_days
train_days = cfg.sys_train_generation_days
bs_dr = cfg.bs_data_rate
cu_dr = cfg.cu_data_rate
os_dr = cfg.os_data_rate
delay_bs = cfg.delay_bs
delay_cu = cfg.delay_cu
delay_os = cfg.delay_os
delay_n_cu = cfg.delay_os


# content_sizes still not figured out completely
class ThesisSimulation:

    def __init__(self, req_num, irm_mlp, observation_window, bs_count, tot_cache_size, size_fraction, bs_sizes):

        self.bs_count = bs_count
        self.tot_cache_size = tot_cache_size
        self.size_fraction = size_fraction
        if bs_sizes is not None:
            self.bs_cache_size = np.mean(np.array(bs_sizes))
            self.cu_cache_size = tot_cache_size - np.sum(np.array(bs_sizes))
        else:
            self.bs_cache_size, self.cu_cache_size = self.calc_cache_sizes()

        self.cloud_unit = CloudUnit(self.cu_cache_size, req_num, irm_mlp, observation_window)
        self.BSs = []

        for i in range(bs_count):
            if bs_sizes is not None:
                self.BSs.append(BaseStationThesis(req_num, bs_sizes[i], observation_window))
            else:
                self.BSs.append(BaseStationThesis(req_num, self.bs_cache_size, observation_window))

    def sim(self, requests):
        deliveries = []
        hq.heapify(requests)
        hq.heapify(deliveries)
        # simulation
        next_t = train_days + 1
        while len(requests) > 0:
            sys_time, f, size, bs = hq.heappop(requests)
            # to check if its update_window
            if sys_time >= next_t:
                # update cu cache
                self.cloud_unit.update_cache()
                self.cloud_unit.next_slot_observation()
                for i in range(self.bs_count):
                    self.BSs[i].next_slot_observation()
                next_t = next_t + 1
                pass
            delivery_time = 0
            delivery_type = ''
            f = int(f)
            bs = int(bs)
            # checks if f is snm/irm
            is_snm = self.BSs[bs].is_snm(f)
            # check bs
            if self.BSs[bs].is_cached(f):
                # delivery_time = sys_time + size / bs_dr
                delivery_time = delay_bs
                delivery_type = 'bs'
            # checks cu
            elif self.cloud_unit.is_cached(f, size):
                # delivery_time = sys_time + size / cu_dr
                delivery_time = delay_cu
                delivery_type = 'cu'
            # get it from os
            else:
                # delivery_time = sys_time + size / os_dr
                delivery_time = delay_os
                delivery_type = 'os'
                self.BSs[bs].cache_f(f, size)
            hq.heappush(deliveries, (f, sys_time, delivery_time, delivery_type))

        return np.array(deliveries)

    def calc_cache_sizes(self):
        bs_cache_size = round(self.tot_cache_size / (self.bs_count + self.size_fraction))
        cu_cache_size = self.tot_cache_size - round(self.bs_count * bs_cache_size)
        print(bs_cache_size, cu_cache_size)
        return bs_cache_size, cu_cache_size

    def calc_cache_size(self, f):
        bs_cache_size = round(self.tot_cache_size / (self.bs_count + f))
        cu_cache_size = self.tot_cache_size - round(self.bs_count * bs_cache_size)
        print(bs_cache_size, cu_cache_size)
        return bs_cache_size, cu_cache_size

    def restart_net(self, fraction):
        self.size_fraction = fraction
        self.bs_cache_size, self.cu_cache_size = self.calc_cache_sizes()
        self.cloud_unit.restart(self.cu_cache_size)
        for i in range(self.bs_count):
            self.BSs[i].restart(self.bs_cache_size)


class PaperSimulation:

    def __init__(self, req_num, irm_mlp, ws_mlp, ws_fraction, observation_window, bs_count, tot_cache_size, sharing_percent, bs_sizes):

        self.sharing_list = self.create_sharing_list(bs_count, sharing_percent)
        self.bs_count = bs_count
        self.tot_cache_size = tot_cache_size
        self.bs_cache_size = round(tot_cache_size / bs_count)
        self.sharing_percent = sharing_percent

        if bs_sizes is not None:
            self.control_unit = ControlUnit(np.sum(np.array(bs_sizes)), req_num, irm_mlp, ws_mlp, ws_fraction,
                                            observation_window)
        else:
            self.control_unit = ControlUnit(tot_cache_size, req_num, irm_mlp, ws_mlp, ws_fraction, observation_window)

        fraction, irm = self.control_unit.next_fraction_contents()
        self.control_unit.next_slot_observation()
        # bs_irm = int(len(irm) / self.bs_count)
        self.BSs = []

        index = 0
        for i in range(self.bs_count):
            if bs_sizes is not None:
                bs_size = bs_sizes[i]
            else:
                bs_size = self.bs_cache_size
            bs = BaseStationPaper(req_num, bs_size, fraction)
            self.BSs.append(bs)
            bs_fraction = round(bs_size * (1-fraction))
            if index + bs_fraction > len(irm):
                bs_fraction = len(irm) - index
            bs.cache_irm_f(irm[index: bs_fraction], 1)
            # bs.cache_irm_f(irm[i * bs_irm: (i + 1) * bs_irm], 1)
            index = bs_fraction + index

    def sim(self, requests):
        deliveries = []
        fractions = []
        hq.heapify(requests)
        hq.heapify(deliveries)

        # simulation
        next_t = train_days + 1
        while len(requests) > 0:
            sys_time, f, size, bs = hq.heappop(requests)
            delivery_time = 0
            # to check if its update_window
            if sys_time >= next_t:
                fraction, irm = self.control_unit.next_fraction_contents()
                fractions.append(fraction)
                self.control_unit.next_slot_observation()
                bs_irm = int(len(irm) / self.bs_count)
                for i in range(self.bs_count):
                    delivery_time = delivery_time + 1
                    self.BSs[i].update_fraction(fraction)
                    self.BSs[i].cache_irm_f(irm[i * bs_irm: (i + 1) * bs_irm], 1)
                    delivery_time = delivery_time + len(irm[i * bs_irm: (i + 1) * bs_irm])
                next_t = next_t + 1
                pass
            delivery_type = ''
            f = int(f)
            bs = int(bs)
            is_snm = self.control_unit.is_snm(f)
            # check bs
            if self.BSs[bs].is_cached(f):
                # delivery_time = sys_time + size / bs_dr
                # delivery_time = delay_bs
                # delivery_time = 0
                delivery_type = 'bs'
            # checks another BSs
            else:
                delivery_time = delivery_time + self.sharing_percent
                for i in range(self.bs_count):
                    if self.sharing_list[bs][i] == 1:
                        if self.BSs[i].is_cached(f):
                            # delivery_time = sys_time + size / cu_dr
                            # delivery_time = delay_n_cu
                            delivery_time = delivery_time + 2
                            delivery_type = 'cu'
                    else:
                        if self.BSs[i].is_cached(f) and i != bs:
                            # delivery_time = sys_time + size / cu_dr
                            # delivery_time = delay_os
                            delivery_type = '!cu'
                if delivery_type != 'cu':
                    # delivery_time = sys_time + size / cu_dr
                    # delivery_time = delay_os
                    if delivery_type != '!cu':
                        delivery_type = 'os'
                    self.BSs[bs].cache_f(f, size)
            hq.heappush(deliveries, (f, sys_time, delivery_time, delivery_type))
        print(fractions)
        return np.array(deliveries)

    def create_sharing_list(self, bs_count, sharing_percent):
        if sharing_percent == 0:
            sharing_percent = 0
        elif sharing_percent == 1:
            sharing_percent = round((bs_count - 1) / 3)
        elif sharing_percent == 2:
            sharing_percent = round((bs_count - 1) / 3 * 2)
        elif sharing_percent == 3:
            sharing_percent = bs_count - 1
        sharing_list = np.zeros((bs_count, bs_count))
        for i in range(bs_count):
            shares = random.sample(range(1, bs_count + 1), sharing_percent)
            for share in shares:
                sharing_list[i][share - 1] = 1
            if sharing_list[i][i] == 1:
                sharing_list[i][i] = 0
                for j in range(bs_count):
                    if j != i and sharing_list[i][j] != 1:
                        sharing_list[i][j] = 1
                        break
        return sharing_list
