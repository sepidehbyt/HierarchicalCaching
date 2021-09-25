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
import heapq as hq


t_observation = cfg.sys_observation_window_days
sim_days = cfg.sys_simulation_days
train_days = cfg.sys_train_generation_days
bs_dr = cfg.bs_data_rate
cu_dr = cfg.cu_data_rate
os_dr = cfg.os_data_rate


# content_sizes still not figured out completely
class ThesisSimulation:

    def __init__(self, req_num, train_window, observation_window, bs_count, tot_cache_size, size_fraction):

        self.bs_count = bs_count
        self.tot_cache_size = tot_cache_size
        self.size_fraction = size_fraction
        self.bs_cache_size, self.cu_cache_size = self.calc_cache_sizes()

        self.cloud_unit = CloudUnit(self.cu_cache_size, req_num, train_window, observation_window)
        self.BSs = []
        for i in range(bs_count):
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
            if is_snm and self.cloud_unit.just_is_cached(f, size):
                print(f)
            if self.BSs[bs].is_cached(f) and self.cloud_unit.just_is_cached(f, size):
                print(f)
            if is_snm and self.BSs[bs].is_cached(f):
                delivery_time = sys_time + size / bs_dr
                delivery_type = 'bs'
            # checks cu
            elif self.cloud_unit.is_cached(f, size):
                delivery_time = sys_time + size / cu_dr
                delivery_type = 'cu'
            # get it from os
            else:
                delivery_time = sys_time + size / cu_dr
                delivery_type = 'os'
                self.BSs[bs].cache_f(f, size)
            hq.heappush(deliveries, (f, sys_time, delivery_time, delivery_type))

        return np.array(deliveries)

    def calc_cache_sizes(self):
        bs_cache_size = round(self.tot_cache_size / (self.bs_count + self.size_fraction))
        cu_cache_size = self.tot_cache_size - round(self.bs_count * bs_cache_size)
        print('cache sizes changed')
        print(bs_cache_size, cu_cache_size)
        return bs_cache_size, cu_cache_size

    def restart_net(self):
        self.cloud_unit.restart(self.cu_cache_size)
        for i in range(self.bs_count):
            self.BSs[i].restart(self.bs_cache_size)


class PaperSimulation:

    def __init__(self, req_num, train_window, ws_window, observation_window, bs_count, tot_cache_size):

        self.bs_count = bs_count
        self.tot_cache_size = tot_cache_size
        self.bs_cache_size = round(tot_cache_size / bs_count)

        self.control_unit = ControlUnit(bs_count, self.bs_cache_size, req_num, ws_window, train_window, observation_window)
        fraction, irm = self.control_unit.next_fraction_contents()
        self.control_unit.next_slot_observation()
        bs_irm = int(len(irm) / self.bs_count)
        self.BSs = []
        for i in range(self.bs_count):
            bs = BaseStationPaper(req_num, self.bs_cache_size, fraction)
            self.BSs.append(bs)
            bs.cache_irm_f(irm[i * bs_irm: (i + 1) * bs_irm], 1)

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
                fraction, irm = self.control_unit.next_fraction_contents()
                self.control_unit.next_slot_observation()
                bs_irm = int(len(irm) / self.bs_count)
                for i in range(self.bs_count):
                    self.BSs[i].update_fraction(fraction)
                    self.BSs[i].cache_irm_f(irm[i * bs_irm: (i + 1) * bs_irm], 1)
                next_t = next_t + 1
                pass
            delivery_time = 0
            delivery_type = ''
            f = int(f)
            bs = int(bs)
            is_snm = self.control_unit.is_snm(f)
            # check bs
            if self.BSs[bs].is_cached(f):
                delivery_time = sys_time + size / bs_dr
                delivery_type = 'bs'
            # checks another BSs
            else:
                for i in range(self.bs_count):
                    if i != bs:
                        if self.BSs[i].is_cached(f):
                            delivery_time = sys_time + size / cu_dr
                            delivery_type = 'cu'
                if delivery_type != 'cu':
                    delivery_time = sys_time + size / cu_dr
                    delivery_type = 'os'
                    self.BSs[bs].cache_f(f, size)
            hq.heappush(deliveries, (f, sys_time, delivery_time, delivery_type))
        return np.array(deliveries)
