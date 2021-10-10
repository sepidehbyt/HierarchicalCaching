import csv
import numpy as np
import math
import sys
import random
import config as cfg
from PoissonProcess import generate_fix_rate


skewness = cfg.snm_volume_skewness
volume_min = cfg.snm_volume_min
snm_lifespan = cfg.snm_lifespan
number_of_requests_start_index = cfg.irm_number_contents


def generate_snm_requests(sim_days, number_of_bs, arrival_rate):
    # (ts, f, size, bs)
    requests = []
    add_index = 0
    for i in range(number_of_bs):
        requests_arrival_time = generate_fix_rate(sim_days, arrival_rate, sys.maxsize)
        # requests_arrival_time = generate_fix_rate(sim_days - snm_lifespan, arrival_rate, sys.maxsize)
        number_of_requests = len(requests_arrival_time)
        snm_lifespans = np.random.poisson(snm_lifespan, number_of_requests) + 1
        # request_sizes = np.random.poisson(mean_size, number_of_requests) + 1
        request_sizes = np.ones(number_of_requests)
        volumes = np.round((np.random.pareto(skewness, number_of_requests) + 1) * volume_min)
        print('volumes mean: ' + str(np.mean(volumes)))
        print('volumes max: ' + str(np.max(volumes)))
        print(number_of_requests)
        for j in range(number_of_requests):
            arrival_time = requests_arrival_time[j]
            request_index = j + number_of_requests_start_index + add_index
            volume = volumes[j]
            request = generate_fix_rate(snm_lifespans[j], volume / snm_lifespan, volume)
            for k in range(len(request)):
                if request[k] + arrival_time < sim_days:
                    requests.append((request[k] + arrival_time, request_index, request_sizes[j], i))
        add_index = add_index + number_of_requests
    return requests


if __name__ == '__main__':
    req = generate_snm_requests(10, 3)
    print('done')
