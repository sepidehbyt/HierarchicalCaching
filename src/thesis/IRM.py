import csv
import numpy as np
import math
import sys
import random
import config as cfg
from PoissonProcess import generate_fix_rate

mean_size = cfg.irm_mean_content_size
number_of_requests = cfg.irm_number_contents
skewness = cfg.irm_skewness
arrival_rate = cfg.irm_arrival_rate
rth_thesis = cfg.rth_thesis


def calc_zipf_distribution():
    omega = 1 / np.sum(1 / pow(i + 1, skewness) for i in range(number_of_requests))
    return np.stack(omega / pow(i+1, skewness) for i in range(number_of_requests))


def generate_irm_requests(sim_days):
    proba_dist = calc_zipf_distribution()
    irm_rate = proba_dist * arrival_rate
    # request_sizes = np.random.poisson(mean_size, number_of_requests) + 1
    request_sizes = np.ones(number_of_requests)
    requests = []
    for i in range(number_of_requests):
        request = generate_fix_rate(sim_days, irm_rate[i], sys.maxsize)
        for j in range(len(request)):
            requests.append((i, request[j], request_sizes[i]))
    requests = np.array(requests)
    return requests[requests[:, 1].argsort()]


def generate_number_of_irm_fill(observation_window):
    proba_dist = calc_zipf_distribution()
    irm_rate = proba_dist * arrival_rate
    requests = []
    for i in range(number_of_requests):
        request = generate_fix_rate(observation_window, irm_rate[i], sys.maxsize)
        organized_request = []
        index = 0
        count = np.where(np.logical_and(request >= index, request < index + 1))
        while index < observation_window:
            index = index + 1
            organized_request.append(len(count[0]))
            count = np.where(np.logical_and(request >= index, request < index + 1))
        requests.append(organized_request)
    return np.array(requests)


def generate_number_of_irm_train(sim_days, observation_window):
    proba_dist = calc_zipf_distribution()
    irm_rate = proba_dist * arrival_rate
    requests = []
    for i in range(number_of_requests):
        request = generate_fix_rate(sim_days, irm_rate[i], sys.maxsize)
        organized_request = []
        index = 0
        count = np.where(np.logical_and(request >= index, request < index + 1))
        while index < sim_days:
            index = index + 1
            organized_request.append(len(count[0]))
            count = np.where(np.logical_and(request >= index, request < index + 1))
        pointer = 0
        while pointer + observation_window + 1 <= sim_days:
            organized_request_p = organized_request[pointer:pointer+observation_window+1]
            if np.sum(organized_request_p) >= rth_thesis:
                requests.append(organized_request_p)
            pointer = pointer + 1
    return np.array(requests)


if __name__ == '__main__':
    requests, request_sizes = generate_irm_requests(10)
    # requests = generate_number_of_irm_fill(5)
    print('done')
