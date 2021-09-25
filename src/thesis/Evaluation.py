import numpy as np
import math
import random
import config as cfg
from GenerateRequests import generate_sim_requests
from GenerateTrainData import turn_requests_into_observation_window
from GenerateTrainData import turn_observation_into_w_s
from GenerateTrainData import turn_observation_into_irm
from Graphs import plot_stuff
from Graphs import plot_stuff_
from Simulation import ThesisSimulation
from Simulation import PaperSimulation

sim_days = cfg.sys_simulation_days
train_days = cfg.sys_train_generation_days
number_of_bs = cfg.bs_count
t_observation = cfg.sys_observation_window_days
t_update = cfg.sys_update_window_days
bs_dr = cfg.bs_data_rate
cu_dr = cfg.cu_data_rate
os_dr = cfg.os_data_rate
thesis_bs_cache_size = cfg.thesis_bs_cache_size
paper_bs_cache_size = cfg.paper_bs_cache_size
tot_cache_size = cfg.total_cache_size
fraction = cfg.fraction_thesis


def calc_hitRate(data):
    return (1 - len(np.where(data[:, 3] == 'os')[0]) / len(data)) * 100


if __name__ == '__main__':

    tot_requests = generate_sim_requests(sim_days, number_of_bs)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    sim_requests = np.ndarray.tolist(sim_requests)

    thesis_sim = ThesisSimulation(tot_req_count, irm_window, total_observation_window, number_of_bs, tot_cache_size, fraction)
    thesis_delivery = thesis_sim.sim(sim_requests.copy())
    # thesis_delivery = ThesisSimulation(sim_requests.copy(), tot_req_count, irm_window, total_observation_window, bs)
    # paper_delivery = PaperSimulation(sim_requests.copy(), tot_req_count, irm_window, fraction_window, total_observation_window)

    # plot_stuff(thesis_delivery, paper_delivery)

    print('total HitRate thesis ' + str(calc_hitRate(thesis_delivery)))
    # print('total HitRate paper ' + str(calc_hitRate(paper_delivery)))

    plot_stuff_(thesis_delivery)
    # print('done')
