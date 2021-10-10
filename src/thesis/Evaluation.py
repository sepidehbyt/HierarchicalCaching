import numpy as np
import math
import random
import config as cfg
from GenerateRequests import generate_sim_requests
from GenerateTrainData import turn_requests_into_observation_window
from GenerateTrainData import turn_observation_into_w_s
from GenerateTrainData import turn_observation_into_irm
from Graphs import plot_stuff
from Graphs import plot_2d_array
from Graphs import plot_stuff_
from Graphs import plot_all_stuff
from Simulation import ThesisSimulation
from Simulation import PaperSimulation
from random import randint


sim_days = cfg.sys_simulation_days
train_days = cfg.sys_train_generation_days
number_of_bs = cfg.bs_count
t_observation = cfg.sys_observation_window_days
t_update = cfg.sys_update_window_days
bs_dr = cfg.bs_data_rate
cu_dr = cfg.cu_data_rate
os_dr = cfg.os_data_rate
fraction = cfg.fraction_thesis


def calc_hitRate(data):
    return (1 - len(np.where(data[:, 3] == 'os')[0]) / len(data)) * 100


def check_fraction(req_num, train_window, tot_observation_window, bs_count, total_cache, cache_fraction, sim_req):
    thesis_sim = ThesisSimulation(req_num, train_window, tot_observation_window, bs_count, total_cache, cache_fraction, False)

    res = []
    for f in np.arange(0, 10.5, 1):
        thesis_sim.restart_net(f)
        thesis_delivery = thesis_sim.sim(sim_req.copy())
        hit_rate = calc_hitRate(thesis_delivery)
        res.append((f, hit_rate))
        print('total HitRate thesis ' + str(hit_rate) + ' fraction: ' + str(f))

    plot_2d_array(np.array(res), 'fraction', 'hitRate', 'thesis fraction hitRate')


def load_up():
    thesis, paper_best, paper_worst, titles = np.zeros(5), np.zeros(5), np.zeros(5), []
    tot_cache_size = cfg.total_cache_size * number_of_bs

    index = 0
    for i in range(1, 6):
        snm_arrival_rate = cfg.snm_new_arrival_rate * i
        irm_arrival_rate = cfg.irm_arrival_rate * i
        tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
        tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
        train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
        total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
        fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
        irm_window = turn_observation_into_irm(total_observation_window, t_observation)
        sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
        sim_requests = np.ndarray.tolist(sim_requests)

        thesis_sim = ThesisSimulation(tot_req_count, irm_window, total_observation_window, number_of_bs, tot_cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())
        paper_best_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                         number_of_bs,
                                         tot_cache_size, False, None)
        paper_best_delivery = paper_best_sim.sim(sim_requests.copy())
        paper_worst_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                          number_of_bs, tot_cache_size, True, None)
        paper_worst_delivery = paper_worst_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_best[index] = calc_hitRate(paper_best_delivery)
        paper_worst[index] = calc_hitRate(paper_worst_delivery)
        titles.append(str(irm_arrival_rate) + ' snm_rate: ' + str(snm_arrival_rate))
        print('total HitRate thesis ' + str(thesis[index]))
        print('total HitRate paper best ' + str(paper_best[index]))
        print('total HitRate paper worst ' + str(paper_worst[index]))

        index = index + 1

    plot_all_stuff(thesis, paper_best, paper_worst, titles)


def size_up():
    thesis, paper_best, paper_worst, titles = np.zeros(5), np.zeros(5), np.zeros(5), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate

    index = 0
    for i in range(1, 6):
        cache_size = cfg.total_cache_size * i * number_of_bs
        tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
        tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
        train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
        total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
        fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
        irm_window = turn_observation_into_irm(total_observation_window, t_observation)
        sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
        sim_requests = np.ndarray.tolist(sim_requests)

        thesis_sim = ThesisSimulation(tot_req_count, irm_window, total_observation_window, number_of_bs, cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())
        paper_best_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                         number_of_bs,
                                         cache_size, False, None)
        paper_best_delivery = paper_best_sim.sim(sim_requests.copy())
        paper_worst_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                          number_of_bs, cache_size, True, None)
        paper_worst_delivery = paper_worst_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_best[index] = calc_hitRate(paper_best_delivery)
        paper_worst[index] = calc_hitRate(paper_worst_delivery)
        titles.append('every BS cache size: ' + str(cfg.total_cache_size * i))
        print('total HitRate thesis ' + str(thesis[index]))
        print('total HitRate paper best ' + str(paper_best[index]))
        print('total HitRate paper worst ' + str(paper_worst[index]))

        index = index + 1

    plot_all_stuff(thesis, paper_best, paper_worst, titles)


def bs_up():
    thesis, paper_best, paper_worst, titles = np.zeros(5), np.zeros(5), np.zeros(5), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate

    index = 0
    for i in range(1, 6):
        bs_count = cfg.bs_count + index
        cache_size = cfg.total_cache_size * bs_count
        tot_requests = generate_sim_requests(sim_days, bs_count, snm_arrival_rate, irm_arrival_rate, None)
        tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
        train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
        total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
        fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
        irm_window = turn_observation_into_irm(total_observation_window, t_observation)
        sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
        sim_requests = np.ndarray.tolist(sim_requests)

        thesis_sim = ThesisSimulation(tot_req_count, irm_window, total_observation_window, bs_count, cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())
        paper_best_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                         bs_count,
                                         cache_size, False, None)
        paper_best_delivery = paper_best_sim.sim(sim_requests.copy())
        paper_worst_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                          bs_count, cache_size, True, None)
        paper_worst_delivery = paper_worst_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_best[index] = calc_hitRate(paper_best_delivery)
        paper_worst[index] = calc_hitRate(paper_worst_delivery)
        titles.append('numer of BSs: ' + str(bs_count))
        print('total HitRate thesis ' + str(thesis[index]))
        print('total HitRate paper best ' + str(paper_best[index]))
        print('total HitRate paper worst ' + str(paper_worst[index]))

        index = index + 1

    plot_all_stuff(thesis, paper_best, paper_worst, titles)


def irm_snm_ratio_up():
    thesis, paper_best, paper_worst, titles = np.zeros(5), np.zeros(5), np.zeros(5), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    cache_size = cfg.total_cache_size * number_of_bs
    irm_snm_list = [(1, 1), (0.6, 1.4), (0.4, 1.2), (0.2, 1.8), (1.8, 0.2), (1.4, 0.6), (1.2, 0.4)]

    index = 0
    for i in range(1, 8):
        irm_snm = irm_snm_list[index]
        tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, irm_snm)
        tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
        train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
        total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
        fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
        irm_window = turn_observation_into_irm(total_observation_window, t_observation)
        sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
        sim_requests = np.ndarray.tolist(sim_requests)

        thesis_sim = ThesisSimulation(tot_req_count, irm_window, total_observation_window, number_of_bs, cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())
        paper_best_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                         number_of_bs,
                                         cache_size, False, None)
        paper_best_delivery = paper_best_sim.sim(sim_requests.copy())
        paper_worst_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                          number_of_bs, cache_size, True, None)
        paper_worst_delivery = paper_worst_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_best[index] = calc_hitRate(paper_best_delivery)
        paper_worst[index] = calc_hitRate(paper_worst_delivery)
        titles.append('IRM ' + str(irm_snm[index][0]/2*100) + '% SNM ' + str(irm_snm[index][1]/2*100) + '%')
        print('total HitRate thesis ' + str(thesis[index]))
        print('total HitRate paper best ' + str(paper_best[index]))
        print('total HitRate paper worst ' + str(paper_worst[index]))

        index = index + 1

    plot_all_stuff(thesis, paper_best, paper_worst, titles)


def bs_ratio_up():
    thesis, paper_best, paper_worst, titles = np.zeros(5), np.zeros(5), np.zeros(5), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    cache_size = cfg.total_cache_size * number_of_bs

    index = 0
    for i in range(1, 6):
        bs_sizes = []
        for j in range(number_of_bs - 1):
            size = randint(1, cache_size - sum(bs_sizes) - number_of_bs + j)
            while size < cfg.total_cache_size / cfg.min_ratio_cache_size:
                size = randint(1, cache_size - sum(bs_sizes) - number_of_bs + j)
            bs_sizes.append(size)
        bs_sizes.append(cache_size - sum(bs_sizes))
        tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
        tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
        train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
        total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
        fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
        irm_window = turn_observation_into_irm(total_observation_window, t_observation)
        sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
        sim_requests = np.ndarray.tolist(sim_requests)

        thesis_sim = ThesisSimulation(tot_req_count, irm_window, total_observation_window, number_of_bs, cache_size,
                                      fraction, bs_sizes)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())
        paper_best_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                         number_of_bs,
                                         cache_size, False, bs_sizes)
        paper_best_delivery = paper_best_sim.sim(sim_requests.copy())
        paper_worst_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window,
                                          number_of_bs, cache_size, True, bs_sizes)
        paper_worst_delivery = paper_worst_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_best[index] = calc_hitRate(paper_best_delivery)
        paper_worst[index] = calc_hitRate(paper_worst_delivery)
        titles.append('bs sizes: ' + str(bs_sizes)[1:-1])
        print('total HitRate thesis ' + str(thesis[index]))
        print('total HitRate paper best ' + str(paper_best[index]))
        print('total HitRate paper worst ' + str(paper_worst[index]))

        index = index + 1

    plot_all_stuff(thesis, paper_best, paper_worst, titles)


if __name__ == '__main__':
    # tot_requests = generate_sim_requests(sim_days, number_of_bs)
    # tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    # train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    # total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    # fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    # irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    # sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    # sim_requests = np.ndarray.tolist(sim_requests)

    # check_fraction(tot_req_count, irm_window, total_observation_window, number_of_bs, tot_cache_size, fraction, sim_requests)

    # thesis_sim = ThesisSimulation(tot_req_count, irm_window, total_observation_window, number_of_bs, tot_cache_size, fraction, False)
    # thesis_delivery = thesis_sim.sim(sim_requests.copy())
    # paper_best_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window, number_of_bs, tot_cache_size, False, False)
    # paper_best_delivery = paper_best_sim.sim(sim_requests.copy())
    # paper_worst_sim = PaperSimulation(tot_req_count, irm_window, fraction_window, total_observation_window, number_of_bs, tot_cache_size, True, False)
    # paper_worst_delivery = paper_worst_sim.sim(sim_requests.copy())

    # print('total HitRate thesis ' + str(calc_hitRate(thesis_delivery)))
    # print('total HitRate paper best ' + str(calc_hitRate(paper_best_delivery)))
    # print('total HitRate paper worst ' + str(calc_hitRate(paper_worst_delivery)))

    # plot_stuff(thesis_delivery, paper_delivery)
    # plot_stuff_(thesis_delivery)

    # load_up()
    # size_up()
    # bs_up()
    # irm_snm_ratio_up()
    bs_ratio_up()

    print('done')
