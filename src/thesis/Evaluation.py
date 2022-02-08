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
from Graphs import plot_one_stuff
from Graphs import plot_one_stuff_hr
from Graphs import plot_all_process_stuff
from Graphs import plot_deliver_cu_stuff
from Graphs import plot_all_stuff
from Graphs import plot_all_stuff_line
from Graphs import plot_all_redundant_stuff
from Graphs import plot_redundant_stuff
from Simulation import ThesisSimulation
from Simulation import PaperSimulation
from MLP import MultilayerPerceptron
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
    print((len(np.where(data[:, 3] == 'bs')[0]) / len(data)) * 100)
    print((len(np.where(data[:, 3] == 'cu')[0]) / len(data)) * 100)
    return (1 - len(np.where(np.logical_or(data[:, 3] == 'os', data[:, 3] == '!cu'))[0]) / len(data)) * 100


def calc_delay(data):
    number_of_cu = len(np.where(data[:, 3] == 'cu')[0])
    number_of_bs_ = len(np.where(data[:, 3] == 'bs')[0])
    # return (number_of_bs_ + number_of_cu * 5) / (number_of_cu + number_of_bs_)
    # return (number_of_bs_ + number_of_cu * 5) / len(data) * 100 / calc_hitRate(data)
    return np.sum(data[:, 2].astype(int)) / len(data)


def calc_cu(data):
    return len(np.where((data[:, 3] == 'cu'))[0]) / len(data) * 100


def calc_redundunt_irm(data):
    return len(np.where(data[:, 3] == '!cu')[0])


def check_fraction(req_num, train_window, tot_observation_window, bs_count, total_cache, cache_fraction, sim_req):
    thesis_sim = ThesisSimulation(req_num, train_window, tot_observation_window, bs_count, total_cache, cache_fraction,
                                  False)

    res = []
    for f in np.arange(0, 20.5, 2):
        thesis_sim.restart_net(f)
        thesis_delivery = thesis_sim.sim(sim_req.copy())
        hit_rate = calc_hitRate(thesis_delivery)
        res.append((f, hit_rate))
        print('total HitRate thesis ' + str(hit_rate) + ' fraction: ' + str(f))

    plot_2d_array(np.array(res), 'fraction', 'hitRate', 'thesis fraction hitRate')


def check_fraction_delay():
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    tot_cache_size = cfg.total_cache_size * number_of_bs
    tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    sim_requests = np.ndarray.tolist(sim_requests)
    irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
    thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, number_of_bs, tot_cache_size,
                                  fraction, None)
    res_delay = []
    res_hitRate = []
    index = 0
    for f in [0, 0.5, 1, 2, 3, 4, 6, 8, 12, 20, 32, 1000]:
        thesis_sim.restart_net(f)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())
        delay = calc_delay(thesis_delivery)
        hit_rate = calc_hitRate(thesis_delivery)
        res_delay.append((index, delay))
        res_hitRate.append((index, hit_rate))
        index = index + 1
        print('mean Delay ' + str(delay) + ' fraction: ' + str(f))
        print('total HitRate ' + str(hit_rate) + ' fraction: ' + str(f))

    plot_2d_array(np.array(res_hitRate), 'fraction', 'hitRate', 'fraction hitRate')
    plot_2d_array(np.array(res_delay), 'fraction', 'expected delay', 'fraction expected delay')


def check_fraction_delay_all():
    thesis, hit_rates, titles = np.zeros(5), np.zeros(5), []
    tot_cache_size = cfg.total_cache_size * number_of_bs
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    sim_requests = np.ndarray.tolist(sim_requests)
    irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
    ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

    thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, number_of_bs, tot_cache_size,
                                  fraction, None)
    thesis_delivery = thesis_sim.sim(sim_requests.copy())

    paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, number_of_bs, tot_cache_size, 0, None)
    paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

    paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, number_of_bs, tot_cache_size, 1, None)
    paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

    paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, number_of_bs, tot_cache_size, 2, None)
    paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

    paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, number_of_bs, tot_cache_size, 3, None)
    paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

    thesis[0] = calc_delay(thesis_delivery)
    hit_rates[0] = calc_hitRate(thesis_delivery)
    titles.append('Thesis')
    thesis[1] = calc_delay(paper_share_0_delivery)
    hit_rates[1] = calc_hitRate(paper_share_0_delivery)
    titles.append('Sharing 0%')
    thesis[2] = calc_delay(paper_share_1_delivery)
    hit_rates[2] = calc_hitRate(paper_share_1_delivery)
    titles.append('Sharing 33%')
    thesis[3] = calc_delay(paper_share_2_delivery)
    hit_rates[3] = calc_hitRate(paper_share_2_delivery)
    titles.append('Sharing 66%')
    thesis[4] = calc_delay(paper_share_3_delivery)
    hit_rates[4] = calc_hitRate(paper_share_3_delivery)
    titles.append('Sharing 100%')
    print('thesis ' + str(thesis[0]))
    print('paper sharing 0% ' + str(thesis[1]))
    print('paper sharing 33% ' + str(thesis[2]))
    print('paper sharing 66% ' + str(thesis[3]))
    print('paper sharing 100% ' + str(thesis[4]))
    plot_one_stuff("Average Delay of requests", "Hit Rate", thesis, hit_rates, titles)
    # plot_one_stuff_hr("Hit Rate", hit_rates, titles)


def cu_ratio_up():
    thesis, hit_rates, titles = np.zeros(2), np.zeros(2), []
    # tot_cache_size = cfg.total_cache_size * number_of_bs
    # snm_arrival_rate = cfg.snm_new_arrival_rate
    # irm_arrival_rate = cfg.irm_arrival_rate
    # tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
    # tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    # train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    # total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    # fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    # irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    # sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    # sim_requests = np.ndarray.tolist(sim_requests)
    # irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
    # ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')
    #
    # thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, number_of_bs, tot_cache_size,
    #                               fraction, None)
    # thesis_delivery = thesis_sim.sim(sim_requests.copy())
    #
    # paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
    #                                     total_observation_window, number_of_bs, tot_cache_size, 3, None)
    # paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

    # thesis[0] = calc_cu(thesis_delivery)
    thesis[0] = 10.5
    # hit_rates[0] = calc_hitRate(thesis_delivery)
    hit_rates[0] = 21
    titles.append('Thesis')
    # thesis[1] = calc_cu(paper_share_3_delivery)
    thesis[1] = 14.8
    # hit_rates[1] = calc_hitRate(paper_share_3_delivery)
    hit_rates[1] = 21
    titles.append('Sharing 100%')
    print('thesis ' + str(thesis[0]))
    print('paper sharing 100% ' + str(thesis[1]))
    plot_deliver_cu_stuff(thesis, hit_rates)
    # plot_one_stuff_hr("Hit Rate", hit_rates, titles)


def sort_list(item):
    return item[0]


def load_up():
    thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles = \
        np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), []
    tot_cache_size = cfg.total_cache_size * number_of_bs
    tot_requests_backup = []

    index = 0
    for i in range(1, 6):
        snm_arrival_rate = cfg.snm_new_arrival_rate * i
        irm_arrival_rate = cfg.irm_arrival_rate * i
        tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
        tot_requests.sort(key=sort_list)
        # tot_requests_backup = tot_requests.copy()
        tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
        print('tot_req_count' + ' ' + str(len(tot_requests)))
        train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
        total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
        fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
        irm_window = turn_observation_into_irm(total_observation_window, t_observation)
        sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
        sim_requests = np.ndarray.tolist(sim_requests)
        irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
        ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

        thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, number_of_bs, tot_cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())

        paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, tot_cache_size, 0, None)
        paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

        paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, tot_cache_size, 1, None)
        paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

        paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, tot_cache_size, 2, None)
        paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

        paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, tot_cache_size, 3, None)
        paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_share_0[index] = calc_hitRate(paper_share_0_delivery)
        paper_share_1[index] = calc_hitRate(paper_share_1_delivery)
        paper_share_2[index] = calc_hitRate(paper_share_2_delivery)
        paper_share_3[index] = calc_hitRate(paper_share_3_delivery)
        # titles.append('irm_rate: ' + str(irm_arrival_rate) + ' snm_rate: ' + str(snm_arrival_rate))
        titles.append(str(i))
        print('thesis ' + str(thesis[index]))
        print('paper sharing 0% ' + str(paper_share_0[index]))
        print('paper sharing 33% ' + str(paper_share_1[index]))
        print('paper sharing 66% ' + str(paper_share_2[index]))
        print('paper sharing 100% ' + str(paper_share_3[index]))

        index = index + 1

    plot_all_stuff_line("Increasing Load of requests (both SNM and IRM)",
                        thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles)


def size_up():
    thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles = \
        np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    sim_requests = np.ndarray.tolist(sim_requests)
    irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
    ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

    index = 0
    for i in range(1, 6):
        cache_size = cfg.total_cache_size * i * number_of_bs
        thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, number_of_bs, cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())

        paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 0, None)
        paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

        paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 1, None)
        paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

        paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 2, None)
        paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

        paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 3, None)
        paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_share_0[index] = calc_hitRate(paper_share_0_delivery)
        paper_share_1[index] = calc_hitRate(paper_share_1_delivery)
        paper_share_2[index] = calc_hitRate(paper_share_2_delivery)
        paper_share_3[index] = calc_hitRate(paper_share_3_delivery)
        titles.append('BS cache size: ' + str(cfg.total_cache_size * i))
        print('thesis ' + str(thesis[index]))
        print('paper sharing 0% ' + str(paper_share_0[index]))
        print('paper sharing 33% ' + str(paper_share_1[index]))
        print('paper sharing 66% ' + str(paper_share_2[index]))
        print('paper sharing 100% ' + str(paper_share_3[index]))

        index = index + 1

    plot_all_stuff("Increasing size of each BS's cache",
                   thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles)


def bs_up():
    thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles = \
        np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), []
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
        irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
        ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

        thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, bs_count, cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())

        paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, bs_count, cache_size, 0, None)
        paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

        paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, bs_count, cache_size, 1, None)
        paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

        paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, bs_count, cache_size, 2, None)
        paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

        paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, bs_count, cache_size, 3, None)
        paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_share_0[index] = calc_hitRate(paper_share_0_delivery)
        paper_share_1[index] = calc_hitRate(paper_share_1_delivery)
        paper_share_2[index] = calc_hitRate(paper_share_2_delivery)
        paper_share_3[index] = calc_hitRate(paper_share_3_delivery)
        titles.append('#BS: ' + str(bs_count))
        print('thesis ' + str(thesis[index]))
        print('paper sharing 0% ' + str(paper_share_0[index]))
        print('paper sharing 33% ' + str(paper_share_1[index]))
        print('paper sharing 66% ' + str(paper_share_2[index]))
        print('paper sharing 100% ' + str(paper_share_3[index]))

        index = index + 1

    plot_all_stuff("Increasing number of BSs",
                   thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles)


def irm_snm_ratio_up():
    thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles = \
        np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    cache_size = cfg.total_cache_size * number_of_bs
    irm_snm_list = [(0.25, 1.75), (0.5, 1.5), (0.75, 1.25), (1, 1), (1.25, 0.75), (1.5, 0.5), (1.75, 0.25)]

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
        irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
        ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

        thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, number_of_bs, cache_size,
                                      fraction, None)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())

        paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 0, None)
        paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

        paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 1, None)
        paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

        paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 2, None)
        paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

        paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 3, None)
        paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_share_0[index] = calc_hitRate(paper_share_0_delivery)
        paper_share_1[index] = calc_hitRate(paper_share_1_delivery)
        paper_share_2[index] = calc_hitRate(paper_share_2_delivery)
        paper_share_3[index] = calc_hitRate(paper_share_3_delivery)
        titles.append('IRM ' + str(round(irm_snm[0] / 2 * 100))
                      + '% SNM ' + str(round(irm_snm[1] / 2 * 100)) + '%')
        print('thesis ' + str(thesis[index]))
        print('paper sharing 0% ' + str(paper_share_0[index]))
        print('paper sharing 33% ' + str(paper_share_1[index]))
        print('paper sharing 66% ' + str(paper_share_2[index]))
        print('paper sharing 100% ' + str(paper_share_3[index]))

        index = index + 1

    # plot_all_stuff("Set different ratio for SNM and IRM contents",
    #                thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles)
    plot_all_stuff_line("Set different ratio for SNM and IRM contents",
                        thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles)


def bs_ratio_up():
    thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles = \
        np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    cache_size = cfg.total_cache_size * number_of_bs
    tot_requests = generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, None)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    train_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    sim_requests = np.ndarray.tolist(sim_requests)
    irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
    ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

    index = 0
    for i in range(1, 6):
        bs_sizes = []
        bs_sizes_paper = []
        t_cache_size = round(cache_size / (number_of_bs + cfg.fraction_thesis) * number_of_bs)
        for j in range(number_of_bs - 1):
            size = randint(1, t_cache_size - sum(bs_sizes) - number_of_bs + j)
            bs_sizes.append(size)
            bs_sizes_paper.append(size)
        bs_sizes.append(t_cache_size - sum(bs_sizes))
        bs_sizes_paper.append(t_cache_size - sum(bs_sizes_paper))
        print(bs_sizes)
        added_amount = cache_size - t_cache_size
        print(added_amount)
        added = 0
        c = 0
        while added < added_amount:
            if c >= len(bs_sizes):
                c = 0
            bs_sizes_paper[c] = bs_sizes_paper[c] + 1
            c = c + 1
            added = added + 1
        print(bs_sizes_paper)

        thesis_sim = ThesisSimulation(tot_req_count, irm_mlp, total_observation_window, number_of_bs, cache_size,
                                      fraction, bs_sizes)
        thesis_delivery = thesis_sim.sim(sim_requests.copy())

        paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 0, bs_sizes_paper)
        paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

        paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 1, bs_sizes_paper)
        paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

        paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 2, bs_sizes_paper)
        paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

        paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, number_of_bs, cache_size, 3, bs_sizes_paper)
        paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

        thesis[index] = calc_hitRate(thesis_delivery)
        paper_share_0[index] = calc_hitRate(paper_share_0_delivery)
        paper_share_1[index] = calc_hitRate(paper_share_1_delivery)
        paper_share_2[index] = calc_hitRate(paper_share_2_delivery)
        paper_share_3[index] = calc_hitRate(paper_share_3_delivery)
        # titles.append('thesis BS sizes: ' + str(bs_sizes)[1:-1] + 'paper BS sizes: ' + str(bs_sizes_paper)[1:-1])
        # titles.append('thesis BS sizes: ' + str(bs_sizes)[1:-1])
        titles.append(str(bs_sizes)[1:-1])
        print('thesis ' + str(thesis[index]))
        print('paper sharing 0% ' + str(paper_share_0[index]))
        print('paper sharing 33% ' + str(paper_share_1[index]))
        print('paper sharing 66% ' + str(paper_share_2[index]))
        print('paper sharing 100% ' + str(paper_share_3[index]))

        index = index + 1

    plot_all_stuff("Set different sizes for each BS's cache",
                   thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles)


def irm_redundancy_bs_up():
    paper_share_0, paper_share_1, paper_share_2, titles = np.zeros(5), np.zeros(5), np.zeros(5), []
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate

    index = 0
    for i in range(1, 6):
        bs_count = cfg.bs_count + index
        cache_size = cfg.total_cache_size * bs_count
        tot_requests = generate_sim_requests(sim_days, bs_count, snm_arrival_rate, irm_arrival_rate, None)
        tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
        train_requests = np.array(tot_requests)[
            np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
        total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
        fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
        irm_window = turn_observation_into_irm(total_observation_window, t_observation)
        sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
        sim_requests = np.ndarray.tolist(sim_requests)
        irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
        ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

        paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, bs_count, cache_size, 0, None)
        paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

        paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, bs_count, cache_size, 1, None)
        paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

        paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                            total_observation_window, bs_count, cache_size, 2, None)
        paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

        paper_share_0[index] = calc_redundunt_irm(paper_share_0_delivery)
        paper_share_1[index] = calc_redundunt_irm(paper_share_1_delivery)
        paper_share_2[index] = calc_redundunt_irm(paper_share_2_delivery)
        titles.append('number of BSs: ' + str(bs_count))
        print('paper sharing 0% ' + str(paper_share_0[index]))
        print('paper sharing 33% ' + str(paper_share_1[index]))
        print('paper sharing 66% ' + str(paper_share_2[index]))

        index = index + 1

    plot_redundant_stuff("IRM redundant cached requests by Increasing the number of BSs",
                         paper_share_0, paper_share_1, paper_share_2, titles)


def irm_redundancy_days_graph():
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    bs_count = cfg.bs_count
    cache_size = cfg.total_cache_size * bs_count
    tot_requests = generate_sim_requests(sim_days, bs_count, snm_arrival_rate, irm_arrival_rate, None)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    train_requests = np.array(tot_requests)[
        np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    sim_requests = np.ndarray.tolist(sim_requests)
    irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
    ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

    paper_share_0_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, bs_count, cache_size, 0, None)
    paper_share_0_delivery = paper_share_0_sim.sim(sim_requests.copy())

    paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, bs_count, cache_size, 1, None)
    paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

    paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, bs_count, cache_size, 2, None)
    paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

    print('paper sharing 0% ' + str(calc_hitRate(paper_share_0_delivery)))
    print('paper sharing 33% ' + str(calc_hitRate(paper_share_1_delivery)))
    print('paper sharing 66% ' + str(calc_hitRate(paper_share_2_delivery)))

    plot_all_redundant_stuff(paper_share_0_delivery, paper_share_1_delivery, paper_share_2_delivery)


def process_days_graph():
    snm_arrival_rate = cfg.snm_new_arrival_rate
    irm_arrival_rate = cfg.irm_arrival_rate
    bs_count = cfg.bs_count
    cache_size = cfg.total_cache_size * bs_count
    tot_requests = generate_sim_requests(sim_days, bs_count, snm_arrival_rate, irm_arrival_rate, None)
    tot_req_count = int(np.max(np.array(tot_requests)[:, 1])) + 1
    train_requests = np.array(tot_requests)[
        np.where((np.array(tot_requests)[:, 0]).astype(float) <= train_days)]
    total_observation_window = turn_requests_into_observation_window(tot_req_count, train_requests, train_days)
    fraction_window = turn_observation_into_w_s(total_observation_window, t_observation, train_days)
    irm_window = turn_observation_into_irm(total_observation_window, t_observation)
    sim_requests = np.array(tot_requests)[np.where((np.array(tot_requests)[:, 0]).astype(float) > train_days)]
    sim_requests = np.ndarray.tolist(sim_requests)
    irm_mlp = MultilayerPerceptron(irm_window, False, 'irm')
    ws_mlp = MultilayerPerceptron(fraction_window, False, 'w_s')

    paper_share_1_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, bs_count, cache_size, 1, None)
    paper_share_1_delivery = paper_share_1_sim.sim(sim_requests.copy())

    paper_share_2_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, bs_count, cache_size, 2, None)
    paper_share_2_delivery = paper_share_2_sim.sim(sim_requests.copy())

    paper_share_3_sim = PaperSimulation(tot_req_count, irm_mlp, ws_mlp, fraction_window,
                                        total_observation_window, bs_count, cache_size, 3, None)
    paper_share_3_delivery = paper_share_3_sim.sim(sim_requests.copy())

    print('paper sharing 33% ' + str(calc_hitRate(paper_share_1_delivery)))
    print('paper sharing 66% ' + str(calc_hitRate(paper_share_2_delivery)))
    print('paper sharing 100% ' + str(calc_hitRate(paper_share_3_delivery)))

    plot_all_process_stuff(paper_share_1_delivery, paper_share_2_delivery, paper_share_3_delivery)


if __name__ == '__main__':
    # check_fraction_delay()
    # check_fraction_delay_all()
    # cu_ratio_up()
    # process_days_graph()

    # plot_stuff(thesis_delivery, paper_delivery)
    # plot_stuff_(thesis_delivery)

    load_up()
    # size_up()
    # bs_up()
    # irm_snm_ratio_up()
    # bs_ratio_up()
    # irm_redundancy_bs_up()
    # irm_redundancy_days_graph()

    print('done')
