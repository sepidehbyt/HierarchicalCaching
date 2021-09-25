import csv
import numpy as np
import math
import config as cfg
from GenerateRequests import generate_sim_requests
from IRM import generate_number_of_irm_train
# from MLP import train_mlp_type

rth_thesis = cfg.rth_thesis
rth_paper = cfg.rth_paper
snm_window = cfg.train_min_window_threshold


def turn_requests_into_observation_window(req_num, requests, sim_days):
    observation = np.zeros((req_num, sim_days))
    for i in range(sim_days):
        observe_indexes = np.where(np.logical_and((requests[:, 0]).astype(float) >= i, (requests[:, 0]).astype(float) < i + 1))[0]
        for j in observe_indexes:
            index = int(requests[j][1])
            observation[index][i] = observation[index][i] + 1
    return observation


def turn_observation_into_irm(observation, t_observation):
    data = []
    for observe in observation:
        for i in range(len(observe) - t_observation):
            if np.sum(observe[i:i+t_observation]) >= rth_thesis \
                    and len(np.where(observe[i:i + t_observation] != 0)[0]) >= snm_window:
                data.append((observe[i:i+t_observation + 1]).astype(int))
    gen_train_data_type(t_observation, np.array(data), 'irm')
    return np.array(data)


def turn_observation_into_w_s(observation, t_observation, sim_days):
    w_s_all = np.zeros(sim_days - t_observation)
    w_s_window = np.zeros((sim_days - 2 * t_observation, t_observation + 1))
    for i in range(sim_days - t_observation):
        w_s_all[i] = calculate_w_s(observation[:, i:i + t_observation + 1], t_observation)
    for i in range(sim_days - 2 * t_observation):
        for j in range(t_observation + 1):
            w_s_window[i][j] = w_s_all[j + i]
    gen_train_data_type(t_observation, w_s_window, 'w_s')
    return w_s_window


def calculate_w_s(calc_window_i, t_observation):
    observation_window_i = calc_window_i[:, :t_observation]
    calc_window = calc_window_i[:, t_observation]
    contents = np.where(calc_window.astype(int) >= rth_paper)[0]
    temp = 0
    for content in contents:
        if (np.where(observation_window_i[content] != 0)[0].shape[0]) == 0:
            temp = temp + calc_window[content]
    w_s = np.round(temp / (np.sum(calc_window[contents])), 2)
    return w_s


def gen_train_data():
    t_observation = cfg.sys_observation_window_days
    train_days = cfg.sys_simulation_days
    header = list(str('T' + str(i)) for i in range(t_observation + 1))

    with open('_train_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        irm = generate_number_of_irm_train(train_days, t_observation)

        for i in range(len(irm)):
            data = irm[i]
            writer.writerow(data)
    return irm


def gen_train_data_type(t, data, train_type):
    header = list(str('T' + str(i)) for i in range(t + 1))

    with open(train_type + '_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(data)):
            writer.writerow(data[i])


if __name__ == '__main__':
    # gen_train_data()
    sim = 15
    req = generate_sim_requests(sim, 2)
    req = np.array(req)
    observation_window = turn_requests_into_observation_window(req, sim)
    fraction_window = turn_observation_into_w_s(observation_window, 5, sim)
    irm_window = turn_observation_into_irm(observation_window, 5)
    # train_mlp_type(irm_window, False, 'irm')
    # train_mlp_type(fraction_window, False, 'w_s')
    print('done')
