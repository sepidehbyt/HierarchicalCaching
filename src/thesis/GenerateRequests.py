import numpy as np
import config as cfg
from IRM import generate_irm_requests
from SNM import generate_snm_requests


def generate_sim_requests(sim_days, number_of_bs):
    # (ts, f, size, bs)
    irm = generate_irm_requests(sim_days)
    requests_list = generate_snm_requests(sim_days, number_of_bs)
    bs = 0
    for i in irm:
        requests_list.append((i[1], i[0], i[2], bs))
        bs = bs + 1
        if bs == number_of_bs:
            bs = 0
    return requests_list


if __name__ == '__main__':
    generate_sim_requests(40, 3)
