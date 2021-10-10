import numpy as np
import config as cfg
from IRM import generate_irm_requests
from SNM import generate_snm_requests


def generate_sim_requests(sim_days, number_of_bs, snm_arrival_rate, irm_arrival_rate, irm_snm):
    # (ts, f, size, bs)
    irm_rate = irm_arrival_rate * number_of_bs
    if irm_snm is not None:
        irm_rate = irm_snm[0] * irm_rate
        snm_arrival_rate = irm_snm[1] * snm_arrival_rate
    irm = generate_irm_requests(sim_days, irm_rate)
    requests_list = generate_snm_requests(sim_days, number_of_bs, snm_arrival_rate)
    bs = 0
    for i in irm:
        requests_list.append((i[1], i[0], i[2], bs))
        bs = bs + 1
        if bs == number_of_bs:
            bs = 0
    return requests_list
