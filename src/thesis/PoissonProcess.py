import random
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
import sys


def generate_fix_rate(time, _lambda, volume):

    _event_num = []
    _inter_event_times = []
    _event_times = []
    _event_times_result = []
    _event_time = 0
    i = 0

    while _event_time < time and i < volume:

        i = i + 1
        _event_num.append(i)

        n = random.random()

        _inter_event_time = -math.log(1.0 - n) / _lambda

        _inter_event_times.append(_inter_event_time)

        _event_time = _event_time + _inter_event_time
        _event_times.append(_event_time)

        if _event_time <= time:
            _event_times_result.append(_event_time)

        # print(str(i) + ',' + str(_inter_event_time) + ',' + str(_event_time))

    return np.array(_event_times_result)


if __name__ == '__main__':
    a, m = 1.5, 100.  # shape and mode
    V = np.round((np.random.pareto(a, 1) + 1) * m)[0]
    result = generate_fix_rate(1, 1 * V, V)
    print(V)
    print(result.shape)
