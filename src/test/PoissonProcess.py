import random
import math
import statistics
import matplotlib.pyplot as plt

_lambda = 5
_num_events = 100
_event_num = []
_inter_event_times = []
_event_times = []
_event_time = 0

print('EVENT_NUM,INTER_EVENT_T,EVENT_T')

for i in range(_num_events):
    _event_num.append(i)

    # Get a random probability value from the uniform distribution's PDF
    n = random.random()

    # Generate the inter-event time from the exponential distribution's CDF using the Inverse-CDF technique
    _inter_event_time = -math.log(1.0 - n) / _lambda
    _inter_event_times.append(_inter_event_time)

    # Add the inter-event time to the running sum to get the next absolute event time
    _event_time = _event_time + _inter_event_time
    _event_times.append(_event_time)

    # print it all out
    print(str(i) + ',' + str(_inter_event_time) + ',' + str(_event_time))

# plot the inter-event times
fig = plt.figure()
fig.suptitle('Times between consecutive events in a simulated Poisson process')
plot, = plt.plot(_event_num, _inter_event_times, 'bo-', label='Inter-event time')
plt.legend(handles=[plot])
plt.xlabel('Index of event')
plt.ylabel('Time')
plt.show()

# plot the absolute event times
fig = plt.figure()
fig.suptitle('Absolute times of consecutive events in a simulated Poisson process')
plot, = plt.plot(_event_num, _event_times, 'bo-', label='Absolute time of event')
plt.legend(handles=[plot])
plt.xlabel('Index of event')
plt.ylabel('Time')
plt.show()

_interval_nums = []
_num_events_in_interval = []
_interval_num = 1
_num_events = 0

print('INTERVAL_NUM,NUM_EVENTS')

for i in range(len(_event_times)):
    _event_time = _event_times[i]
    if _event_time <= _interval_num:
        _num_events += 1
    else:
        _interval_nums.append(_interval_num)
        _num_events_in_interval.append(_num_events)

        print(str(_interval_num) + ',' + str(_num_events))

        _interval_num += 1

        _num_events = 1

# print the mean number of events per unit time
print(statistics.mean(_num_events_in_interval))

# plot the number of events in consecutive intervals
fig = plt.figure()
fig.suptitle('Number of events occurring in consecutive intervals in a simulated Poisson process')
plt.bar(_interval_nums, _num_events_in_interval)
plt.xlabel('Index of interval')
plt.ylabel('Number of events')
plt.show()
