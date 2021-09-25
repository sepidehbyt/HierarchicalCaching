import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

sim_days = cfg.sys_simulation_days
train_days = cfg.sys_train_generation_days


def plot_stuff(thesis_deliveries, paper_deliveries):
    cu_t, bs_t, os_t, irm_t, snm_t, tot_t = plot_each(thesis_deliveries)
    cu_p, bs_p, os_p, irm_p, snm_p, tot_p = plot_each(paper_deliveries)
    plot_both_delivery(cu_t, bs_t, os_t, cu_p, bs_p, os_p)
    plot_both_hit_rate(irm_t, snm_t, tot_t, irm_p, snm_p, tot_p)


def plot_stuff_(thesis_deliveries):
    cu_t, bs_t, os_t, irm_t, snm_t, tot_t = plot_each(thesis_deliveries)


def plot_both_delivery(cu_t, bs_t, os_t, cu_p, bs_p, os_p):
    # x_axis = np.ones([i for i in range(train_days, sim_days)])
    plt.plot(cu_t, color='green', marker="^", label='cu thesis')
    plt.plot(cu_p, color='green', marker="v", label='cu paper')
    plt.plot(bs_t, color='red', marker="^", label='bs thesis')
    plt.plot(bs_p, color='red', marker="v", label='bs paper')
    plt.plot(os_t, color='blue', marker="^", label='os thesis')
    plt.plot(os_p, color='blue', marker="v", label='os paper')

    plt.legend(shadow=True, loc="lower right")

    plt.title("delivery from destination:")
    plt.xlabel("days")
    plt.ylabel("#requests")
    plt.show()


def plot_both_hit_rate(irm_t, snm_t, tot_t, irm_p, snm_p, tot_p):
    # x_axis = np.ones([i for i in range(train_days, sim_days)])
    plt.plot(irm_t, color='green', marker="^", label='irm thesis')
    plt.plot(irm_p, color='green', marker="v", label='irm paper')
    plt.plot(snm_t, color='red', marker="^", label='snm thesis')
    plt.plot(snm_p, color='red', marker="v", label='snm paper')
    plt.plot(tot_t, color='blue', marker="^", label='total thesis')
    plt.plot(tot_p, color='blue', marker="v", label='total paper')

    plt.legend(shadow=True, loc="lower right")

    plt.title("Hit Rate")
    plt.xlabel("days")
    plt.ylabel("#requests")
    plt.show()


def plot_each(deliveries):
    snm = np.zeros(sim_days - train_days)
    irm = np.zeros(sim_days - train_days)
    tot = np.zeros(sim_days - train_days)
    bs = np.zeros(sim_days - train_days)
    cu = np.zeros(sim_days - train_days)
    os = np.zeros(sim_days - train_days)
    for i in range(train_days, sim_days):
        bs[i - train_days] = len(np.where(np.logical_and(deliveries[:, 3] == 'bs',
                                                         np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                                        (deliveries[:, 1]).astype(float) < i + 1)))[0])
        cu[i - train_days] = len(np.where(np.logical_and(deliveries[:, 3] == 'cu',
                                                         np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                                        (deliveries[:, 1]).astype(float) < i + 1)))[0])
        os[i - train_days] = len(np.where(np.logical_and(deliveries[:, 3] == 'os',
                                                         np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                                        (deliveries[:, 1]).astype(float) < i + 1)))[0])
        all_irm = len(np.where(np.logical_and((deliveries[:, 0]).astype(float) < 5000.0,
                                              np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                             (deliveries[:, 1]).astype(float) < i + 1)))[0])
        irm[i - train_days] = len(np.where(np.logical_and(deliveries[:, 3] == 'os',
                                             np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                            np.logical_and((deliveries[:, 1]).astype(float) < i + 1,
                                                                           (deliveries[:, 0]).astype(float) < 5000.0))))[
                         0]) / all_irm
        all_snm = len(np.where(np.logical_and((deliveries[:, 0]).astype(float) >= 5000.0,
                                              np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                             (deliveries[:, 1]).astype(float) < i + 1)))[0])

        snm[i - train_days] = len(np.where(np.logical_and(deliveries[:, 3] == 'os',
                                             np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                            np.logical_and((deliveries[:, 1]).astype(float) < i + 1,
                                                                           (deliveries[:, 0]).astype(float) >= 5000.0))))[
                         0]) / all_snm
        all_req = len(np.where(np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                              (deliveries[:, 1]).astype(float) < i + 1))[0])
        tot[i - train_days] = os[i - train_days] / all_req

    tot_hit_rate = (np.sum(1-tot) / (sim_days - train_days))
    print('total HitRate: ', tot_hit_rate)
    irm = 1 - irm
    snm = 1 - snm
    tot = 1 - tot
    plotdata = pd.DataFrame({
        "IRM": irm,
        "SNM": snm,
        "Total": tot},
        index=list(str('day' + str(i)) for i in range(train_days, sim_days)))
    plotdata.plot(kind="bar")
    plt.title("Hit rate for SNM/IRM contents with total: " + str(tot_hit_rate))
    plt.xlabel("Hit Rate")
    plt.ylabel("Content Type")
    plt.show()
    plotdata = pd.DataFrame({
        "CloudUnit": cu,
        "BaseStation": bs,
        "OriginalServer": os},
        index=list(str('day' + str(i)) for i in range(train_days, sim_days)))
    plotdata.plot(kind="bar")
    plt.title("Deliveries Count per Station")
    plt.xlabel("#Requests")
    plt.ylabel("Delivery Type")
    plt.show()

    return cu, bs, os, irm, snm, tot
