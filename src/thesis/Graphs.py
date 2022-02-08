import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

sim_days = cfg.sys_simulation_days
train_days = cfg.sys_train_generation_days


def get_title():
    return ' config snm_lifespan ' + str(cfg.snm_lifespan) + ' snm_rate ' + str(cfg.snm_new_arrival_rate) + \
           ' irm_rate' + str(cfg.irm_arrival_rate)


def plot_stuff(thesis_deliveries, paper_deliveries):
    cu_t, bs_t, os_t, irm_t, snm_t, tot_t = plot_each(thesis_deliveries)
    cu_p, bs_p, os_p, irm_p, snm_p, tot_p = plot_each(paper_deliveries)
    plot_both_delivery(cu_t, bs_t, os_t, cu_p, bs_p, os_p)
    plot_both_hit_rate(irm_t, snm_t, tot_t, irm_p, snm_p, tot_p)


def plot_stuff_(thesis_deliveries):
    cu_t, bs_t, os_t, irm_t, snm_t, tot_t = plot_each(thesis_deliveries)


def plot_2d_array(array, x_title, y_title, title):
    plt.plot(array[:, 0], array[:, 1], color='blue')
    plt.title(title + get_title())
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


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
    plt.ylabel("#Requests")
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

    plt.title("Total SOR")
    plt.xlabel("days")
    plt.ylabel("#Requests")
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
                                                                         np.logical_and(
                                                                             (deliveries[:, 1]).astype(float) < i + 1,
                                                                             (deliveries[:, 0]).astype(
                                                                                 float) < 5000.0))))[
                                      0]) / all_irm
        all_snm = len(np.where(np.logical_and((deliveries[:, 0]).astype(float) >= 5000.0,
                                              np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                             (deliveries[:, 1]).astype(float) < i + 1)))[0])

        snm[i - train_days] = len(np.where(np.logical_and(deliveries[:, 3] == 'os',
                                                          np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                                                         np.logical_and(
                                                                             (deliveries[:, 1]).astype(float) < i + 1,
                                                                             (deliveries[:, 0]).astype(
                                                                                 float) >= 5000.0))))[
                                      0]) / all_snm
        all_req = len(np.where(np.logical_and((deliveries[:, 1]).astype(float) >= i,
                                              (deliveries[:, 1]).astype(float) < i + 1))[0])
        tot[i - train_days] = os[i - train_days] / all_req

    tot_hit_rate = (np.sum(1 - tot) / (sim_days - train_days))
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
    plt.title("Total SOR for SNM/IRM contents with total: " + str(tot_hit_rate))
    plt.xlabel("Total SOR")
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

    print(cu)

    return cu, bs, os, irm, snm, tot


def plot_all_redundant_stuff(deliveries_0, deliveries_1, deliveries_2):
    cu_0 = np.zeros(sim_days - train_days)
    cu_1 = np.zeros(sim_days - train_days)
    cu_2 = np.zeros(sim_days - train_days)
    for i in range(train_days, sim_days):
        cu_0[i - train_days] = len(np.where(np.logical_and(deliveries_0[:, 3] == '!cu',
                                                           np.logical_and((deliveries_0[:, 1]).astype(float) >= i,
                                                                          (deliveries_0[:, 1]).astype(float) < i + 1)))[
                                       0]) / \
                               len(np.where(np.logical_and((deliveries_0[:, 1]).astype(float) >= i,
                                                           (deliveries_0[:, 1]).astype(float) < i + 1))[0])
        cu_1[i - train_days] = len(np.where(np.logical_and(deliveries_1[:, 3] == '!cu',
                                                           np.logical_and((deliveries_1[:, 1]).astype(float) >= i,
                                                                          (deliveries_1[:, 1]).astype(float) < i + 1)))[
                                       0]) / \
                               len(np.where(np.logical_and((deliveries_1[:, 1]).astype(float) >= i,
                                                           (deliveries_1[:, 1]).astype(float) < i + 1))[0])
        cu_2[i - train_days] = len(np.where(np.logical_and(deliveries_2[:, 3] == '!cu',
                                                           np.logical_and((deliveries_2[:, 1]).astype(float) >= i,
                                                                          (deliveries_2[:, 1]).astype(float) < i + 1)))[
                                       0]) / \
                               len(np.where(np.logical_and((deliveries_2[:, 1]).astype(float) >= i,
                                                           (deliveries_2[:, 1]).astype(float) < i + 1))[0])

    plotdata = pd.DataFrame({
        "P-Hybrid-0": cu_0 * 100,
        "P-Hybrid-33": cu_1 * 100,
        "P-Hybrid-66": cu_2 * 100},
        index=list(str('day' + str(i)) for i in range(1, 11)))
    plotdata.plot(kind="bar")
    plt.title("IRM redundant cached requests per day")
    plt.xlabel("Day")
    plt.ylabel("Redundant Ratio")
    plt.show()


def plot_all_process_stuff(deliveries_0, deliveries_1, deliveries_2):
    cu_0 = np.zeros(sim_days - train_days)
    cu_1 = np.zeros(sim_days - train_days)
    cu_2 = np.zeros(sim_days - train_days)
    cu_3 = np.zeros(sim_days - train_days)
    for i in range(train_days, sim_days):
        cu_0[i - train_days] = np.sum(deliveries_0[np.where(np.logical_and((deliveries_0[:, 1]).astype(float) >= i, (deliveries_0[:, 1]).astype(float) < i + 1)), 2].astype(int)) \
                               / len(np.where(np.logical_and((deliveries_0[:, 1]).astype(float) >= i, (deliveries_0[:, 1]).astype(float) < i + 1))[0])
        cu_1[i - train_days] = np.sum(deliveries_1[np.where(np.logical_and((deliveries_1[:, 1]).astype(float) >= i, (deliveries_1[:, 1]).astype(float) < i + 1)), 2].astype(int)) \
                               / len(np.where(np.logical_and((deliveries_1[:, 1]).astype(float) >= i, (deliveries_1[:, 1]).astype(float) < i + 1))[0])
        cu_2[i - train_days] = np.sum(deliveries_2[np.where(np.logical_and((deliveries_2[:, 1]).astype(float) >= i, (deliveries_2[:, 1]).astype(float) < i + 1)), 2].astype(int)) \
                               / len(np.where(np.logical_and((deliveries_2[:, 1]).astype(float) >= i, (deliveries_2[:, 1]).astype(float) < i + 1))[0])
    plotdata = pd.DataFrame({
        "P-Hybrid-33": cu_0,
        "P-Hybrid-66": cu_1,
        "P-Hybrid-100": cu_2},
        index=list(str('day' + str(i)) for i in range(1, 11)))
    plotdata.plot(kind="bar")
    plt.title("Process Price per day")
    plt.xlabel("Day")
    plt.ylabel("Process Price")
    plt.show()


def plot_deliver_cu_stuff(cu_, hit_rates):
    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Algorithms Comparison')
    fig.tight_layout()

    plotdata1 = pd.DataFrame({
        "HHRPCP": cu_[0],
        "P-Hybrid-100": cu_[1]},
        index=[" "])
    plotdata1.plot(kind="bar", ax=ax1)
    ax1.set_title('')
    ax1.set_ylabel("CU Delivery percentage")
    # ax1.legend(fancybox=True, shadow=False, fontsize='x-small')

    plotdata2 = pd.DataFrame({
        "HHRPCP": hit_rates[0],
        "P-Hybrid-100": hit_rates[1]},
        index=[" "])
    plotdata2.plot(kind="bar", ax=ax2)
    ax2.set_title('')
    ax2.set_ylabel("Total SOR")
    # ax2.legend(fancybox=True, shadow=False, fontsize='x-small')

    plt.tight_layout()
    plt.show()


def plot_one_stuff(title_1, title_2, thesis, hit_rate, titles):
    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Algorithms Comparison')
    fig.tight_layout()

    plotdata1 = pd.DataFrame({
        "HHRPCP": thesis[0],
        "P-Hybrid-0": thesis[1],
        "P-Hybrid-33": thesis[2],
        "P-Hybrid-66": thesis[3],
        "P-Hybrid-100": thesis[4]},
        index=[" "])
    # plotdata1 = pd.DataFrame(thesis, index=titles)
    # plotdata1.plot(kind="bar", legend=False, ax=ax1)
    plotdata1.plot(kind="bar", ax=ax1)
    ax1.set_title(title_1)
    # ax1.set_xlabel("Algorithms")
    ax1.set_ylabel("Average Delay")
    ax1.legend(fancybox=True, shadow=False, fontsize='x-small')

    plotdata2 = pd.DataFrame({
        "HHRPCP": hit_rate[0],
        "P-Hybrid-0": hit_rate[1],
        "P-Hybrid-33": hit_rate[2],
        "P-Hybrid-66": hit_rate[3],
        "P-Hybrid-100": hit_rate[4]},
        index=[" "])
    # plotdata2 = pd.DataFrame(hit_rate, index=titles)
    # plotdata2.plot(kind="bar", legend=False, ax=ax2)
    plotdata2.plot(kind="bar", ax=ax2)
    ax2.set_title(title_2)
    # ax2.set_xlabel("Algorithms")
    ax2.set_ylabel("Total SOR")
    ax2.legend(fancybox=True, shadow=False, fontsize='x-small')

    # plotdata = pd.DataFrame(thesis, index=titles)
    # plotdata.plot(kind="bar", legend=False)
    # plt.title(title_1)
    # plt.xlabel("Algorithms")
    # plt.ylabel("Average Delay")
    #
    # plotdata = pd.DataFrame(hit_rate, index=titles)
    # plotdata.plot(kind="bar", legend=False)
    # plt.title(title_2)
    # plt.xlabel("Algorithms")
    # plt.ylabel("Total SOR")

    plt.tight_layout()
    plt.show()


def plot_one_stuff_hr(title, thesis, hit_rate, titles):
    plotdata = pd.DataFrame(thesis, index=titles)
    plotdata.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Algorithms")
    plt.ylabel("Total SOR")
    plt.show()


def plot_all_stuff(title, thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles):
    plotdata = pd.DataFrame({
        "HHRPCP": thesis,
        "P-Hybrid-0": paper_share_0,
        "P-Hybrid-33": paper_share_1,
        "P-Hybrid-66": paper_share_2,
        "P-Hybrid-100": paper_share_3},
        index=titles)
    plotdata.plot(kind="bar")
    plt.title(title)
    plt.xticks(range(0, len(plotdata.index)), plotdata.index, rotation='vertical')
    plt.ylabel("Total SOR")
    plt.tight_layout()
    plt.show()


def plot_all_stuff_line(title, thesis, paper_share_0, paper_share_1, paper_share_2, paper_share_3, titles):
    plotdata = pd.DataFrame({
        "HHRPCP": thesis,
        "P-Hybrid-0": paper_share_0,
        "P-Hybrid-33": paper_share_1,
        "P-Hybrid-66": paper_share_2,
        "P-Hybrid-100": paper_share_3},
        index=titles)
    ax = plotdata.plot(kind="line")

    marker = ['*', '.', 'o', '^', 'v']
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(marker[i])

    plt.title(title)
    # plt.xticks(range(0, len(plotdata.index)), plotdata.index, rotation='vertical')
    plt.ylabel("Total SOR")
    plt.xlabel("Load Ratio")
    plt.tight_layout()
    plt.show()


def plot_redundant_stuff(title, paper_share_0, paper_share_1, paper_share_2, titles):
    plotdata = pd.DataFrame({
        "P-Hybrid-0": paper_share_0,
        "P-Hybrid-33": paper_share_1,
        "P-Hybrid-66": paper_share_2},
        index=titles)
    plotdata.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Algorithms")
    plt.ylabel("#requests")
    plt.show()
