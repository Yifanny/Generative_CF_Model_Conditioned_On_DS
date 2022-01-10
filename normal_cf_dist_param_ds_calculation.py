# normal_cf_ds_classification_by_ufl_w_t_dis.py
# 1. fix a_max a_conf S_jam for a driver; mix seq points and random points for initialization: failed
# 2. fix S_jam for a driver; mix seq points and random points for initialization: still tried
# 3. add temporal distance when calculating distance for assigning labels 

import numpy as np
from scipy.optimize import leastsq
import scipy.stats
import matplotlib.pyplot as plt
import pickle, time, copy, os, warnings
import pandas as pd
import pylab
from scipy.stats import entropy as kl_div
import threading, sys
from datetime import datetime
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

f = open('all_data_for_cf_model_w_t_pre_info_1101.pkl', 'rb')
all_data_for_cf_model = pickle.load(f)
f.close()

from scipy.optimize import minimize, basinhopping, brute, differential_evolution, shgo, dual_annealing
import random
from pathos.multiprocessing import ProcessingPool as Pool


def set_cons(a_max_n_boundary=[0.1, 2.5], desired_V_n_boundary=[1, 40], a_comf_n_boundary=[0.1, 5],
             S_jam_boundary=[0.1, 10], desired_T_n_boundary=[0.1, 5], beta_boundary=[4, 4]):
    # constraints: eq or ineq
    a_max_n_boundary = a_max_n_boundary
    desired_V_n_boundary = desired_V_n_boundary
    a_comf_n_boundary = a_comf_n_boundary
    S_jam_boundary = S_jam_boundary
    desired_T_n_boundary = desired_T_n_boundary
    beta_boundary = beta_boundary

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - a_max_n_boundary[0]}, \
            {'type': 'ineq', 'fun': lambda x: -x[0] + a_max_n_boundary[1]}, \
            {'type': 'ineq', 'fun': lambda x: x[1] - desired_V_n_boundary[0]}, \
            {'type': 'ineq', 'fun': lambda x: -x[1] + desired_V_n_boundary[1]}, \
            {'type': 'ineq', 'fun': lambda x: x[2] - a_comf_n_boundary[0]}, \
            {'type': 'ineq', 'fun': lambda x: -x[2] + a_comf_n_boundary[1]}, \
            {'type': 'ineq', 'fun': lambda x: x[3] - S_jam_boundary[0]}, \
            {'type': 'ineq', 'fun': lambda x: -x[3] + S_jam_boundary[1]}, \
            {'type': 'ineq', 'fun': lambda x: x[4] - desired_T_n_boundary[0]}, \
            {'type': 'ineq', 'fun': lambda x: -x[4] + desired_T_n_boundary[1]}, \
            {'type': 'ineq', 'fun': lambda x: x[5] - beta_boundary[0]}, \
            {'type': 'ineq', 'fun': lambda x: -x[5] + beta_boundary[1]})
    return cons


def initialize(a_max_n_boundary=[0.1, 2.5], desired_V_n_boundary=[1, 40], a_comf_n_boundary=[0.1, 5], S_jam_boundary=[0.1, 10], \
               desired_T_n_boundary=[0.1, 5], beta_boundary=[4, 4]):
    # a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    x0 = (random.uniform(a_max_n_boundary[0], a_max_n_boundary[1]),
          random.uniform(desired_V_n_boundary[0], desired_V_n_boundary[1]), \
          random.uniform(a_comf_n_boundary[0], a_comf_n_boundary[1]),
          random.uniform(S_jam_boundary[0], S_jam_boundary[1]), \
          random.uniform(desired_T_n_boundary[0], desired_T_n_boundary[1]), 4)
    return x0


def IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        # if a_max_n * a_comf_n <= 0:
        #     print("a_max_n", a_max_n, "a_comf_n", a_comf_n)
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        # if V_n_t * desired_T_n - (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n)) > 0:
        #     item2 = V_n_t * desired_T_n - (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        # else:
        #     item2 = 0
        return item1 + max(0, item2 + item3)

    a_n_t = []
    for i in range(len(delta_V_n_t)):
        desired_S_n = desired_space_hw(S_jam_n, V_n_t[i], desired_T_n, delta_V_n_t[i], a_max_n, a_comf_n)
        a_n_t.append(a_max_n * (1 - (V_n_t[i] / desired_V_n) ** beta - (desired_S_n / S_n_t[i]) ** 2))

    return np.array(a_n_t)


def tv_IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        # if a_max_n * a_comf_n <= 0:
        #     print("a_max_n", a_max_n, "a_comf_n", a_comf_n)
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        # if V_n_t * desired_T_n - (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n)) > 0:
        #     item2 = V_n_t * desired_T_n - (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        # else:
        #     item2 = 0
        return item1 + max(0, item2 + item3)

    a_n_t = []
    for i in range(len(a_max_n)):
        desired_S_n = desired_space_hw(S_jam_n[i], V_n_t, desired_T_n[i], delta_V_n_t, a_max_n[i], a_comf_n[i])
        a_n_t.append(a_max_n[i] * (1 - (V_n_t / desired_V_n[i]) ** beta - (desired_S_n / S_n_t) ** 2))

    return np.array(a_n_t)


def IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        # if V_n_t * desired_T_n - (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n)) > 0:
        #     item2 = V_n_t * desired_T_n - (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        # else:
        #     item2 = 0
        return item1 + max(0, item2 + item3)

    desired_S_n = desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n)
    a_n_t = a_max_n * (1 - (V_n_t / desired_V_n) ** beta - (desired_S_n / S_n_t) ** 2)

    return a_n_t


def obj_func(args):
    a, delta_V_n_t, S_n_t, V_n_t = args
    # x[0:6]: a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    # err = lambda x: np.sqrt( np.sum( ( (a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) / a ) ** 2) / len(a) )
    # err = lambda x: np.sqrt( np.sum((a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2) / np.sum(a**2))
    err = lambda x: np.sqrt(
        np.sum((a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2) / len(a))
    return err


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder " + path + " ...  ---")
        print("---  OK  ---")


def _timed_run(func, distribution, args=(), kwargs={}, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout is exceeded.
    http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    """

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
            self.exc_info = (None, None, None)

        def run(self):
            try:
                self.result = func(args, **kwargs)
            except Exception as err:  # pragma: no cover
                self.exc_info = sys.exc_info()

        def suicide(self):  # pragma: no cover
            raise RuntimeError('Stop has been called')

    it = InterruptableThread()
    it.start()
    started_at = datetime.now()
    it.join(self.timeout)
    ended_at = datetime.now()
    diff = ended_at - started_at

    if it.exc_info[0] is not None:  # pragma: no cover ;  if there were any exceptions
        a, b, c = it.exc_info
        raise Exception(a, b, c)  # communicate that to caller

    if it.isAlive():  # pragma: no cover
        it.suicide()
        raise RuntimeError
    else:
        return it.result


def fit_posterior(data, Nbest=3, timeout=10):
    param_names = ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]
    common_distributions = ['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'rayleigh',
                            'uniform']
    distributions = {}
    data = np.array(data).T
    for i in range(len(data)):
        fitted_param = {}
        fitted_pdf = {}
        sumsquare_error = {}
        y, x = np.histogram(data[i], bins=100, density=True)
        x = [(this + x[i + 1]) / 2. for i, this in enumerate(x[0:-1])]
        for distribution in common_distributions:
            try:
                # need a subprocess to check time it takes. If too long, skip it
                dist = eval("scipy.stats." + distribution)

                param = dist.fit(data[i])

                pdf_fitted = dist.pdf(x, *param)

                fitted_param[distribution] = param[:]
                fitted_pdf[distribution] = pdf_fitted

                # calculate error
                sq_error = pylab.sum((fitted_pdf[distribution] - y) ** 2)
                sumsquare_error[distribution] = sq_error

                # calcualte information criteria
                # logLik = np.sum(dist.logpdf(x, *param))
                # k = len(param[:])
                # n = len(data[i])
                # aic = 2 * k - 2 * logLik
                # bic = n * np.log(sq_error / n) + k * np.log(n)

                # calcualte kullback leibler divergence
                # kullback_leibler = kl_div(fitted_pdf[distribution], self.y)

                 # compute some errors now
                # _fitted_errors[distribution] = sq_error
                # _aic[distribution] = aic
                # _bic[distribution] = bic
                # _kldiv[distribution] = kullback_leibler
            except Exception:  # pragma: no cover
                print("SKIPPED {} distribution (taking more than {} seconds)".format(distribution, timeout))
                # print(Exception)
                # if we cannot compute the error, set it to large values
                # fitted_param[distribution] = []
                # fitted_pdf[distribution] = np.nan
                # sumsquare_error[distribution] = np.inf
        srt_sq_error = sorted(sumsquare_error.items(), key=lambda kv:(kv[1], kv[0]))
        for j in range(Nbest):
            dist_name = srt_sq_error[j][0]
            sq_error = srt_sq_error[j][1]
            param = fitted_param[dist_name]
            pdf = fitted_pdf[dist_name]
            if not param_names[i] in distributions:
                distributions[param_names[i]] = [{"distribution": dist_name, "fitted_param": param, "sq_error": sq_error}]
            else:
                distributions[param_names[i]].append({"distribution": dist_name, "fitted_param": param, "sq_error": sq_error})
    return distributions


def get_peak_value(data):
    a_max_n_boundary = [0.1, 2.5]
    desired_V_n_boundary = [1, 40]
    a_comf_n_boundary = [0.1, 5]
    S_jam_boundary = [0.1, 10]
    desired_T_n_boundary = [0.1, 5]

    data = data.T

    a_max_hist, a_max_bins = np.histogram(data[0], bins=100, range=(a_max_n_boundary[0], a_max_n_boundary[1]))
    desired_V_hist, desired_V_bins = np.histogram(data[1], bins=100, range=(desired_V_n_boundary[0], desired_V_n_boundary[1]))
    a_comf_hist, a_comf_bins = np.histogram(data[2], bins=100, range=(a_comf_n_boundary[0], a_comf_n_boundary[1]))
    S_jam_hist, S_jam_bins = np.histogram(data[3], bins=100, range=(S_jam_boundary[0], S_jam_boundary[1]))
    desired_T_hist, desired_T_bins = np.histogram(data[4], bins=100, range=(desired_T_n_boundary[0], desired_T_n_boundary[1]))

    idx = np.argmax(a_max_hist)
    a_max = (a_max_bins[idx + 1] + a_max_bins[idx]) / 2
    idx = np.argmax(desired_V_hist)
    desired_V = (desired_V_bins[idx + 1] + desired_V_bins[idx]) / 2
    idx = np.argmax(a_comf_hist)
    a_comf = (a_comf_bins[idx + 1] + a_comf_bins[idx]) / 2
    idx = np.argmax(S_jam_hist)
    S_jam = (S_jam_bins[idx + 1] + S_jam_bins[idx]) / 2
    idx = np.argmax(desired_T_hist)
    desired_T = (desired_T_bins[idx + 1] + desired_T_bins[idx]) / 2

    return a_max, desired_V, a_comf, S_jam, desired_T


def get_mean_value(data):
    data = data.T
    return np.mean(data[0]), np.mean(data[1]), np.mean(data[2]), np.mean(data[3]), np.mean(data[4])


def get_tv_params(next_v, v_id, all_cf_data):
    print("-------------------------------------------------------------------------------------------------")
    print(str(next_v) + 'th vehicle with id ' + str(v_id))
    # [delta_v_l, space_hw_l, ego_v_l, a_l]
    delta_V_n_t = np.array(all_cf_data[0])
    S_n_t = np.array(all_cf_data[1])
    V_n_t = np.array(all_cf_data[2])
    a = np.array(all_cf_data[3])
    t = np.array(all_cf_data[4])
    pre_v = np.array(all_cf_data[5])
    pre_tan_acc = np.array(all_cf_data[6])
    pre_lat_acc = np.array(all_cf_data[7])

    print(len(a), np.mean(np.abs(a)))
    print(len(pre_v), np.mean(pre_v))
    print(len(pre_tan_acc), np.mean(np.abs(pre_tan_acc)))
    print(len(pre_lat_acc), np.mean(np.abs(pre_lat_acc)))

    args = (a, delta_V_n_t, S_n_t, V_n_t)
    cons = set_cons()

    if os.path.exists('0714_dist_param/'+str(int(v_id))+'/using_all_data.txt'):
        res_param = np.loadtxt('0714_dist_param/'+str(int(v_id))+'/using_all_data.txt')
    else:
        return False, False, False
        while True:
            try:
                x0 = np.asarray(initialize())
                res = minimize(obj_func(args), x0, constraints=cons, method='trust-constr')
                if res.success:
                    break
            except ValueError:
                continue
        rmse_using_all_data = res.fun
        # f = open('0704_dist_param/res_using_all_data.txt', 'a+')
        # f.write("v id " + str(int(v_id)) + " " + str(res.success) + " | RMSE: " + str(res.fun) + " | a_max: " + str(
        #     res.x[0]) + " | desired_V: " + str(res.x[1]) + " | a_comf: " + str(res.x[2]) + " | S_jam: " + str(res.x[3]) +
        #         " | desired_T: " + str(res.x[4]) + " | beta: " + str(res.x[5]) + "\n")
        # f.close()
        mkdir('0714_dist_param/'+str(int(v_id))+'/')
        mkdir('0714_dist_param/' + str(int(v_id)) + '/posterior_figure/')
        np.savetxt('0714_dist_param/'+str(int(v_id))+'/using_all_data.txt', np.array(res.x))
        res_param = res.x

    fix_a_max = res_param[0]
    fix_desired_V = res_param[1]
    fix_a_comf = res_param[2]
    fix_S_jam = res_param[3]
    fix_desired_T = res_param[4]
    fix_beta = res_param[5]

    data_array = np.array([delta_V_n_t, S_n_t, V_n_t, a, t]).T
    data_array = data_array[data_array[:, -1].argsort()]
    t = np.array(data_array[:, 4])
    # data_array = data_array[:, 0:-1]
    data = data_array.tolist()

    sample_size = 10000
    i = 0
    fix_sum_err = 0
    tv_sum_err = 0
    # tv_params = []
    tv_params_mean = []
    for frame in data:
        if i % 100 == 0:
            print(str(next_v)+"th vehicle v_id "+str(v_id)+" frame "+str(i))
        if os.path.exists('0714_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt'):
            with open('0714_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt') as f:
                accept_tv_params = np.array([line.strip().split() for line in f], float)
            # accept_tv_params = np.loadtxt('0714_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt')
            # tv_params.append(accept_tv_params)
            a_max, desired_V, a_comf, S_jam, desired_T = get_mean_value(accept_tv_params)
            tv_params_mean.append([a_max, desired_V, a_comf, S_jam, desired_T])
        else:
            if i == 0:
                return False, False, False
            print('0714_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt does not exist!!!')
            i += 1
            a_max, desired_V, a_comf, S_jam, desired_T = fix_a_max, fix_desired_V, fix_a_comf, fix_S_jam, fix_desired_T
            tv_params_mean.append([a_max, desired_V, a_comf, S_jam, desired_T])
        delta_V_n_t = frame[0]
        S_n_t = frame[1]
        V_n_t = frame[2]
        a_n_t = frame[3]
        fix_err = (a_n_t - IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, fix_a_max, fix_desired_V, fix_a_comf, fix_S_jam,
                                              fix_desired_T, 4)) ** 2

        fix_sum_err += fix_err

        # a_max, desired_V, a_comf, S_jam, desired_T = get_peak_value(accept_tv_params)
        a_n_t_hat = IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, a_max, desired_V, a_comf, S_jam, desired_T, 4)

        tv_sum_err += (a_n_t_hat - a_n_t) ** 2

        # if (a_n_t_hat - a_n_t) ** 2 > fix_err:
        #     distributions = fit_posterior(accept_tv_params)
        #     print("--------------------------------------------------------------------------------------------")
        #     print(next_v, "v_id", v_id, "frame", i, "tv_err", (a_n_t_hat - a_n_t) ** 2, "fix_err", fix_err)
        #     for param_name in ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]:
        #         for dist in distributions[param_name]:
        #             print(v_id, param_name, dist["distribution"], dist["fitted_param"], dist["sq_error"])
        #     print("--------------------------------------------------------------------------------------------")

        i += 1
    print("all data %d | RMSE: %.4f | a_max: %.4f | desired_V: %.4f | a_comf: %.4f | S_jam: %.4f | desired_T: %.4f | beta: %.3f" % \
        (v_id, np.sqrt(fix_sum_err / len(a)), res_param[0], res_param[1], res_param[2], res_param[3], res_param[4], res_param[5]))
    print(str(int(v_id)), "RMSE:", np.sqrt(fix_sum_err / len(a)), np.sqrt(tv_sum_err / len(a)))
    # print(str(int(v_id)), "mean:", np.mean(np.abs(a-a_hat)), np.std(np.abs(a-a_hat)))
    # tv_params = np.array(tv_params)
    # print(str(int(v_id)), "tv params:", tv_params.shape)
    np.savetxt('0714_dist_param/' + str(int(v_id)) + '/tv_params_mean.txt', np.array(tv_params_mean))
    return np.array(tv_params_mean), np.sqrt(tv_sum_err / len(a))


def JS_divergence(p, q):
    # lb = boundary[0]
    # ub = boundary[1]
    # interval = (ub - lb) / 3000
    # x = np.arange(lb, ub, interval)
    # p = eval("scipy.stats." + dist1["name"]).pdf(x, *dist1["param"])
    # q = eval("scipy.stats." + dist2["name"]).pdf(x, *dist2["param"])
    p = p.reshape(1000, 5)
    q = q.reshape(1000, 5)
    p = p.T
    q = q.T
    js = 0
    for i in range(5):
        M = (p[i] + q[i]) / 2
        js += 0.5 * scipy.stats.entropy(p[i], M) + 0.5 * scipy.stats.entropy(q[i], M)
    # print(js)
    return js


def get_raw_features(next_v, v_id):
    print("-------------------------------------------------------------------------------------------------")
    print(str(next_v) + 'th vehicle with id ' + str(v_id))

    if os.path.exists('0714_dist_param/' + str(int(v_id)) + '/tv_params_mean.txt'):
        tv_params_mean = np.loadtxt('0714_dist_param/'+str(int(v_id))+'/tv_params_mean.txt')
    else:
        return


    mean = np.mean(tv_params_mean, axis=0)
    std = np.std(tv_params_mean, axis=0)
    per25 = np.percentile(tv_params_mean, 0.25, axis=0)
    per75 = np.percentile(tv_params_mean, 0.75, axis=0)

    a_max_inc_diff_seq = []
    a_max_dec_diff_seq = []
    desired_V_inc_diff_seq = []
    desired_V_dec_diff_seq = []
    a_comf_inc_diff_seq = []
    a_comf_dec_diff_seq = []
    S_jam_inc_diff_seq = []
    S_jam_dec_diff_seq = []
    desired_T_inc_diff_seq = []
    desired_T_dec_diff_seq = []
    print(tv_params_mean.shape)
    for i in range(len(tv_params_mean) - 1):
        diff = tv_params_mean[i + 1] - tv_params_mean[i]
        if diff[0] > 0:
            a_max_inc_diff_seq.append(diff[0])
        elif diff[0] < 0:
            a_max_dec_diff_seq.append(diff[0])

        if diff[1] > 0:
            desired_V_inc_diff_seq.append(diff[1])
        elif diff[1] < 0:
            desired_V_dec_diff_seq.append(diff[1])

        if diff[2] > 0:
            a_comf_inc_diff_seq.append(diff[2])
        elif diff[2] < 0:
            a_comf_dec_diff_seq.append(diff[2])

        if diff[3] > 0:
            S_jam_inc_diff_seq.append(diff[3])
        elif diff[3] < 0:
            S_jam_dec_diff_seq.append(diff[3])

        if diff[4] > 0:
            desired_T_inc_diff_seq.append(diff[4])
        elif diff[4] < 0:
            desired_T_dec_diff_seq.append(diff[4])
    diff_seq = [np.mean(a_max_inc_diff_seq), np.std(a_max_inc_diff_seq), np.mean(a_max_dec_diff_seq), np.std(a_max_dec_diff_seq),
                np.mean(desired_V_inc_diff_seq), np.std(desired_V_inc_diff_seq), np.mean(desired_V_dec_diff_seq), np.std(desired_V_dec_diff_seq),
                np.mean(a_comf_inc_diff_seq), np.std(a_comf_inc_diff_seq), np.mean(a_comf_dec_diff_seq), np.std(a_comf_dec_diff_seq),
                np.mean(S_jam_inc_diff_seq), np.std(S_jam_inc_diff_seq), np.mean(S_jam_dec_diff_seq), np.std(S_jam_dec_diff_seq),
                np.mean(desired_T_inc_diff_seq), np.std(desired_T_inc_diff_seq), np.mean(desired_T_dec_diff_seq), np.std(desired_T_dec_diff_seq)]

    return mean, std, per25, per75, diff_seq


def scale_features(mean_l, std_l, per25_l, per75_l, diff_seq_l):
    scaled_mean_l = minmax_scale(mean_l, axis=0)
    scaled_std_l = minmax_scale(std_l, axis=0)
    scaled_per25_l = minmax_scale(per25_l, axis=0)
    scaled_per75_l = minmax_scale(per75_l, axis=0)
    scaled_diff_seq_l = minmax_scale(diff_seq_l, axis=0)

    return scaled_mean_l, scaled_std_l, scaled_per25_l, scaled_per75_l, scaled_diff_seq_l


def cal_agg_index(mean, per25, per75, diff_seq):
    # a_max, desired_v, a_comf, S_jam, deisred_T
    relation_op = [1, 1, 1, -1, -1]
    res1 = 0
    res2 = 0
    res3 = 0
    for i in range(5):
        res1 += relation_op[i] * (mean[i] + per25[i] + per75[i])
        res2 += relation_op[i] * (diff_seq[i * 4] + diff_seq[i * 4 + 1])
        res3 += relation_op[i] * (diff_seq[i * 4 + 2] + diff_seq[i * 4 + 3])
    return [res1, res2, res3]


def cal_agg_matrix(mean, std, per25, per75, diff_seq):
    # a_max, desired_v, a_comf, S_jam, deisred_T
    relation_op = [1, 1, 1, -1, -1]
    agg_matrix = np.zeros((2, 5))
    for i in range(5):
        agg_matrix[0][i] = mean[i]
        agg_matrix[1][i] = std[i]
        # agg_matrix[2][i] = per75[i]
        # agg_matrix[3][i] = diff_seq[i * 4]
        # agg_matrix[4][i] = diff_seq[i * 4 + 1]
        # agg_matrix[5][i] = diff_seq[i * 4 + 2]
        # agg_matrix[6][i] = diff_seq[i * 4 + 3]
    return agg_matrix



def analyze_tv_params(v_id, tv_params, tv_params_mean):
    # plot tv_params_mean
    mkdir("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/")
    tv_params_mean = tv_params_mean.T
    # plt.figure()
    # plt.plot(range(len(tv_params_mean[0])), tv_params_mean[0], color="red")
    # plt.savefig("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/a_max.png")
    # plt.figure()
    # plt.plot(range(len(tv_params_mean[0])), tv_params_mean[1]/4, color="blue")
    # plt.savefig("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/desired_V.png")
    # plt.figure()
    # plt.plot(range(len(tv_params_mean[0])), tv_params_mean[2], color="green")
    # plt.savefig("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/a_comf.png")
    # plt.figure()
    # plt.plot(range(len(tv_params_mean[0])), tv_params_mean[3], color="orange")
    # plt.savefig("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/S_jam.png")
    # plt.figure()
    # plt.plot(range(len(tv_params_mean[0])), tv_params_mean[4], color="pink")
    # plt.savefig("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/all.png")

    # calculate cov
    cov = []
    corrcoef = []
    for t in range(len(tv_params)):
        params = tv_params[t].T
        cov.append(np.cov(params))
        this_corrcoef = np.corrcoef(params)
        for i in range(5):
            this_corrcoef[i][i] = 0
        corrcoef.append(this_corrcoef)

    print(np.array(cov).shape)
    print(np.mean(cov, axis=0))       # covariance almost 0
    print(np.array(corrcoef).shape)
    print(np.mean(corrcoef, axis=0))

    print(np.cov(tv_params_mean))
    print(np.corrcoef(tv_params_mean))
    exit()

    tv_params_mean = tv_params_mean.T
    norm_tv_params_mean = normalize(tv_params_mean)

    tidy_tv_params = []
    for params in tv_params:
        sample_idx = np.random.choice(np.arange(len(params)), 1000, replace=False)
        tidy_tv_params.append(params[sample_idx].reshape(1000 * 5, ))
    tidy_tv_params = normalize(np.array(tidy_tv_params))
    print(tidy_tv_params.shape)

    # KMeans clustering
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tidy_tv_params)
        print("kmeans", k, silhouette_score(tidy_tv_params, kmeans.labels_))
        plt.figure()
        plt.plot(range(len(kmeans.labels_)), kmeans.labels_)
        plt.savefig("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/1000d_kmeans_cluster_" + str(int(k)) + ".png")
    # exit()

    # hierarchical clustering using specific distance metric:
    linkage_matrix = linkage(tidy_tv_params, method="average") #, metric=lambda u, v: JS_divergence(u, v))
    print("linkage matrix", linkage_matrix.shape)
    print(linkage_matrix)

    # choose the best k according to silhouette score
    for k in range(2, 10):
        # for c in ['maxclust', 'inconsistent', 'distance', 'monocrit', 'maxclust_monocrit']:
        c = 'maxclust'
        labels = fcluster(linkage_matrix, k, criterion=c)
        plt.figure()
        plt.plot(range(len(labels)), labels)
        plt.savefig("0714_dist_param/" + str(int(v_id)) + "/tv_params_figure/eud_hier_cluster_" + str(int(k)) + ".png")
        score = silhouette_score(tidy_tv_params, labels) #, metric=lambda u, v: JS_divergence(u, v))
        print("hierarchical", k, "score:", score)


if __name__ == "__main__":
    # a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    x0 = (1.0, 20, 0.5, 2, 2, 4)
    next_v = 1
    # req_v_id = int(sys.argv[1])
    print("Start!")
    next_vs = []
    v_ids = []
    all_cf_datas = []
    for v_id, all_cf_data in all_data_for_cf_model.items():
        next_vs.append(next_v)
        v_ids.append(v_id)
        all_cf_datas.append(all_cf_data)
        # if next_v == req_v_id:
        #     cal_tv_params(next_v, v_id, all_cf_data)
        #     exit()
        next_v += 1

    print("next_vs", np.array(next_vs).shape)
    print("v_ids", np.array(v_ids).shape)
    print("all_cf_datas shape:", np.array(all_cf_datas).shape)

    mean_l = []
    per25_l = []
    per75_l = []
    std_l = []
    diff_seq_l = []
    for i in range(200):
        mean, std, per25, per75, diff_seq = get_raw_features(next_vs[i], v_ids[i])
        mean_l.append(mean)
        std_l.append(std)
        per25_l.append(per25)
        per75_l.append(per75)
        diff_seq_l.append(diff_seq)
    mean_l = np.array(mean_l)
    std_l = np.array(std_l)
    per25_l = np.array(per25_l)
    per75_l = np.array(per75_l)
    diff_seq_l = np.array(diff_seq_l)
    print(mean_l.shape, std_l.shape, per25_l.shape, per75_l.shape, diff_seq_l.shape)
    scale_mean_l, scale_std_l, scaled_per25_l, scaled_per75_l, scaled_diff_seq_l = scale_features(mean_l, std_l, per25_l, per75_l, diff_seq_l)
    print(scale_mean_l, scale_std_l, scaled_per25_l, scaled_per75_l, scaled_diff_seq_l)
    print(scale_mean_l.shape, scale_std_l.shape, scaled_per25_l.shape, scaled_per75_l, scaled_diff_seq_l.shape)
    agg_indexes = []
    for i in range(200):
        # agg_matrix = cal_agg_matrix(scale_mean_l[i], scale_std_l[i], scaled_per25_l[i], scaled_per75_l[i], scaled_diff_seq_l[i])
        agg_matrix = cal_agg_index(scale_mean_l[i], scaled_per25_l[i], scaled_per75_l[i], scaled_diff_seq_l[i])
        agg_indexes.append(sum(agg_matrix))
    print(agg_indexes)
    agg_indexes = np.array(agg_indexes)
    print(agg_indexes.shape)
    np.savetxt("0714_dist_param/200_1d_agg_indexes.txt", np.array(agg_indexes))
    exit()
    # analyze_tv_params(v_ids[0], tv_params, tv_params_mean)
    print(np.mean(tv_rmse), np.std(tv_rmse))
    plt.figure()
    plt.boxplot(tv_rmse)
    plt.savefig("0714_dist_param/tv_rmse_boxplot.png")
    exit()
    pool = Pool(nodes=72)
    pool.map(cal_tv_params, next_vs, v_ids, all_cf_datas)
    pool.close()
    pool.join()
