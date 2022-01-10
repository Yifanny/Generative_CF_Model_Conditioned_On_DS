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
from sklearn.metrics import mean_squared_error

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


def initialize(a_max_n_boundary=[0.1, 2.5], desired_V_n_boundary=[1, 40], a_comf_n_boundary=[0.1, 5],
               S_jam_boundary=[0.1, 10], \
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


def sim_new_spacing(new_pre_x, new_pre_y, old_ego_x, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    return np.sqrt((new_pre_x - old_ego_x) ** 2 + (new_pre_y - old_ego_y) ** 2) - (2 * V_n_t + a_n_t * delta_t) / 2 * delta_t


def sim_new_spacing_l(new_pre_x, new_pre_y, old_ego_x, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    res_spacing = []
    for i in range(len(new_pre_x)):
        res_spacing.append(np.sqrt((new_pre_x[i] - old_ego_x[i]) ** 2 + (new_pre_y[i] - old_ego_y[i]) ** 2) - (2 * V_n_t[i] + a_n_t[i] * delta_t) / 2 * delta_t)
    return res_spacing


def tv_sim_new_spacing(new_pre_x, new_pre_y, old_ego_x, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    res_spacing = []
    for i in range(len(a_n_t)):
        res_spacing.append(np.sqrt((new_pre_x - old_ego_x) ** 2 + (new_pre_y - old_ego_y) ** 2) - (2 * V_n_t + a_n_t[i] * delta_t) / 2 * delta_t)
    return np.array(res_spacing)


def cal_new_a(new_pre_x, new_pre_y, old_ego_x, old_ego_y, V_n_t, S_n_t_1, delta_t=0.04):
    return ((2 * (np.sqrt((new_pre_x - old_ego_x) ** 2 + (new_pre_y - old_ego_y) ** 2) - S_n_t_1) / delta_t) - 2 * V_n_t) / delta_t


def obj_func(args):
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_x, next_pre_y, ego_x, ego_y = args
    # x[0:6]: a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    # err = lambda x: np.sqrt( np.sum( ( (a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) / a ) ** 2) / len(a) )
    # err = lambda x: np.sqrt( np.sum((a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2) / np.sum(a**2))
    delta_t = 0.04
    err = lambda x: mean_squared_error(S_n_t_1, sim_new_spacing_l(next_pre_x, next_pre_y, ego_x, ego_y, V_n_t,
                                                                IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])))
    return err


def all_rmse_using_fixed_params(args, x):
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_x, next_pre_y, ego_x, ego_y = args
    # x[0:6]: a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    # err = lambda x: np.sqrt( np.sum( ( (a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) / a ) ** 2) / len(a) )
    # err = lambda x: np.sqrt( np.sum((a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2) / np.sum(a**2))
    delta_t = 0.04
    err = mean_squared_error(S_n_t_1, sim_new_spacing_l(next_pre_x, next_pre_y, ego_x, ego_y, V_n_t,
                                                        IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])))
    return err


def rmse_using_fixed_params(args, params):
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_x, next_pre_y, ego_x, ego_y = args
    # x[0:6]: a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    # err = lambda x: np.sqrt( np.sum( ( (a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) / a ) ** 2) / len(a) )
    # err = lambda x: np.sqrt( np.sum((a - IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], x[5])) ** 2) / np.sum(a**2))
    delta_t = 0.04
    # print(args, params)
    err = S_n_t_1 - sim_new_spacing(next_pre_x, next_pre_y, ego_x, ego_y, V_n_t,
                                    IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, params[0], params[1], params[2],
                                                       params[3], params[4], params[5]))
    return err ** 2


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


def initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=1, size=5000):
    a_max_n_boundary = [0.1, 2.5]
    desired_V_n_boundary = [1, 40]
    a_comf_n_boundary = [0.1, 5]
    S_jam_boundary = [0.1, 10]
    desired_T_n_boundary = [0.1, 5]

    def generate_one_sample(mu, sigma, boundary):
        while True:
            p = np.random.normal(mu, sigma, 1)[0]
            if boundary[0] < p < boundary[1]:
                break
        return p

    new_params = []
    for _ in range(size):
        new_a_max = generate_one_sample(a_max, scale, a_max_n_boundary)
        new_desired_V = generate_one_sample(desired_V, scale*5, desired_V_n_boundary)
        new_a_comf = generate_one_sample(a_comf, scale, a_comf_n_boundary)
        new_S_jam = generate_one_sample(S_jam, scale, S_jam_boundary)
        new_desired_T = generate_one_sample(desired_T, scale, desired_T_n_boundary)
        new_params.append([new_a_max, new_desired_V, new_a_comf, new_S_jam, new_desired_T])

    return np.array(new_params)


def generate_uniform_params(size=5000):
    a_max_n_boundary = [0.1, 2.5]
    desired_V_n_boundary = [1, 40]
    a_comf_n_boundary = [0.1, 5]
    S_jam_boundary = [0.1, 10]
    desired_T_n_boundary = [0.1, 5]

    new_a_max = np.random.uniform(a_max_n_boundary[0], a_max_n_boundary[1], size)
    new_desired_V = np.random.uniform(desired_V_n_boundary[0], desired_V_n_boundary[1], size)
    new_a_comf = np.random.uniform(a_comf_n_boundary[0], a_comf_n_boundary[1], size)
    new_S_jam = np.random.uniform(S_jam_boundary[0], S_jam_boundary[1], size)
    new_desired_T = np.random.uniform(desired_T_n_boundary[0], desired_T_n_boundary[1], size)
    new_params = np.array([new_a_max, new_desired_V, new_a_comf, new_S_jam, new_desired_T]).T

    return new_params


def generate_new_params(distributions, size=5000):
    a_max_n_boundary = [0.1, 2.5]
    desired_V_n_boundary = [1, 40]
    a_comf_n_boundary = [0.1, 5]
    S_jam_boundary = [0.1, 10]
    desired_T_n_boundary = [0.1, 5]

    best_posterior = {}
    for param_name in ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]:
        best_posterior[param_name] = {"name": distributions[param_name][0]["distribution"],
                                      "param": distributions[param_name][0]["fitted_param"]}

    def generate_one_sample(best_posterior, boundary):
        while True:
            p = eval("scipy.stats." + best_posterior["name"] + ".rvs")(*best_posterior["param"], size=1)[0]
            if boundary[0] < p < boundary[1]:
                break
        return p

    new_params = []
    for _ in range(size):
        new_a_max = generate_one_sample(best_posterior["a_max"], a_max_n_boundary)
        new_desired_V = generate_one_sample(best_posterior["desired_V"], desired_V_n_boundary)
        new_a_comf = generate_one_sample(best_posterior["a_comf"], a_comf_n_boundary)
        new_S_jam = generate_one_sample(best_posterior["S_jam"], S_jam_boundary)
        new_desired_T = generate_one_sample(best_posterior["desired_T"], desired_T_n_boundary)
        new_params.append([new_a_max, new_desired_V, new_a_comf, new_S_jam, new_desired_T])

    return np.array(new_params)


def cal_tv_params(next_v, v_id, all_cf_data):
    print("-------------------------------------------------------------------------------------------------")
    print(str(next_v) + 'th vehicle with id ' + str(v_id))
    # data_array = np.array([new_delta_V_n_t, new_S_n_t, new_V_n_t, new_a, new_S_n_t_y, new_ego_x, new_ego_y, new_next_pre_x, new_next_pre_y, new_frame_id]).T
    cons = set_cons()

    # S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_x, next_pre_y, ego_x, ego_y = args
    new_delta_V_n_t = all_cf_data.T[0]
    new_S_n_t = all_cf_data.T[1]
    new_V_n_t = all_cf_data.T[2]
    new_a = all_cf_data.T[3]
    new_S_n_t_y = all_cf_data.T[4]
    new_ego_x = all_cf_data.T[5]
    new_ego_y = all_cf_data.T[6]
    new_next_pre_x = all_cf_data.T[7]
    new_next_pre_y = all_cf_data.T[8]
    new_frame_id = all_cf_data.T[9]
    args = (np.array(new_S_n_t_y), np.array(new_delta_V_n_t), np.array(new_S_n_t), np.array(new_V_n_t), np.array(new_next_pre_x),
            np.array(new_next_pre_y), np.array(new_ego_x), np.array(new_ego_y))

    data_array = all_cf_data
    print("spacing", np.mean(new_S_n_t_y), np.mean(new_S_n_t), np.mean(np.array(new_S_n_t_y) - np.array(new_S_n_t)))
    print(data_array.shape)

    if os.path.exists('/root/AAAI2022_neuma/0803_mop_space_dist_param/'+str(int(v_id))+'/using_all_data.txt'):
        res_param = np.loadtxt('/root/AAAI2022_neuma/0803_mop_space_dist_param/'+str(int(v_id))+'/using_all_data.txt')
        rmse_using_all_data = all_rmse_using_fixed_params(args, res_param)
    else:
        while True:
            try:
                x0 = np.asarray(initialize())
                res = minimize(obj_func(args), x0, constraints=cons, method='trust-constr')
                if res.success:
                    break
            except ValueError:
                continue
        rmse_using_all_data = res.fun
        mkdir('/root/AAAI2022_neuma/0803_mop_space_dist_param/'+str(int(v_id))+'/')
        mkdir('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/posterior_figure/')
        np.savetxt('/root/AAAI2022_neuma/0803_mop_space_dist_param/'+str(int(v_id))+'/using_all_data.txt', np.array(res.x))
        res_param = res.x

    fix_a_max = res_param[0]
    fix_desired_V = res_param[1]
    fix_a_comf = res_param[2]
    fix_S_jam = res_param[3]
    fix_desired_T = res_param[4]
    fix_beta = res_param[5]

    a_max = res_param[0]
    desired_V = res_param[1]
    a_comf = res_param[2]
    S_jam = res_param[3]
    desired_T = res_param[4]
    beta = res_param[5]

    sample_size = 5000
    i = 0
    fix_sum_err = 0
    tv_sum_err = 0
    for frame in data_array:
        print(str(next_v)+"th vehicle v_id "+str(v_id)+" frame "+str(i))
        # if os.path.exists('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt'):
        #     accept_tv_params = np.loadtxt('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt')
        #     if len(accept_tv_params) < sample_size * 0.95:
        #         accept_tv_params = np.loadtxt('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i - 1)) + '_tv_params.txt')
        #         distributions = fit_posterior(accept_tv_params)
                # print("accept tv params is empty", accept_tv_params.shape)
        #     elif not os.path.exists('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i + 1)) + '_tv_params.txt'):
        #         accept_tv_params = np.loadtxt('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt')
        #         distributions = fit_posterior(accept_tv_params)
                # print("next frame not fitted", accept_tv_params.shape)
        #     else:
        #         i += 1
        #         continue

        delta_V_n_t = frame[0]
        S_n_t = frame[1]
        V_n_t = frame[2]
        a = frame[3]
        S_n_t_y = frame[4]
        ego_x = frame[5]
        ego_y = frame[6]
        next_pre_x = frame[7]
        next_pre_y = frame[8]
        frame_id = frame[9]
        args = (S_n_t_y, delta_V_n_t, S_n_t, V_n_t, next_pre_x, next_pre_y, ego_x, ego_y)

        delta_t = 0.04
        fix_err = rmse_using_fixed_params(args, res_param)

        if fix_err < 0.001:
            accept_threshold = fix_err
        else:
            accept_threshold = fix_err/2
        fix_sum_err += fix_err
        scale = 1
        iters = 0
        max_iters = 500
        this_sample_size = copy.deepcopy(sample_size)
        accept_tv_params = []
        while iters < max_iters:
            if len(accept_tv_params) == 0 and iters > max_iters * 0.8:
                accept_threshold = fix_err
            if (iters == 0 and i == 0) or (len(accept_tv_params) < 100 and iters > 10):
                if iters > 100 and len(accept_tv_params) == 0:
                    # print("uniform", len(accept_tv_params), fix_err, opt_err)
                    if fix_err < opt_err:
                        new_params = initialize_params(fix_a_max, fix_desired_V, fix_a_comf, fix_S_jam, fix_desired_T,
                                                       scale=scale, size=sample_size)
                    else:
                        new_params = generate_uniform_params(size=sample_size)
                elif (iters % 10 == 0) and (len(accept_tv_params) != 0 and iters > 100):
                    if iters % 10 == 0:
                        this_sample_size += 5000
                    # print("augment sample size", len(accept_tv_params), scale, this_sample_size, a_max)
                    new_params = initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=scale, size=this_sample_size)
                else:
                    # print("do nothing", len(accept_tv_params), scale, this_sample_size, a_max)
                    new_params = initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=scale, size=sample_size)
            else:
                try:
                    new_params = generate_new_params(distributions, size=sample_size)
                except:
                    new_params = initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=scale,
                                                   size=sample_size)

            a_n_t_hat = tv_IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, new_params.T[0], new_params.T[1], new_params.T[2],
                                        new_params.T[3], new_params.T[4], 4)

            S_n_t_y_hat = tv_sim_new_spacing(next_pre_x, next_pre_y, ego_x, ego_y, V_n_t, a_n_t_hat)

            err = (S_n_t_y_hat - S_n_t_y) ** 2
            accept_idx = np.where(err < accept_threshold)[0]
            new_accept_tv_params = new_params[accept_idx]

            if len(accept_tv_params) > 100 or (len(accept_tv_params) == 0):
                accept_tv_params = new_accept_tv_params
            else:
                accept_tv_params = np.vstack((accept_tv_params, new_accept_tv_params))
            sum_err = np.sum(err[accept_idx])
            opt_err = np.min(err)
            print(next_v, "v_id", v_id, "frame", i, "iters", iters, "accept params num", accept_tv_params.shape, "fix_err", fix_err, "opt_err", opt_err)
            if len(accept_tv_params) > 100:
                distributions = fit_posterior(accept_tv_params)
                if len(accept_tv_params) > sample_size * 0.95:
                    break
            else:
                # a_max, desired_V, a_comf, S_jam, desired_T
                new_mean = new_params[np.argmin(err)]
                a_max = new_mean[0]
                desired_V = new_mean[1]
                a_comf = new_mean[2]
                S_jam = new_mean[3]
                desired_T = new_mean[4]
            iters += 1

        print(next_v, "v_id", v_id, "frame", i, "tv_err", sum_err/len(accept_tv_params), "opt_err", opt_err, "fix_err", fix_err)
        if len(accept_tv_params) > sample_size * 0.95:
            np.savetxt('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt',
                        np.array(accept_tv_params))
        elif len(accept_tv_params) != 0:
            np.savetxt('/root/AAAI2022_neuma/0803_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt',
                       np.array(accept_tv_params))
        for param_name in ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]:
            # plt.figure()
            # distributions[param_name].summary()
            # plt.savefig('0714_dist_param/' + str(int(v_id)) + '/posterior_figure/' + param_name + '_' + str(int(i)) + '.png')
            print("--------------------------------------------------------------------------------------------")
            for dist in distributions[param_name]:
                print(v_id, param_name, dist["distribution"], dist["fitted_param"], dist["sq_error"])
        tv_sum_err += sum_err/len(accept_tv_params)
        # tv_params.append(accept_tv_params)
        i += 1
    print("all data %d | RMSE: %.4f | a_max: %.4f | desired_V: %.4f | a_comf: %.4f | S_jam: %.4f | desired_T: %.4f | beta: %.3f" % \
        (v_id, rmse_using_all_data, res_param[0], res_param[1], res_param[2], res_param[3], res_param[4], res_param[5]))
    print(str(int(v_id)), "RMSE:", np.sqrt(fix_sum_err / len(new_a)), np.sqrt(tv_sum_err / len(new_a)))
    # print(str(int(v_id)), "mean:", np.mean(np.abs(a-a_hat)), np.std(np.abs(a-a_hat)))
    # tv_params = np.array(tv_params)
    # print(str(int(v_id)), "tv params:", tv_params.shape)

    # f = open('0623_res_tv_desired_v/prior' + str(int(v_id)) + '.pkl', 'wb')
    # pickle.dump(prior, f)
    # f.close()


def get_data_with_pos():
    f = open('all_data_for_cf_model_w_t_pre_info_pos_1101.pkl', 'rb')
    all_data_for_cf_model = pickle.load(f)
    f.close()

    next_vs = []
    v_ids = []
    all_cf_datas = []
    next_v = 1
    for v_id, all_cf_data in all_data_for_cf_model.items():
        print("-------------------------------------------------------------------------------------------------")
        print(str(next_v) + 'th vehicle with id ' + str(v_id))
        next_vs.append(next_v)
        v_ids.append(v_id)
        next_v += 1
        # [delta_v_l, space_hw_l, ego_v_l, a_l]
        delta_V_n_t = np.array(all_cf_data[0])
        S_n_t = np.array(all_cf_data[1])
        V_n_t = np.array(all_cf_data[2])
        a = np.array(all_cf_data[3])
        t = np.array(all_cf_data[4])
        pre_v = np.array(all_cf_data[5])
        pre_tan_acc = np.array(all_cf_data[6])
        pre_lat_acc = np.array(all_cf_data[7])
        pre_v_id = np.array(all_cf_data[8])
        ego_x = np.array(all_cf_data[9])
        ego_y = np.array(all_cf_data[10])
        pre_x = np.array(all_cf_data[11])
        pre_y = np.array(all_cf_data[12])

        print(len(a), np.mean(np.abs(a)))
        print(len(pre_v), np.mean(pre_v))
        print(len(pre_tan_acc), np.mean(np.abs(pre_tan_acc)))
        print(len(pre_lat_acc), np.mean(np.abs(pre_lat_acc)))

        data_array = np.array([delta_V_n_t, S_n_t, V_n_t, a, ego_x, ego_y, pre_x, pre_y, pre_v_id, t]).T
        data_array = data_array[data_array[:, -1].argsort()]
        t = np.array(data_array[:, -1])
        # data_array = data_array[:, 0:-1]
        segs = []
        this_seg = []
        for i in range(len(data_array) - 1):
            current_t = data_array[i][-1]
            next_t = data_array[i + 1][-1]
            current_pre_v_id = data_array[i][-2]
            next_pre_v_id = data_array[i + 1][-2]
            if np.abs(next_t - current_t - 0.04) < 0.0001 and np.abs(current_pre_v_id - next_pre_v_id) < 0.0001:
                this_seg.append(np.append(data_array[i], i))
                if i == len(data_array) - 2:
                    this_seg.append(np.append(data_array[i + 1], i + 1))
                    this_seg = np.array(this_seg)
                    if len(this_seg) > 1:
                        print(this_seg.shape)
                        segs.append(this_seg)
                    break
                continue
            else:
                this_seg.append(np.append(data_array[i], i))
                this_seg = np.array(this_seg)
                if len(this_seg) > 1:
                    print(this_seg.shape)
                    segs.append(this_seg)
                this_seg = []
        print(len(segs))

        new_delta_V_n_t = []
        new_S_n_t = []
        check_S_n_t = []
        new_V_n_t = []
        new_S_n_t_y = []
        new_ego_x = []
        new_ego_y = []
        new_next_pre_x = []
        new_next_pre_y = []
        new_frame_id = []
        sim_S_n_t_y = []
        new_a = []
        clean_a = []
        diff_s = []
        diff_a = []
        for seg in segs:
            for i in range(len(seg) - 1):
                new_delta_V_n_t.append(seg[i][0])
                new_S_n_t.append(seg[i][1])
                check_S_n_t.append(np.sqrt((seg[i][6] - seg[i][4]) ** 2 + (seg[i][7] - seg[i][5]) ** 2))
                new_V_n_t.append(seg[i][2])
                new_a.append(seg[i][3])
                sim_spacing = sim_new_spacing(seg[i + 1][6], seg[i + 1][7], seg[i][4], seg[i][5], seg[i][2], seg[i][3])
                cal_a = cal_new_a(seg[i + 1][6], seg[i + 1][7], seg[i][4], seg[i][5], seg[i][2], seg[i + 1][1])
                sim_S_n_t_y.append(sim_spacing)
                clean_a.append(cal_a)
                new_S_n_t_y.append(seg[i + 1][1])
                new_ego_x.append(seg[i][4])
                new_ego_y.append(seg[i][5])
                new_next_pre_x.append(seg[i + 1][6])
                new_next_pre_y.append(seg[i + 1][7])
                diff_s.append(np.abs(seg[i + 1][1] - sim_spacing))
                diff_a.append(np.abs(seg[i][3] - cal_a))
                new_frame_id.append(seg[i][-1])

        if not data_array.shape[0] - 2 == new_frame_id[-1]:
            print("error", data_array.shape, new_frame_id[-1])
            time.sleep(5)
        data_array = np.array([new_delta_V_n_t, new_S_n_t, new_V_n_t, new_a, new_S_n_t_y, new_ego_x, new_ego_y, new_next_pre_x, new_next_pre_y, new_frame_id]).T
        # print("spacing", np.mean(new_S_n_t_y), np.mean(new_S_n_t), np.mean(check_S_n_t),
        #       np.mean(np.array(new_S_n_t_y) - np.array(new_S_n_t)),
        #       np.mean(diff_s), np.mean(diff_a))
        print(data_array.shape)
        all_cf_datas.append(data_array)

    return next_vs, v_ids, all_cf_datas


if __name__ == "__main__":
    # a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    x0 = (1.0, 20, 0.5, 2, 2, 4)
    next_v = 1
    # req_v_id = int(sys.argv[1])
    print("Start!")
    next_vs, v_ids, all_cf_datas = get_data_with_pos()
    print("next_vs", np.array(next_vs).shape)
    print("v_ids", np.array(v_ids).shape)
    print("all_cf_datas shape:", np.array(all_cf_datas).shape)

    # cal_tv_params(next_vs[0], v_ids[0], all_cf_datas[0])
    # exit()
    pool = Pool(nodes=72)
    pool.map(cal_tv_params, next_vs, v_ids, all_cf_datas)
    pool.close()
    pool.join()
