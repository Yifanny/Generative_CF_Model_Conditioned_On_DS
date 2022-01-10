import numpy as np
from scipy.optimize import minimize, curve_fit
import scipy.stats
import pickle, os, random, time
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.metrics import mean_squared_error
import logging


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
    for i in range(len(delta_V_n_t)):
        desired_S_n = desired_space_hw(S_jam_n[i], V_n_t[i], desired_T_n[i], delta_V_n_t[i], a_max_n[i], a_comf_n[i])
        a_n_t.append(a_max_n[i] * (1 - (V_n_t[i] / desired_V_n[i]) ** beta - (desired_S_n / S_n_t[i]) ** 2))

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


a_max_n_boundary = [0.1, 2.5]
desired_V_n_boundary = [1, 40]
a_comf_n_boundary = [0.1, 5]
S_jam_n_boundary = [0.1, 10]
desired_T_n_boundary = [0.1, 5]
boundary = [a_max_n_boundary, desired_V_n_boundary, a_comf_n_boundary, S_jam_n_boundary, desired_T_n_boundary]


def save_pkl_file(file, var):
    pkl_file = open(file, 'wb')
    pickle.dump(var, pkl_file)
    pkl_file.close()


def read_pkl_file(file):
    pkl_file = open(file, 'rb')
    var = pickle.load(pkl_file)
    pkl_file.close()
    return var



f = open('all_data_for_cf_model_w_t_pre_info_1101.pkl', 'rb')
all_data_for_cf_model = pickle.load(f)
f.close()

from sklearn.preprocessing import normalize, minmax_scale
from sklearn.metrics import mean_squared_error

all_r_c = np.loadtxt("0714_dist_param/200_r_c.txt")
print(all_r_c.shape)
agg_indexes = np.loadtxt("0714_dist_param/200_1d_agg_indexes.txt")
print(agg_indexes.shape)
# f = open("0714_dist_param/non_scaled_original_agg_matrix.pkl", "rb")
# mean_l, std_l, per25_l, per75_l, diff_seq_l = pickle.load(f)
# f.close()
# agg_indexes = np.hstack((normalize(mean_l, axis=1), normalize(std_l, axis=1), normalize(per25_l, axis=1),
#                          normalize(per75_l, axis=1), normalize(diff_seq_l, axis=1)))
# print(agg_indexes.shape)
# all_r_c = normalize(all_r_c, axis=1)


# agg_indexes = minmax_scale(agg_indexes, axis=0)
# all_r_c = minmax_scale(all_r_c, axis=0)

print(np.mean(all_r_c, axis=0))

from sklearn.decomposition import PCA, KernelPCA, LatentDirichletAllocation, FactorAnalysis, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau

pca = PCA(n_components=1)
# pca = LinearDiscriminantAnalysis()
pca.fit(all_r_c, agg_indexes)
all_r_c_pca = pca.transform(all_r_c).reshape(200, )
# all_r_c_pca_norm = normalize(all_r_c_pca).reshape(200, )
print(all_r_c_pca)

print("first r_c after pca:", all_r_c_pca[0])
print("first agg_index:", agg_indexes[0])


r_c_range = (np.mean(all_r_c_pca) - 3 * np.std(all_r_c_pca), np.mean(all_r_c_pca) + 3 * np.std(all_r_c_pca))
agg_indexes_range = (np.mean(agg_indexes) - 3 * np.std(agg_indexes), np.mean(agg_indexes) + 3 * np.std(agg_indexes))
new_r_c_pca = []
new_r_c = []
new_agg_indexes = []
work = 0
print(r_c_range)
print(agg_indexes_range)
for i in range(len(all_r_c_pca)):
    if r_c_range[0] < all_r_c_pca[i] < r_c_range[1] and agg_indexes_range[0] < agg_indexes[i] < agg_indexes_range[1]:
        new_r_c_pca.append(all_r_c_pca[i])
        new_agg_indexes.append(agg_indexes[i])
        new_r_c.append(all_r_c[i])
    else:
        print(all_r_c_pca[i], agg_indexes[i])


def func(x, a, b):
    return a * x + b

new_r_c_pca = np.array(new_r_c_pca)
new_r_c = np.array(new_r_c)
new_agg_indexes = np.array(new_agg_indexes)
print(new_r_c_pca.shape, new_agg_indexes.shape, new_r_c.shape)

popt, pcov = curve_fit(func, new_agg_indexes, new_r_c_pca, maxfev=100000)

print(mean_squared_error(func(new_agg_indexes, *popt), new_r_c_pca, squared=False))
print(popt)

print(pearsonr(new_agg_indexes, new_r_c_pca))
print(spearmanr(new_agg_indexes, new_r_c_pca))
print(kendalltau(new_agg_indexes, new_r_c_pca))

recon_r_c = func(new_agg_indexes, *popt).reshape(149, 1) * pca.components_ + pca.mean_
print(pca.components_)
print(pca.mean_)
print(func(new_agg_indexes, *popt))
print(mean_squared_error(recon_r_c, new_r_c, squared=False))

# print("non existed agg. indexes:", agg_indexes[0]-2, agg_indexes[0]+2)
non_existed_r_c = func(np.array([0, 5]), *popt).reshape(2, 1) * pca.components_ + pca.mean_
print(non_existed_r_c)
np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/new_ds.txt", non_existed_r_c)

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(23, 6))
ax = plt.subplot(1, 3, 1)
plt.hist(all_r_c_pca)
ax.set_title("Reduced r after PCA")

ax = plt.subplot(1, 3, 2)
plt.hist(agg_indexes)
ax.set_title("Aggressiveness index")

ax = plt.subplot(1, 3, 3)
plt.plot(new_agg_indexes, func(new_agg_indexes, *popt))
plt.scatter(new_agg_indexes, new_r_c_pca)
plt.xlabel("agg. index")
plt.ylabel("reduced r")
ax.set_title("Relations between aggressiveness\nindexes and reduced r")
plt.savefig("/root/AAAI2022_neuma/0714_dist_param/map_agg_2_r/agg_indxes_pca_r_c.png")
exit()


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


net = Net()
net.apply(weight_init)
print(net)

lr = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_func = torch.nn.MSELoss()

x = torch.from_numpy(agg_indexes).type(torch.FloatTensor)
y = torch.from_numpy(all_r_c).type(torch.FloatTensor)
print(x.size(), y.size())

t = 0
while True:
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 100 == 0:
        print("Epoch:", t, " | Loss:", loss.data.numpy())
    t += 1

    # if loss.data.numpy() < 0.5:
    #     break
    if t > 10000:
        break

torch.save(net.state_dict(), "0714_dist_param/map_agg_2_r/map_40d_agg_2_5d_r.pt")

exit()


# corrcoef = np.corrcoef(agg_indexes.T)
# a_max, desired_v, a_comf, S_jam, deisred_T
# mean q1 q3 inc_mean inc_std dec_mean dec_std
param_name = ["a_max", "desired_v", "a_comf", "S_jam", "deisred_T"]
stat_name = ["mean", "std"]





for i in range(5):
    plt.figure()
    plt.hist(agg_indexes[:, i])
    plt.savefig("0714_dist_param/map_agg_2_r/" + param_name[i] + "_" + stat_name[0] + ".png")
    plt.figure()
    plt.hist(agg_indexes[:, i + 5])
    plt.savefig("0714_dist_param/map_agg_2_r/" + param_name[i] + "_" + stat_name[1] + ".png")
exit()


def fun(args):
    r_ci, mean, std, n = args
    n1, n2 = n
    return lambda x: np.sum((r_ci - (x[0] * mean ** n1 + x[1] * std ** n2 + x[2])) ** 2)/len(r_ci)


all_r_c = all_r_c.T
n_l = []
exp_l = [-3, -2, -1, 1, 2, 3]
for n1 in exp_l:
    for n2 in exp_l:
        n_l.append((n1, n2))

err = np.zeros((5, 5))
best_n = {}
max_iters = 10
for i in range(5):
    for j in range(5):
        for n in n_l:
            data = (all_r_c[i], np.array(agg_indexes[:, j]), np.array(agg_indexes[:, j + 5]), n)
            for _ in range(max_iters):
                x0 = np.asarray((np.random.randn(), np.random.randn(), np.random.randn()))
                res = minimize(fun(data), x0)
                if res.success:
                    break
            if res.success:
                if res.fun < err[i][j] or err[i][j] == 0:
                    err[i][j] = res.fun
                    best_n[(i, j)] = n
                    print("r_c_" + str(int(i)), param_name[j], n, res.fun)
print(err)
print(best_n)
exit()


p = 1

plt.figure()
plt.scatter(all_r_c[:, 0], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 0], agg_indexes[:, 2], marker="^", color="skyblue")
z1 = np.polyfit(all_r_c[:, 0], agg_indexes[:, 1], p)
p1 = np.poly1d(z1)
plt.plot(all_r_c[:, 0], p1(all_r_c[:, 0]), color="lightcoral")
z1 = np.polyfit(all_r_c[:, 0], agg_indexes[:, 2], p)
p1 = np.poly1d(z1)
plt.plot(all_r_c[:, 0], p1(all_r_c[:, 0]), color="skyblue")
plt.savefig("0714_dist_param/100_2d_agg_indexes_r_c1_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 1], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 1], agg_indexes[:, 2], marker="^", color="skyblue")
z2 = np.polyfit(all_r_c[:, 1], agg_indexes[:, 1], p)
p2 = np.poly1d(z2)
plt.plot(all_r_c[:, 1], p2(all_r_c[:, 1]), color="lightcoral")
z2 = np.polyfit(all_r_c[:, 1], agg_indexes[:, 2], p)
p2 = np.poly1d(z2)
plt.plot(all_r_c[:, 1], p2(all_r_c[:, 1]), color="skyblue")
plt.savefig("0714_dist_param/100_2d_agg_indexes_r_c2_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 2], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 2], agg_indexes[:, 2], marker="^", color="skyblue")
z3 = np.polyfit(all_r_c[:, 2], agg_indexes[:, 1], p)
p3 = np.poly1d(z3)
plt.plot(all_r_c[:, 2], p3(all_r_c[:, 2]), color="lightcoral")
z3 = np.polyfit(all_r_c[:, 2], agg_indexes[:, 2], p)
p3 = np.poly1d(z3)
plt.plot(all_r_c[:, 2], p3(all_r_c[:, 2]), color="skyblue")
plt.savefig("0714_dist_param/100_2d_agg_indexes_r_c3_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 3], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 3], agg_indexes[:, 2], marker="^", color="skyblue")
z4 = np.polyfit(all_r_c[:, 3], agg_indexes[:, 1], p)
p4 = np.poly1d(z4)
plt.plot(all_r_c[:, 3], p4(all_r_c[:, 3]), color="lightcoral")
z4 = np.polyfit(all_r_c[:, 3], agg_indexes[:, 2], p)
p4 = np.poly1d(z4)
plt.plot(all_r_c[:, 3], p4(all_r_c[:, 3]), color="skyblue")
plt.savefig("0714_dist_param/100_2d_agg_indexes_r_c4_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 4], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 4], agg_indexes[:, 2], marker="^", color="skyblue")
z5 = np.polyfit(all_r_c[:, 4], agg_indexes[:, 1], p)
p5 = np.poly1d(z5)
plt.plot(all_r_c[:, 4], p5(all_r_c[:, 4]), color="lightcoral")
z5 = np.polyfit(all_r_c[:, 4], agg_indexes[:, 2], p)
p5 = np.poly1d(z5)
plt.plot(all_r_c[:, 4], p5(all_r_c[:, 4]), color="skyblue")
plt.savefig("0714_dist_param/100_2d_agg_indexes_r_c5_reg.png")

exit()






err = np.zeros((5, 3))
max_iters = 10
best_p = np.zeros((5, 3))

print(agg_indexes.shape)

from sklearn.metrics import mean_squared_error

for p in range(1, 5):
    for i in range(3):
        for j in range(5):
            z1 = np.polyfit(all_r_c[:, j], agg_indexes[:, i + 1], p)
            p1 = np.poly1d(z1)
            mse = mean_squared_error(p1(all_r_c[:, j]), agg_indexes[:, i + 1])
            if err[j][i] == 0:
                err[j][i] = mse
            if mse < err[j][i]:
                err[j][i] = mse
                best_p[j][i] = p
                print("r", j, "H", i, p, mse)
print(err)
print(best_p)
exit()


p = 1

plt.figure()
plt.scatter(all_r_c[:, 0], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 0], agg_indexes[:, 2], marker="^", color="skyblue")
plt.scatter(all_r_c[:, 0], agg_indexes[:, 3], marker="*", color="orange")
z1 = np.polyfit(all_r_c[:, 0], agg_indexes[:, 1], p)
p1 = np.poly1d(z1)
plt.plot(all_r_c[:, 0], p1(all_r_c[:, 0]), color="lightcoral")
z1 = np.polyfit(all_r_c[:, 0], agg_indexes[:, 2], p)
p1 = np.poly1d(z1)
plt.plot(all_r_c[:, 0], p1(all_r_c[:, 0]), color="skyblue")
z1 = np.polyfit(all_r_c[:, 0], agg_indexes[:, 3], p)
p1 = np.poly1d(z1)
plt.plot(all_r_c[:, 0], p1(all_r_c[:, 0]), color="orange")
plt.savefig("0714_dist_param/100_3d_agg_indexes_r_c1_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 1], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 1], agg_indexes[:, 2], marker="^", color="skyblue")
plt.scatter(all_r_c[:, 1], agg_indexes[:, 3], marker="*", color="orange")
z2 = np.polyfit(all_r_c[:, 1], agg_indexes[:, 1], p)
p2 = np.poly1d(z2)
plt.plot(all_r_c[:, 1], p2(all_r_c[:, 1]), color="lightcoral")
z2 = np.polyfit(all_r_c[:, 1], agg_indexes[:, 2], p)
p2 = np.poly1d(z2)
plt.plot(all_r_c[:, 1], p2(all_r_c[:, 1]), color="skyblue")
z2 = np.polyfit(all_r_c[:, 1], agg_indexes[:, 3], p)
p2 = np.poly1d(z2)
plt.plot(all_r_c[:, 1], p2(all_r_c[:, 1]), color="orange")
plt.savefig("0714_dist_param/100_3d_agg_indexes_r_c2_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 2], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 2], agg_indexes[:, 2], marker="^", color="skyblue")
plt.scatter(all_r_c[:, 2], agg_indexes[:, 3], marker="*", color="orange")
z3 = np.polyfit(all_r_c[:, 2], agg_indexes[:, 1], p)
p3 = np.poly1d(z3)
plt.plot(all_r_c[:, 2], p3(all_r_c[:, 2]), color="lightcoral")
z3 = np.polyfit(all_r_c[:, 2], agg_indexes[:, 2], p)
p3 = np.poly1d(z3)
plt.plot(all_r_c[:, 2], p3(all_r_c[:, 2]), color="skyblue")
z3 = np.polyfit(all_r_c[:, 2], agg_indexes[:, 3], p)
p3 = np.poly1d(z3)
plt.plot(all_r_c[:, 2], p3(all_r_c[:, 2]), color="orange")
plt.savefig("0714_dist_param/100_3d_agg_indexes_r_c3_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 3], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 3], agg_indexes[:, 2], marker="^", color="skyblue")
plt.scatter(all_r_c[:, 3], agg_indexes[:, 3], marker="*", color="orange")
z4 = np.polyfit(all_r_c[:, 3], agg_indexes[:, 1], p)
p4 = np.poly1d(z4)
plt.plot(all_r_c[:, 3], p4(all_r_c[:, 3]), color="lightcoral")
z4 = np.polyfit(all_r_c[:, 3], agg_indexes[:, 2], p)
p4 = np.poly1d(z4)
plt.plot(all_r_c[:, 3], p4(all_r_c[:, 3]), color="skyblue")
z4 = np.polyfit(all_r_c[:, 3], agg_indexes[:, 3], p)
p4 = np.poly1d(z4)
plt.plot(all_r_c[:, 3], p4(all_r_c[:, 3]), color="orange")
plt.savefig("0714_dist_param/100_3d_agg_indexes_r_c4_reg.png")

plt.figure()
plt.scatter(all_r_c[:, 4], agg_indexes[:, 1], marker=".", color="lightcoral")
plt.scatter(all_r_c[:, 4], agg_indexes[:, 2], marker="^", color="skyblue")
plt.scatter(all_r_c[:, 4], agg_indexes[:, 3], marker="*", color="orange")
z5 = np.polyfit(all_r_c[:, 4], agg_indexes[:, 1], p)
p5 = np.poly1d(z5)
plt.plot(all_r_c[:, 4], p5(all_r_c[:, 4]), color="lightcoral")
z5 = np.polyfit(all_r_c[:, 4], agg_indexes[:, 2], p)
p5 = np.poly1d(z5)
plt.plot(all_r_c[:, 4], p5(all_r_c[:, 4]), color="skyblue")
z5 = np.polyfit(all_r_c[:, 4], agg_indexes[:, 3], p)
p5 = np.poly1d(z5)
plt.plot(all_r_c[:, 4], p5(all_r_c[:, 4]), color="orange")
plt.savefig("0714_dist_param/100_3d_agg_indexes_r_c5_reg.png")

exit()




def fun(args):
    x1, x2, x3, x4, x5, y, n = args
    n1, n2, n3, n4, n5 = n
    return lambda x: np.sqrt(np.sum((y - (x[0] * x1 ** n1 + x[1] * x2 ** n2 + x[2] * x3 ** n3 + x[3] * x4 ** n4 + x[4] * x5 ** n5)) ** 2)/len(y))


all_r_c = all_r_c.T
n_l = []
exp_l = [-2, -1, 1, 2]
for n1 in exp_l:
    for n2 in exp_l:
        for n3 in exp_l:
            for n4 in exp_l:
                for n5 in exp_l:
                    n_l.append((n1, n2, n3, n4, n5))

for i in range(3):
    for n in n_l:
        data = (all_r_c[0], all_r_c[1], all_r_c[2], all_r_c[3], all_r_c[4], np.array(agg_indexes[:, 1 + i]), n)
        for _ in range(max_iters):
            x0 = np.asarray((np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()))
            res = minimize(fun(data), x0)
            if res.success:
                break
        if res.success:
            if res.fun < err[i]:
                err[i] = res.fun
                best_n[i] = n
                print(i, n, res.fun)
print(err)
print(best_n)
exit()



import itertools
print(list(itertools.combinations([0, 1, 2, 3, 4], 4)))


def fun3(args):
    x1, x2, x3, y, n = args
    n1, n2, n3 = n
    return lambda x: np.sqrt(np.sum((y - (x[0] * x1 ** n1 + x[1] * x2 ** n2 + x[2] * x3 ** n3)) ** 2)/len(y))


def fun4(args):
    x1, x2, x3, x4, y, n = args
    n1, n2, n3, n4 = n
    return lambda x: np.sqrt(np.sum((y - (x[0] * x1 ** n1 + x[1] * x2 ** n2 + x[2] * x3 ** n3 + x[3] * x4 ** n4)) ** 2)/len(y))


for l in list(itertools.combinations([0, 1, 2, 3, 4], 4)):
    err = 999
    best_n = None
    print("-------------------------------------------------------")
    for n1 in np.arange(-5, 5):
        for n2 in np.arange(-5, 5):
            for n3 in np.arange(-5, 5):
                for n4 in np.arange(-5, 5):
                    n = (n1, n2, n3, n4)
                    data = (all_r_c[l[0]], all_r_c[l[1]], all_r_c[l[2]], all_r_c[l[3]], np.array(agg_indexes[:, 1]), n)
                    for _ in range(max_iters):
                        x0 = np.asarray((np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()))
                        res = minimize(fun4(data), x0)
                        if res.success:
                            break
                    if err > res.fun:
                        print(l, n, res.fun)
                        best_n = n
                        err = res.fun
exit()



n = (2, 1, 2, -1, 2)
print(np.mean(np.abs(np.array(agg_indexes[:, 1]))))
print(n)
data = (all_r_c[0], all_r_c[1], all_r_c[2], all_r_c[3], all_r_c[4], np.array(agg_indexes[:, 1]), n)
while True:
    x0 = np.asarray((np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()))
    res = minimize(fun(data), x0)
    if res.success:
        break
print(res)
exit()


def fun2(args):
    x1, x2, y, n = args
    n1, n2 = n
    return lambda x: np.sqrt(np.sum((y - (x[0] * x1 ** n1 + x[1] * x2 ** n2)) ** 2)/len(y))


for i in range(5):
    for j in range(i + 1, 5):
        err = 999
        best_n = None
        print("-------------------------------------------------------")
        for n1 in np.arange(-5, 5, 0.5):
            for n2 in np.arange(-5, 5, 0.5):
                n = (n1, n2)
                data = (all_r_c[i], all_r_c[j], np.array(agg_indexes[:, 1]), n)
                for _ in range(max_iters):
                    x0 = np.asarray((np.random.randn(), np.random.randn()))
                    res = minimize(fun2(data), x0)
                    if res.success:
                        break
                if err > res.fun:
                    print(i, j, n, res.fun)
                    best_n = n
                    err = res.fun
exit()





