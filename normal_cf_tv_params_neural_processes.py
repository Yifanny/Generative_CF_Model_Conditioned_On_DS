import numpy as np
from scipy.optimize import minimize
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


def context_target_split(b_x, b_y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)
    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)
    num_context : int
        Number of context points.
    num_extra_target : int
        Number of additional target points.
    """
    x_context = []
    y_context = []
    x_target = []
    y_target = []
    for i in range(len(b_x)):
        x = np.array(b_x[i])
        y = np.array(b_y[i]).reshape(len(b_y[i]), 1)
        # print(x.shape, y.shape)
        num_points = x.shape[0]
        # Sample locations of context and target points
        # print(num_points, num_context, num_extra_target, num_context + num_extra_target)
        if num_context + num_extra_target < num_points:
            locations = np.random.choice(num_points,
                                        size=num_context + num_extra_target,
                                        replace=False)
        else:
            locations = np.random.choice(num_points,
                                         size=num_context + num_extra_target,
                                         replace=True)
            for j in range(len(locations)):
                if locations[j] > num_points:
                    while True:
                        new_loc = np.random.choice(locations, size=1)
                        if new_loc < num_points:
                            locations[j] = new_loc
                            break
        x_context.append(x[locations[:num_context], :])
        y_context.append(y[locations[:num_context], :])
        x_target.append(x[locations, :])
        y_target.append(y[locations, :])
    x_context = np.array(x_context)
    y_context = np.array(y_context)
    x_target = np.array(x_target)
    y_target = np.array(y_target)
    # print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)
    return x_context, y_context, x_target, y_target


import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from random import randint
from torch.distributions.kl import kl_divergence


class DeterministicEncoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    y_dim : int
        Dimension of y values.
    h_dim : int
        Dimension of hidden layer.
    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(DeterministicEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)
        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class LatentEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.
    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim):
        super(LatentEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim

        self.xy_to_hidden = nn.Linear(x_dim + y_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, x, y, batch_size, num_points):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)
        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        hidden = torch.relu(self.xy_to_hidden(input_pairs))
        hidden = hidden.view(batch_size, num_points, self.r_dim)
        hidden = torch.mean(hidden, dim=1)
        mu = torch.relu(self.hidden_to_mu(hidden))
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


# constrained output
class activation(nn.Module):
    def __init__(self, a_max_n_boundary = [-5.0, 5.0]):
        super().__init__()
        self.a_max_n_boundary = a_max_n_boundary

    def forward(self, inputs):
        for i in range(len(inputs)):
            if inputs[i] < self.a_max_n_boundary[0]:
                inputs[i] = self.a_max_n_boundary[0]
            elif inputs[i] > self.a_max_n_boundary[1]:
                inputs[i] = self.a_max_n_boundary[1]

        return inputs


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layer.
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim, r_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + z_dim + r_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)
        self.constrain_output = activation()

    def forward(self, x, z, r):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        z : torch.Tensor
            Shape (batch_size, z_dim)
        r : torch.Tensor
            Shape (batch_size, r_dim)
        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        r = r.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x, z, and r to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        r_flat = r.view(batch_size * num_points, self.r_dim)
        # print(x_flat.size(), z_flat.size(), r_flat.size())
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat, r_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        mu = self.constrain_output(mu)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma


class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.deterministic_encoder = DeterministicEncoder(x_dim, y_dim, h_dim, r_dim)
        self.latent_encoder = LatentEncoder(x_dim, y_dim, r_dim, z_dim)
        self.decoder = Decoder(x_dim, z_dim, h_dim, y_dim, r_dim)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.
        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def deterministic_rep(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.deterministic_encoder(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # print("deterministic encoder r_i size", r_i.size())
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return deterministic representation
        return r

    def latent_rep(self, x, y):
        """
                Maps (x, y) pairs into the mu and sigma parameters defining the normal
                distribution of the latent variables z.
                Parameters
                ----------
                x : torch.Tensor
                    Shape (batch_size, num_points, x_dim)
                y : torch.Tensor
                    Shape (batch_size, num_points, y_dim)
                """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Return parameters of latent representation
        mu, sigma = self.latent_encoder(x_flat, y_flat, batch_size, num_points)
        return mu, sigma

    def forward(self, x_context, y_context, x_target, y_target=None, main_train=True):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        if main_train:
            batch_size, num_context, x_dim = x_context.size()
            _, num_target, _ = x_target.size()
            _, _, y_dim = y_context.size()

        if self.training:
            if main_train:
                # Encode target and context (context needs to be encoded to
                # calculate kl term)
                mu_target, sigma_target = self.latent_rep(x_target, y_target)
                mu_context, sigma_context = self.latent_rep(x_context, y_context)
                # Sample from encoded distribution using reparameterization trick
                q_target = Normal(mu_target, sigma_target)
                q_context = Normal(mu_context, sigma_context)
                z_sample = q_target.rsample()
                r = self.deterministic_rep(x_context, y_context)
                # Get parameters of output distribution
                # print("x_target size", x_target.size())
                # print("z_sample size", z_sample.size())
                # print("r size", r.size())
                y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample, r)
                p_y_pred = Normal(y_pred_mu, y_pred_sigma)

                return p_y_pred, q_target, q_context, y_pred_mu
            else:
                rs = []
                for i in range(len(x_context)):
                    x = x_context[i]
                    y = y_context[i]
                    x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0)
                    y = torch.from_numpy(y).type(torch.FloatTensor).unsqueeze(0)
                    rs.append(self.deterministic_rep(x, y).detach().numpy().reshape(10,))
                return np.array(rs)

        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.latent_rep(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred


class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.
    Parameters
    ----------
    device : torch.device
    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance
    optimizer : one of torch.optim optimizers
    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.
    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.
    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=10):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs):
        """
        Trains Neural Process.
        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance
        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            epoch_loss_n = 0
            for i, data in data_loader.items():
                for _ in range(10):      # try with different number of context points
                    self.optimizer.zero_grad()

                    # Sample number of context and target points
                    num_context = randint(*self.num_context_range)
                    num_extra_target = randint(*self.num_extra_target_range)

                    # Create context and target points and apply neural process
                    x, y = data
                    # print(np.array(x).shape, np.array(y).shape, np.array(ds).shape)
                    x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_extra_target)
                    x_context = torch.from_numpy(x_context).type(torch.FloatTensor)
                    y_context = torch.from_numpy(y_context).type(torch.FloatTensor)
                    x_target = torch.from_numpy(x_target).type(torch.FloatTensor)
                    y_target = torch.from_numpy(y_target).type(torch.FloatTensor)
                    p_y_pred, q_target, q_context, y_pred_mu = self.neural_process(x_context, y_context, x_target, y_target, True)

                    loss = self._loss(p_y_pred, y_target, q_target, q_context, y_pred_mu)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_loss_n += 1

                self.steps += 1

                # if self.steps % self.print_freq == 0:
                batch_size, num_points, _ = y_pred_mu.size()
                y_pred_mu = y_pred_mu.view(batch_size * num_points, )
                y_target = y_target.view(batch_size * num_points, )
                print(y_pred_mu.size(), y_target.size())
                print("iteration {} | loss {:.3f} | train accuracy {:.3f}".format(self.steps, loss.item(),
                       mean_squared_error(y_pred_mu.detach().numpy(), y_target.detach().numpy())))
                logging.info("iteration {} | loss {:.3f} | train accuracy {:.3f}".format(self.steps, loss.item(),
                       mean_squared_error(y_pred_mu.detach().numpy(), y_target.detach().numpy())))

            logging.info("Avg_loss: {}".format(epoch_loss / epoch_loss_n))
            self.epoch_loss_history.append(epoch_loss / epoch_loss_n)
            print("Epoch: {}, Avg_loss: {}, Min_loss: {}".format(epoch, epoch_loss / epoch_loss_n,
                                                             min(self.epoch_loss_history)))
        return epoch_loss / epoch_loss_n


    def _loss(self, p_y_pred, y_target, q_target, q_context, y_pred_mu):
        """
        Computes Neural Process loss.
        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.
        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)
        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.
        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).mean()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).mean()
        # reconstruction error
        batch_size, num_points, _ = y_pred_mu.size()
        y_pred_mu = y_pred_mu.view(batch_size * num_points, )
        y_target = y_target.view(batch_size * num_points, )
        recon = mean_squared_error(y_pred_mu.detach().numpy(), y_target.detach().numpy(), squared=True) * 10
        return -log_likelihood + kl + recon


f = open('all_data_for_cf_model_w_t_pre_info_1101.pkl', 'rb')
all_data_for_cf_model = pickle.load(f)
f.close()

train_x = []
train_y = []
train_ds = []
test_x = []
test_y = []

next_v = 1
all_a = []
for v_id, all_cf_data in all_data_for_cf_model.items():
    # print("------------------------------------------------------------------------------------------------------")
    # print(str(next_v) + 'th vehicle with id ' + str(v_id))

    # [delta_v_l, space_hw_l, ego_v_l, a_l] data
    delta_V_n_t = np.array(all_cf_data[0])
    S_n_t = np.array(all_cf_data[1])
    V_n_t = np.array(all_cf_data[2])
    a = np.array(all_cf_data[3])
    t = np.array(all_cf_data[4])
    pre_v = np.array(all_cf_data[5])
    pre_tan_acc = np.array(all_cf_data[6])
    pre_lat_acc = np.array(all_cf_data[7])

    all_a.extend(list(all_cf_data[3]))

    data_array = np.array([delta_V_n_t, S_n_t, V_n_t, pre_v, pre_tan_acc, pre_lat_acc, a, t]).T
    data_array = data_array[data_array[:, -1].argsort()]
    # print(data_array.shape)

    # if not next_v in train:
    #     test_x.append(data_array[:, 0:-2])
    #     test_y.append(data_array[:, -2])
    #     print(test_x[-1].shape, test_y[-1].shape)
    # else:
    if len(train_x) >= 200:
        test_x.append(data_array[:, 0:-2])
        test_y.append(data_array[:, -2])
        print(test_x[-1].shape, test_y[-1].shape)
        continue

    train_x.append(data_array[:, 0:-2])
    train_y.append(data_array[:, -2])
    print(train_x[-1].shape, train_y[-1].shape)

    next_v += 1


print(len(train_x), len(train_y), len(train_ds))
print(np.array(train_ds).shape)
print(np.array(train_ds))
# print(len(test_x), len(test_y))


from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from time import strftime


def get_dataloader(x, y, batch_size, shuffle=False):
    data_loader = {}
    id = 0
    ids = []
    if shuffle:
        l = list(range(len(x)))
        while True:
            s_l = np.random.choice(l, batch_size, replace=False)
            ids.append(s_l)
            for i in s_l:
                l.remove(i)
            if len(l) <= batch_size:
                ids.append(np.array(l))
                break
    else:
        l = list(range(len(x)))
        for i in range(0, len(x), batch_size):
            ids.append(l[i: i + batch_size])

    x = np.array(x)
    y = np.array(y)
    for s_ids in ids:
        data_loader[id] = (x[s_ids], y[s_ids])
        id += 1
    return data_loader


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Create a folder to store experiment results
timestamp = strftime("%Y-%m-%d")
directory = "neural_processes_results_{}".format(timestamp)
if not os.path.exists(directory):
    os.makedirs(directory)

batch_size = 8
x_dim = 6
y_dim = 1
r_dim = 10
h_dim = 128
z_dim = 1
num_context_range = (300, 500)
num_extra_target_range = (100, 200)
epochs = 500
lr = 0.001

data_loader = get_dataloader(train_x, train_y, batch_size, shuffle=False)

cf_np = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
# cf_np.load_state_dict(torch.load("/root/AAAI2022_neuma/neural_processes_results_2021-07-23/922model4.015029139816761.pt"))
print(cf_np)

optimizer = torch.optim.Adam(cf_np.parameters(), lr=lr)

np_trainer = NeuralProcessTrainer(device, cf_np, optimizer,
                                  num_context_range, num_extra_target_range,
                                  print_freq=100)

logging.basicConfig(filename=directory + '/training.log',
                    format='%(asctime)s :  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p',
                    level=10)


for epoch in range(epochs):
    print("Epoch {}".format(epoch + 1))
    logging.info("Epoch {}".format(epoch + 1))
    loss = np_trainer.train(data_loader, 1)
    # Save losses at every epoch
    # with open(directory + '/losses.json', 'w') as f:
    #     json.dump(np_trainer.epoch_loss_history, f)
    # Save model at every epoch
    torch.save(np_trainer.neural_process.state_dict(), directory + '/' + str(int(epoch))+'model'+str(loss)+'.pt')
    data_loader = get_dataloader(train_x, train_y, batch_size, shuffle=True)
    if epoch % 100 == 0 and epoch > 0:
        if epoch % 1000 == 0:
            lr = 0.001
        else:
            lr *= 0.9
        optimizer = torch.optim.Adam(cf_np.parameters(), lr=lr)