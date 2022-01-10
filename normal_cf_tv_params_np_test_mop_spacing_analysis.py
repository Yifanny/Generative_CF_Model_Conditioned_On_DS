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
        # self.training = training

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

    def forward(self, x_context, y_context, x_target, y_target=None, given_r=None):
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
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
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
            # At testing time, encode only context
            mu_context, sigma_context = self.latent_rep(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            r = self.deterministic_rep(x_context, y_context)
            # Predict target points based on context
            if given_r is None:
                y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample, r)
            else:
                y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample, given_r)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, y_pred_mu, y_pred_sigma, r, mu_context, sigma_context


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
                for _ in range(1):      # try with different number of context points
                    self.optimizer.zero_grad()

                    # Sample number of context and target points
                    num_context = randint(*self.num_context_range)
                    num_extra_target = randint(*self.num_extra_target_range)

                    # Create context and target points and apply neural process
                    x, y = data
                    # print(np.array(x).shape, np.array(y).shape)
                    x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_extra_target)
                    x_context = torch.from_numpy(x_context).type(torch.FloatTensor)
                    y_context = torch.from_numpy(y_context).type(torch.FloatTensor)
                    x_target = torch.from_numpy(x_target).type(torch.FloatTensor)
                    y_target = torch.from_numpy(y_target).type(torch.FloatTensor)
                    p_y_pred, q_target, q_context, y_pred_mu = self.neural_process(x_context, y_context, x_target, y_target)

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
        recon = mean_squared_error(y_pred_mu.detach().numpy(), y_target.detach().numpy(), squared=False) * 10
        return -log_likelihood + kl + recon


def get_data_with_pos():
    f = open('all_data_for_cf_model_w_t_pre_info_pos_1101.pkl', 'rb')
    all_data_for_cf_model = pickle.load(f)
    f.close()

    next_vs = []
    v_ids = []
    all_cf_datas = []
    next_v = 1
    segs_info = []
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

        data_array = np.array([delta_V_n_t, S_n_t, V_n_t, a, ego_x, ego_y, pre_x, pre_y, pre_v, pre_tan_acc, pre_lat_acc, pre_v_id, t]).T
        data_array = data_array[data_array[:, -1].argsort()]
        t = np.array(data_array[:, -1])
        # data_array = data_array[:, 0:-1]
        segs = []
        this_seg = []
        this_seg_info = []
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
                        this_seg_info.append(this_seg.shape)
                        print(this_seg.shape)
                        segs.append(this_seg)
                    break
                continue
            else:
                this_seg.append(np.append(data_array[i], i))
                this_seg = np.array(this_seg)
                if len(this_seg) > 1:
                    this_seg_info.append(this_seg.shape)
                    print(this_seg.shape)
                    segs.append(this_seg)
                this_seg = []
        print(len(segs))
        segs_info.append(this_seg_info)

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
        new_pre_v = []
        new_pre_tan_acc = []
        new_pre_lat_acc = []
        # clean_a = []
        diff_s = []
        # diff_a = []
        # delta_V_n_t, S_n_t, V_n_t, a, ego_x, ego_y, pre_x, pre_y, pre_v, pre_tan_acc, pre_lat_acc, pre_v_id, t
        for seg in segs:
            for i in range(len(seg) - 1):
                new_delta_V_n_t.append(seg[i][0])
                new_S_n_t.append(seg[i][1])
                check_S_n_t.append(np.sqrt((seg[i][6] - seg[i][4]) ** 2 + (seg[i][7] - seg[i][5]) ** 2))
                new_V_n_t.append(seg[i][2])
                new_a.append(seg[i][3])
                sim_spacing = sim_new_spacing(seg[i + 1][6], seg[i + 1][7], seg[i][4], seg[i][5], seg[i][2], seg[i][3])
                # cal_a = cal_new_a(seg[i + 1][6], seg[i + 1][7], seg[i][4], seg[i][5], seg[i][2], seg[i + 1][1])
                sim_S_n_t_y.append(sim_spacing)
                # clean_a.append(cal_a)
                new_S_n_t_y.append(seg[i + 1][1])
                new_ego_x.append(seg[i][4])
                new_ego_y.append(seg[i][5])
                new_next_pre_x.append(seg[i + 1][6])
                new_next_pre_y.append(seg[i + 1][7])
                diff_s.append(np.abs(seg[i + 1][1] - sim_spacing))
                # diff_a.append(np.abs(seg[i][3] - cal_a))
                new_frame_id.append(seg[i][-1])
                new_pre_v.append(seg[i][8])
                new_pre_tan_acc.append(seg[i][9])
                new_pre_lat_acc.append(seg[i][10])

        if not data_array.shape[0] - 2 == new_frame_id[-1]:
            print("error", data_array.shape, new_frame_id[-1])
            time.sleep(5)
        data_array = np.array(
            [new_delta_V_n_t, new_S_n_t, new_V_n_t, new_a, new_S_n_t_y, new_ego_x, new_ego_y, new_next_pre_x,
             new_next_pre_y, new_pre_v, new_pre_tan_acc, new_pre_lat_acc, new_frame_id]).T
        # print("spacing", np.mean(new_S_n_t_y), np.mean(new_S_n_t), np.mean(check_S_n_t),
        #       np.mean(np.array(new_S_n_t_y) - np.array(new_S_n_t)),
        #       np.mean(diff_s), np.mean(diff_a))
        print(data_array.shape)
        all_cf_datas.append(data_array)

    return next_vs, v_ids, all_cf_datas, segs_info


def cal_ttc(next_v, v_id, all_cf_data):
    # S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_x, next_pre_y, ego_x, ego_y = args

    if os.path.exists('0803_mop_space_dist_param/' + str(int(v_id)) + '/using_all_data.txt'):
        res_param = np.loadtxt('0803_mop_space_dist_param/' + str(int(v_id)) + '/using_all_data.txt')
    else:
        return False, False, False

    fix_a_max = res_param[0]
    fix_desired_V = res_param[1]
    fix_a_comf = res_param[2]
    fix_S_jam = res_param[3]
    fix_desired_T = res_param[4]

    tv_params_mean = np.loadtxt('0803_mop_space_dist_param/' + str(int(v_id)) + '/tv_params_mean.txt')

    for i in range(1, len(all_cf_data)):
        previous_frame = all_cf_data[i - 1]
        current_frame = all_cf_data[i]
        if current_frame[9] - previous_frame[9] != 1:
            p_delta_V_n_t = current_frame[0]
            p_S_n_t = current_frame[1]
            p_V_n_t = current_frame[2]
            p_a_n_t = current_frame[3]
            p_next_S_n_t = current_frame[4]
            p_ego_x = current_frame[5]
            p_ego_y = current_frame[6]
            p_next_pre_x = current_frame[7]
            p_next_pre_y = current_frame[8]
            continue
        delta_V_n_t = current_frame[0]
        S_n_t = current_frame[1]
        V_n_t = current_frame[2]
        a_n_t = current_frame[3]
        next_S_n_t = current_frame[4]
        ego_x = current_frame[5]
        ego_y = current_frame[6]
        next_pre_x = current_frame[7]
        next_pre_y = current_frame[8]
        pre_V_n_t = V_n_t - delta_V_n_t

        if i == 1:
            p_delta_V_n_t = previous_frame[0]
            p_S_n_t = previous_frame[1]
            p_V_n_t = previous_frame[2]
            p_a_n_t = previous_frame[3]
            p_next_S_n_t = previous_frame[4]
            p_ego_x = previous_frame[5]
            p_ego_y = previous_frame[6]
            p_next_pre_x = previous_frame[7]
            p_next_pre_y = previous_frame[8]

        tv_params = tv_params_mean[i]
        a_max, desired_V, a_comf, S_jam, desired_T = tv_params[0], tv_params[1], tv_params[2], tv_params[3], tv_params[
            4]
        a_n_t_hat = IDM_cf_model_for_p(p_delta_V_n_t, p_S_n_t, p_V_n_t, a_max, desired_V, a_comf, S_jam, desired_T, 4)
        tv_sim_V_n_t = p_V_n_t + a_n_t_hat * 0.04

        fix_a_n_t_hat = IDM_cf_model_for_p(p_delta_V_n_t, p_S_n_t, p_V_n_t, fix_a_max, fix_desired_V, fix_a_comf,
                                           fix_S_jam, fix_desired_T, 4)
        fix_sim_V_n_t = p_V_n_t + fix_a_n_t_hat * 0.04

        tv_V_n_t_1 = V_n_t

        ttc = S_n_t / delta_V_n_t
        # fix_ttc =
        # tv_ttc =

        i += 1


def sim_new_spacing(new_pre_x, new_pre_y, old_ego_x, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    return np.sqrt((new_pre_x - old_ego_x) ** 2 + (new_pre_y - old_ego_y) ** 2) - (2 * V_n_t + a_n_t * delta_t) / 2 * delta_t


train_x = []
train_y = []
test_x = []
test_y = []
all_x = []
all_y = []
train = np.random.choice(range(1, 277), 256, replace=False)
next_v = 1
all_a = []

next_vs, v_ids, all_cf_datas, segs_info = get_data_with_pos()

for i in range(len(v_ids)):
    v_id = v_ids[i]
    all_cf_data = all_cf_datas[i].T
    # print("------------------------------------------------------------------------------------------------------")
    # print(str(next_v) + 'th vehicle with id ' + str(v_id))

    # delta_V_n_t, S_n_t, V_n_t, a, S_n_t_y, ego_x, ego_y, next_pre_x, next_pre_y, pre_v, pre_tan_acc, pre_lat_acc, frame_id
    delta_V_n_t = np.array(all_cf_data[0])
    S_n_t = np.array(all_cf_data[1])
    V_n_t = np.array(all_cf_data[2])
    a = np.array(all_cf_data[3])
    S_n_t_y = np.array(all_cf_data[4])
    ego_x = np.array(all_cf_data[5])
    ego_y = np.array(all_cf_data[6])
    next_pre_x = np.array(all_cf_data[7])
    next_pre_y = np.array(all_cf_data[8])
    pre_v = np.array(all_cf_data[9])
    pre_tan_acc = np.array(all_cf_data[10])
    pre_lat_acc = np.array(all_cf_data[11])
    frame_id = np.array(all_cf_data[12])

    v_id = [v_id] * len(a)
    data_array = np.array([v_id, delta_V_n_t, S_n_t, V_n_t, pre_v, pre_tan_acc, pre_lat_acc, S_n_t_y, ego_x, ego_y, next_pre_x, next_pre_y, frame_id, a]).T
    print(data_array.shape)

    if not next_v in train:
        test_x.append(data_array[:, 0:-1])
        test_y.append(data_array[:, -1])
        all_x.append(data_array[:, 0:-1])
        all_y.append(data_array[:, -1])
        print(test_x[-1].shape, test_y[-1].shape)
    else:
        train_x.append(data_array[:, 0:-1])
        train_y.append(data_array[:, -1])
        all_x.append(data_array[:, 0:-1])
        all_y.append(data_array[:, -1])
        print(train_x[-1].shape, train_y[-1].shape)

    next_v += 1

print(len(train_x), len(train_y))
print(len(test_x), len(test_y))


from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from time import strftime


def get_test_dataloader(x, y):
    # v_id, delta_V_n_t, S_n_t, V_n_t, pre_v, pre_tan_acc, pre_lat_acc, S_n_t_y, ego_x, ego_y, next_pre_x, next_pre_y, frame_id, a
    data_loader = {}
    for i in range(len(x)):
        v_id = x[i][0][0]
        print(v_id)
        for_sim_spacing = x[i].T[7::].T
        x[i] = x[i].T[1:7].T
        data_loader[v_id] = ([x[i]], [for_sim_spacing], [y[i]])
    return data_loader


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Create a folder to store experiment results
# timestamp = strftime("%Y-%m-%d")
# directory = "neural_processes_results_{}".format(timestamp)
# if not os.path.exists(directory):
#     os.makedirs(directory)

batch_size = 1
x_dim = 6
y_dim = 1
r_dim = 5
h_dim = 128
z_dim = 1
num_context_range = (300, 500)
num_extra_target_range = (100, 200)
epochs = 2000
lr = 0.001

data_loader = get_test_dataloader(all_x, all_y)
print(len(data_loader))

cf_np = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
cf_np.load_state_dict(torch.load("NP_model.pt"))
print(cf_np)
cf_np.training = False
print(cf_np.training)

# logging.basicConfig(filename=directory + '/test.log',
#                     format='%(asctime)s :  %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S %p',
#                     level=10)
def simulate_agg():
    all_rmse = []
    all_r_c = []
    n = 1
    sim_ttc = []
    sim_spacing = []
    sim_speed = []
    ttc = []
    spacing = []
    speed = []
    a_err = []
    new_seg = False
    direction = 0.9712041389105396
    non_existed_r_c = np.loadtxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/new_ds.txt")
    new_con_r = torch.tensor(non_existed_r_c[0]).type(torch.FloatTensor)
    new_agg_r = torch.tensor(non_existed_r_c[1]).type(torch.FloatTensor)
    response_time = 1.5
    safe_decel = -5

    for i, data in data_loader.items():
        r_l = []
        rmse_l = []
        n += 1
        x, for_sim_spacing, y = data
        print(i)
        for j in range(1, len(x[0])):
            current_frame = x[0][j]
            current_for_sim_spacing = for_sim_spacing[0][j]
            previous_frame = x[0][j - 1]
            previous_for_sim_spacing = for_sim_spacing[0][j - 1]
            if current_for_sim_spacing[-1] - previous_for_sim_spacing[-1] != 1:
                new_seg = True
                break
            # Sample number of context and target points
            # num_context = randint(*num_context_range)
            # num_extra_target = randint(*num_extra_target_range)
            # num_points = num_context + num_extra_target

            # Create context and target points and apply neural process
            num_context = len(x[0])
            num_extra_target = 1
            num_points = len(x[0])
            # x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_extra_target)
            # x: delta_V_n_t, S_n_t, V_n_t, pre_v, pre_tan_acc, pre_lat_acc
            # for sim spacing: S_n_t_y, ego_x, ego_y, next_pre_x, next_pre_y
            if j == 1 or new_seg:
                previous_delta_V_n_t = previous_frame[0]
                previous_S_n_t = previous_frame[1]
                previous_V_n_t = previous_frame[2]
                previous_pre_v = previous_frame[3]
                previous_pre_tan_acc = previous_frame[4]
                previous_pre_lat_acc = previous_frame[5]
            else:
                previous_delta_V_n_t = sim_previous_frame[0]
                previous_S_n_t = sim_previous_frame[1]
                previous_V_n_t = sim_previous_frame[2]
                previous_pre_v = sim_previous_frame[3]
                previous_pre_tan_acc = sim_previous_frame[4]
                previous_pre_lat_acc = sim_previous_frame[5]

            x_target = np.array([[previous_delta_V_n_t, previous_S_n_t, previous_V_n_t, previous_pre_v, previous_pre_tan_acc, previous_pre_lat_acc]])
            fix_x_target = np.array([previous_frame])
            x_context, y_context = np.array(x), np.array([y])
            x_context = torch.from_numpy(x_context).type(torch.FloatTensor)
            y_context = torch.from_numpy(y_context).type(torch.FloatTensor)
            x_target = torch.from_numpy(x_target).type(torch.FloatTensor).view(1, 1, x_dim)
            fix_x_target = torch.from_numpy(fix_x_target).type(torch.FloatTensor).view(1, 1, x_dim)
            # predict acceleration
            # p_y_pred, y_pred_mu, y_pred_sigma, r, mu_context, sigma_context = cf_np(x_context, y_context, x_target, None)
            p_y_pred, y_pred_mu, y_pred_sigma, r, mu_context, sigma_context = cf_np(x_context, y_context, x_target, None, new_agg_r)   # new ds cf model
            # a_n_t = y_pred_mu.detach().numpy().reshape(batch_size * num_points, 1)[0]
            a_n_t = y_pred_mu.detach().numpy().reshape(1,)[0]

            p_y_pred, y_pred_mu, y_pred_sigma, r, mu_context, sigma_context = cf_np(x_context, y_context, fix_x_target, None)
            fix_a_n_t = y_pred_mu.detach().numpy().reshape(1, )[0]
            a_err.append((fix_a_n_t - y[0][j - 1]) ** 2)
            # update velocity
            safe_distance = previous_V_n_t * response_time
            if previous_S_n_t < safe_distance:
                a_n_t = safe_decel
            print(a_n_t, fix_a_n_t, y[0][j - 1])
            V_n_t = previous_V_n_t + a_n_t * 0.04
            # calculate heading direction
            # previous_ego_x = previous_for_sim_spacing[1]
            # previous_ego_y = previous_for_sim_spacing[2]
            # ego_x = current_for_sim_spacing[1]
            # ego_y = current_for_sim_spacing[2]
            # direction = np.arctan((ego_y - previous_ego_y) / (previous_ego_x - ego_x))
            # update the ego vehicle's position
            if j == 1 or new_seg:
                previous_ego_x = previous_for_sim_spacing[1]
                previous_ego_y = previous_for_sim_spacing[2]
                ego_x = previous_ego_x - np.cos(direction) * V_n_t * 0.04
                ego_y = previous_ego_y + np.sin(direction) * V_n_t * 0.04
            else:
                ego_x = sim_previous_ego_x - np.cos(direction) * V_n_t * 0.04
                ego_y = sim_previous_ego_y + np.sin(direction) * V_n_t * 0.04
            pre_x = previous_for_sim_spacing[3]
            pre_y = previous_for_sim_spacing[4]
            print("leading v pos", pre_x, pre_y)
            print("sim ego pos", ego_x, ego_y)
            print("ego pos", current_for_sim_spacing[1], current_for_sim_spacing[2])
            # update the traffic condition
            previous_delta_V_n_t = V_n_t - current_frame[3]
            previous_S_n_t = np.sqrt((pre_y - ego_y) ** 2 + (pre_x - ego_x) ** 2)
            previous_V_n_t = V_n_t
            previous_pre_v = current_frame[3]
            previous_pre_tan_acc = current_frame[4]
            previous_pre_lat_acc = current_frame[5]
            sim_previous_frame = np.array([previous_delta_V_n_t, previous_S_n_t, previous_V_n_t, previous_pre_v, previous_pre_tan_acc, previous_pre_lat_acc])
            sim_previous_ego_x = ego_x
            sim_previous_ego_y = ego_y

            print("sim ttc", previous_S_n_t, previous_delta_V_n_t)
            print("gt ttc", current_frame[1], current_frame[0])
            sim_ttc.append(previous_S_n_t/previous_delta_V_n_t)
            sim_spacing.append(previous_S_n_t)
            sim_speed.append(previous_V_n_t)

            ttc.append(current_frame[1]/current_frame[0])
            spacing.append(current_frame[1])
            speed.append(current_frame[2])

            # print(mu_context, sigma_context, r)
            # r_l.append(r.detach().numpy())
            # print(n, i, mean_squared_error(y_pred_mu.detach().numpy().reshape(batch_size*num_points, 1),
            #                          y_target.detach().numpy().reshape(batch_size*num_points, 1)))
            # rmse_l.append(mean_squared_error(y_pred_mu.detach().numpy().reshape(batch_size*num_points, 1),
             #                         y_target.detach().numpy().reshape(batch_size*num_points, 1)))
        # r_l = np.array(r_l).reshape(20, 5)
        # print(r_l.shape)
        # print(np.mean(r_l, axis=0), np.std(r_l, axis=0))
        # all_r_c.append(np.mean(r_l, axis=0))
        # all_rmse.append(np.mean(rmse_l))
        break
    print(len(sim_ttc))
    print("ttc:", np.mean(sim_ttc), np.mean(ttc))
    print("spacing:", np.mean(sim_spacing), np.mean(spacing))
    print("speed:", np.mean(sim_speed), np.mean(speed))
    print(np.sqrt(np.mean(a_err)))
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/agg_index_5_sim_ttc.txt", sim_ttc)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/agg_index_5_sim_spacing.txt", sim_spacing)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/agg_index_5_sim_speed.txt", sim_speed)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/gt_ttc.txt", ttc)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/gt_spacing.txt", spacing)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/gt_speed.txt", speed)


def simulate_con():
    all_rmse = []
    all_r_c = []
    n = 1
    sim_ttc = []
    sim_spacing = []
    sim_speed = []
    ttc = []
    spacing = []
    speed = []
    a_err = []
    new_seg = False
    direction = 0.9712041389105396
    non_existed_r_c = np.loadtxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/new_ds.txt")
    new_con_r = torch.tensor(non_existed_r_c[0]).type(torch.FloatTensor)
    new_agg_r = torch.tensor(non_existed_r_c[1]).type(torch.FloatTensor)
    response_time = 2.0
    safe_decel = -5

    for i, data in data_loader.items():
        r_l = []
        rmse_l = []
        n += 1
        x, for_sim_spacing, y = data
        print(i)
        for j in range(1, len(x[0])):
            current_frame = x[0][j]
            current_for_sim_spacing = for_sim_spacing[0][j]
            previous_frame = x[0][j - 1]
            previous_for_sim_spacing = for_sim_spacing[0][j - 1]
            if current_for_sim_spacing[-1] - previous_for_sim_spacing[-1] != 1:
                new_seg = True
                break
            # Sample number of context and target points
            # num_context = randint(*num_context_range)
            # num_extra_target = randint(*num_extra_target_range)
            # num_points = num_context + num_extra_target

            # Create context and target points and apply neural process
            num_context = len(x[0])
            num_extra_target = 1
            num_points = len(x[0])
            # x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_extra_target)
            # x: delta_V_n_t, S_n_t, V_n_t, pre_v, pre_tan_acc, pre_lat_acc
            # for sim spacing: S_n_t_y, ego_x, ego_y, next_pre_x, next_pre_y
            if j == 1 or new_seg:
                previous_delta_V_n_t = previous_frame[0]
                previous_S_n_t = previous_frame[1]
                previous_V_n_t = previous_frame[2]
                previous_pre_v = previous_frame[3]
                previous_pre_tan_acc = previous_frame[4]
                previous_pre_lat_acc = previous_frame[5]
            else:
                previous_delta_V_n_t = sim_previous_frame[0]
                previous_S_n_t = sim_previous_frame[1]
                previous_V_n_t = sim_previous_frame[2]
                previous_pre_v = sim_previous_frame[3]
                previous_pre_tan_acc = sim_previous_frame[4]
                previous_pre_lat_acc = sim_previous_frame[5]

            x_target = np.array([[previous_delta_V_n_t, previous_S_n_t, previous_V_n_t, previous_pre_v,
                                  previous_pre_tan_acc, previous_pre_lat_acc]])
            fix_x_target = np.array([previous_frame])
            x_context, y_context = np.array(x), np.array([y])
            x_context = torch.from_numpy(x_context).type(torch.FloatTensor)
            y_context = torch.from_numpy(y_context).type(torch.FloatTensor)
            x_target = torch.from_numpy(x_target).type(torch.FloatTensor).view(1, 1, x_dim)
            fix_x_target = torch.from_numpy(fix_x_target).type(torch.FloatTensor).view(1, 1, x_dim)
            # predict acceleration
            # p_y_pred, y_pred_mu, y_pred_sigma, r, mu_context, sigma_context = cf_np(x_context, y_context, x_target, None)
            p_y_pred, y_pred_mu, y_pred_sigma, r, mu_context, sigma_context = cf_np(x_context, y_context, x_target,
                                                                                    None, new_con_r)  # new ds cf model
            # a_n_t = y_pred_mu.detach().numpy().reshape(batch_size * num_points, 1)[0]
            a_n_t = y_pred_mu.detach().numpy().reshape(1, )[0]

            p_y_pred, y_pred_mu, y_pred_sigma, r, mu_context, sigma_context = cf_np(x_context, y_context, fix_x_target,
                                                                                    None)
            fix_a_n_t = y_pred_mu.detach().numpy().reshape(1, )[0]
            a_err.append((fix_a_n_t - y[0][j - 1]) ** 2)
            # update velocity
            safe_distance = previous_V_n_t * response_time
            if previous_S_n_t < safe_distance:
                a_n_t = safe_decel
            print(a_n_t, fix_a_n_t, y[0][j - 1])
            V_n_t = previous_V_n_t + a_n_t * 0.04
            # calculate heading direction
            # previous_ego_x = previous_for_sim_spacing[1]
            # previous_ego_y = previous_for_sim_spacing[2]
            # ego_x = current_for_sim_spacing[1]
            # ego_y = current_for_sim_spacing[2]
            # direction = np.arctan((ego_y - previous_ego_y) / (previous_ego_x - ego_x))
            # update the ego vehicle's position
            if j == 1 or new_seg:
                previous_ego_x = previous_for_sim_spacing[1]
                previous_ego_y = previous_for_sim_spacing[2]
                ego_x = previous_ego_x - np.cos(direction) * V_n_t * 0.04
                ego_y = previous_ego_y + np.sin(direction) * V_n_t * 0.04
            else:
                ego_x = sim_previous_ego_x - np.cos(direction) * V_n_t * 0.04
                ego_y = sim_previous_ego_y + np.sin(direction) * V_n_t * 0.04
            pre_x = previous_for_sim_spacing[3]
            pre_y = previous_for_sim_spacing[4]
            print("leading v pos", pre_x, pre_y)
            print("sim ego pos", ego_x, ego_y)
            print("ego pos", current_for_sim_spacing[1], current_for_sim_spacing[2])
            # update the traffic condition
            previous_delta_V_n_t = V_n_t - current_frame[3]
            previous_S_n_t = np.sqrt((pre_y - ego_y) ** 2 + (pre_x - ego_x) ** 2)
            previous_V_n_t = V_n_t
            previous_pre_v = current_frame[3]
            previous_pre_tan_acc = current_frame[4]
            previous_pre_lat_acc = current_frame[5]
            sim_previous_frame = np.array(
                [previous_delta_V_n_t, previous_S_n_t, previous_V_n_t, previous_pre_v, previous_pre_tan_acc,
                 previous_pre_lat_acc])
            sim_previous_ego_x = ego_x
            sim_previous_ego_y = ego_y

            print("sim ttc", previous_S_n_t, previous_delta_V_n_t)
            print("gt ttc", current_frame[1], current_frame[0])
            sim_ttc.append(previous_S_n_t / previous_delta_V_n_t)
            sim_spacing.append(previous_S_n_t)
            sim_speed.append(previous_V_n_t)

            ttc.append(current_frame[1] / current_frame[0])
            spacing.append(current_frame[1])
            speed.append(current_frame[2])

            # print(mu_context, sigma_context, r)
            # r_l.append(r.detach().numpy())
            # print(n, i, mean_squared_error(y_pred_mu.detach().numpy().reshape(batch_size*num_points, 1),
            #                          y_target.detach().numpy().reshape(batch_size*num_points, 1)))
            # rmse_l.append(mean_squared_error(y_pred_mu.detach().numpy().reshape(batch_size*num_points, 1),
            #                         y_target.detach().numpy().reshape(batch_size*num_points, 1)))
        # r_l = np.array(r_l).reshape(20, 5)
        # print(r_l.shape)
        # print(np.mean(r_l, axis=0), np.std(r_l, axis=0))
        # all_r_c.append(np.mean(r_l, axis=0))
        # all_rmse.append(np.mean(rmse_l))
        break
    print(len(sim_ttc))
    print("ttc:", np.mean(sim_ttc), np.mean(ttc))
    print("spacing:", np.mean(sim_spacing), np.mean(spacing))
    print("speed:", np.mean(sim_speed), np.mean(speed))
    print(np.sqrt(np.mean(a_err)))
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/con_index_5_sim_ttc.txt", sim_ttc)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/con_index_5_sim_spacing.txt", sim_spacing)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/con_index_5_sim_speed.txt", sim_speed)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/gt_ttc.txt", ttc)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/gt_spacing.txt", spacing)
    np.savetxt("/root/AAAI2022_neuma/0714_dist_param/new_ds_cf_model/gt_speed.txt", speed)

simulate_agg()
simulate_con()
exit()

plt.figure()
plt.hist(spacing, color="lightcoral")
plt.hist(sim_spacing, alpha=0.6, color="skyblue")
plt.savefig("/root/AAAI2022_neuma/0803_mop_space_dist_param/"+str(int(i))+"/comp_spacing.png")
plt.figure()
plt.hist(ttc, color="lightcoral", range=(-50, 50))
plt.hist(sim_ttc, alpha=0.6, color="skyblue", range=(-50, 50))
plt.savefig("/root/AAAI2022_neuma/0803_mop_space_dist_param/"+str(int(i))+"/comp_ttc.png")
plt.figure()
plt.hist(speed, color="lightcoral")
plt.hist(sim_speed, alpha=0.6, color="skyblue")
plt.savefig("/root/AAAI2022_neuma/0803_mop_space_dist_param/"+str(int(i))+"/comp_speed.png")
exit()
print(np.mean(all_rmse), np.std(all_rmse))
all_r_c = np.array(all_r_c)
np.savetxt("0714_dist_param/100_r_c.txt", all_r_c)
print(all_r_c.shape)
agg_indexes = np.loadtxt("0714_dist_param/100_agg_indexes.txt")

plt.figure()
plt.scatter(all_r_c[:, 0], agg_indexes[:, 1])
plt.savefig("0714_dist_param/100_agg_indexes_r_c1_reg.png")
plt.figure()
plt.scatter(all_r_c[:, 1], agg_indexes[:, 1])
plt.savefig("0714_dist_param/100_agg_indexes_r_c2_reg.png")
plt.figure()
plt.scatter(all_r_c[:, 2], agg_indexes[:, 1])
plt.savefig("0714_dist_param/100_agg_indexes_r_c3_reg.png")
plt.figure()
plt.scatter(all_r_c[:, 3], agg_indexes[:, 1])
plt.savefig("0714_dist_param/100_agg_indexes_r_c4_reg.png")
plt.figure()
plt.scatter(all_r_c[:, 4], agg_indexes[:, 1])
plt.savefig("0714_dist_param/100_agg_indexes_r_c5_reg.png")


err_fixed = []

next_v = 1
for v_id, all_cf_data in all_data_for_cf_model.items():
    print("------------------------------------------------------------------------------------------------------")
    print(str(next_v) + 'th vehicle with id ' + str(v_id))
    if next_v > 100:
        break
    next_v += 1

    param_using_all_data = np.loadtxt('0704_dist_param/fixed_params/' + str(int(v_id)) + '_using_all_data.txt')

    # [delta_v_l, space_hw_l, ego_v_l, a_l] data
    delta_V_n_t = np.array(all_cf_data[0])
    S_n_t = np.array(all_cf_data[1])
    V_n_t = np.array(all_cf_data[2])
    a = np.array(all_cf_data[3])
    t = np.array(all_cf_data[4])
    pre_v = np.array(all_cf_data[5])
    pre_tan_acc = np.array(all_cf_data[6])
    pre_lat_acc = np.array(all_cf_data[7])

    data_array = np.array([delta_V_n_t, S_n_t, V_n_t, a, t]).T
    data_array = data_array[data_array[:, -1].argsort()]

    # parameters fitted using all data
    a_max_n = param_using_all_data[0]
    desired_V_n = param_using_all_data[1]
    a_comf_n = param_using_all_data[2]
    S_jam_n = param_using_all_data[3]
    desired_T_n = param_using_all_data[4]
    beta = 4

    # check RMSE
    sum_err_using_fix_param = 0

    for i in range(len(data_array)):
        frame = data_array[i]
        delta_V_n_t = frame[0]
        S_n_t = frame[1]
        V_n_t = frame[2]
        a_n_t = frame[3]

        a_hat = IDM_cf_model_for_p(frame[0], frame[1], frame[2], a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta)

        sum_err_using_fix_param += (a_hat - a_n_t) ** 2

    err_fixed.append(np.sqrt(sum_err_using_fix_param / len(data_array)))

print(np.mean(err_fixed), np.std(err_fixed))

