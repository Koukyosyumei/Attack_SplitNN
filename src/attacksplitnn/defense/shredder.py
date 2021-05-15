import copy

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn


class ShredderNoisyActivation(nn.Module):
    def __init__(self, activation_size, dist_of_noise="norm"):
        super(ShredderNoisyActivation, self).__init__()

        self.activation_size = activation_size
        m = torch.distributions.laplace.Laplace(
            loc=0.0, scale=20.0, validate_args=None)
        self.train_noise = nn.Parameter(m.rsample(activation_size))
        self.train_weight = nn.Parameter(
            torch.Tensor(torch.zeros(activation_size)))
        nn.init.normal_(self.train_weight)

        if dist_of_noise == "norm":
            self.dist_of_noise = st.norm
        elif dist_of_noise == "laplace":
            self.dist_of_noise = st.laplace

        self.dist_params_of_noise = None
        self.params_for_inference = None
        self.eval_weight = None

    def forward(self, input):
        if self.training:
            return self._forward_train(input)
        else:
            return self._forward_eval(input)

    def _forward_train(self, input):
        return input*self.train_weight + self.train_noise

    def _forward_eval(self, input):
        if self.params_for_inference is None:
            self._set_params_for_inference()

        return input*self.eval_weight + self.train_noise

    def _set_params_for_inference(self):

        np_train_weight = self.train_weight.clone().detach().numpy()
        weight_flatten = np_train_weight.flatten()
        sorted_weight_index = np.argsort(weight_flatten)

        self.dist_params_of_noise = self.dist_of_noise.fit(np_train_weight)
        weight_sampled = self.dist_of_noise.rvs(
            loc=self.dist_params_of_noise[-2],
            scale=self.dist_params_of_noise[-1],
            size=np.prod(self.activation_size))
        sorted_sampled_weight_index = np.argsort(weight_sampled)

        weight_flatten[sorted_weight_index] =\
            weight_sampled[sorted_sampled_weight_index]
        updated_weight = weight_flatten.reshape(self.activation_size)
        self.eval_weight = torch.Tensor(updated_weight)


class Shredder:
    def __init__(self, splitnn, intermidiate_shape,
                 epoch, base_criterion,
                 lam=1e3, c=1e3, dist_of_noise="norm"):
        self.splitnn = splitnn
        self.epoch = epoch
        self.base_criterion = base_criterion
        self.lam = lam
        self.c = c

        self.client = self.splitnn.client
        self.model_noise = ShredderNoisyActivation(intermidiate_shape,
                                                   dist_of_noise=dist_of_noise)

    def fit(self, train_dataloader):
        random_dataloader = copy.deepcopy(train_dataloader)

        self.splitnn.eval()
        self.model_noise.train()
        for nitr in range(epoch):
            for i, (data, data_rand) in enumerate(zip(train_dataloader,
                                                      random_dataloader)):
                x, y = data[0], data[1]
                x_rand, y_rand = data_rand[0], data_rand[1]

                with torch.no_grad():
                    intermidiate = self.client.client_model(x)
                    intermidiate_rand = self.client.client_model(x_rand)
                    outputs = self.splitnn(x)

                intermidiate = self.model_noise(intermidiate)
                intermidiate_rand = self.model_noise(intermidiate_rand)

                distance = (intermidiate_rand -
                            intermidiate).square().sqrt().sum()

                positive = (y == y_rand)
                negative = (y != y_rand)

                positive_distance = sum(torch.mul(positive, distance))
                negative_distance = sum(torch.mul(negative, distance))

                loss = self.base_criterion(outputs, y) +\
                    self.lam*(positive_distance +
                              (self.c - negative_distance))
                loss.backward()

                # TODO step, optimizer, etc...

    def predict(self):
        pass
