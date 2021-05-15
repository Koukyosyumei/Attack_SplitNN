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
