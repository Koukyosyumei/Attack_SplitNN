import random

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn


class ShredderNoisyActivation(nn.Module):
    def __init__(self, activation_size, dist_of_noise="norm",
                 loc=0.0, scale=20.0):
        super(ShredderNoisyActivation, self).__init__()

        self.activation_size = activation_size
        self.loc = loc
        self.scale = scale
        if dist_of_noise == "norm":
            self.dist_of_noise = st.norm
        elif dist_of_noise == "laplace":
            self.dist_of_noise = st.laplace

        # initialize the noise tensor
        self.train_noise = None
        self._initialize_noise_tensor()

        self.dist_params_of_noise = None
        self.sorted_noise_index = None
        self.eval_noise = []

    def forward(self, input):
        if self.training:
            return self._forward_train(input)
        else:
            return self._forward_eval(input)

    def _forward_train(self, input):
        return input + self.train_noise

    def _forward_eval(self, input):
        if self.dist_params_of_noise is None:
            self._set_params_for_inference()

        return input + random.choice(self.eval_noise)

    def _initialize_noise_tensor(self):
        m = torch.distributions.laplace.Laplace(
            loc=self.loc, scale=self.scale, validate_args=None)
        self.train_noise = nn.Parameter(m.rsample(self.activation_size))

    def _set_params_for_inference(self):

        # flatten the noise optimized during training
        np_train_noise = self.train_noise.clone().detach().numpy()
        noise_flatten = np_train_noise.flatten()
        # get the order of noise elements
        self.sorted_noise_index = np.argsort(noise_flatten)

        # fit the noise to distriution
        self.dist_params_of_noise = self.dist_of_noise.fit(np_train_noise)

        # sample new noise from fitted distriution
        noise_sampled = self.dist_of_noise.rvs(
            loc=self.dist_params_of_noise[-2],
            scale=self.dist_params_of_noise[-1],
            size=np.prod(self.activation_size))

        # reorder and reshape the new noise
        sorted_sampled_noise_index = np.argsort(noise_sampled)
        noise_flatten[self.sorted_noise_index] =\
            noise_sampled[sorted_sampled_noise_index]
        updated_noise = noise_flatten.reshape(self.activation_size)

        self.eval_noise.append(torch.Tensor(updated_noise))


class Shredder:
    def __init__(self, splitnn, intermidiate_shape,
                 epoch, base_criterion,
                 optimizer,
                 lr=1e2,
                 num_of_distribution=1,
                 alpha=0.01,
                 dist_of_noise="laplace"):

        self.splitnn = splitnn
        self.epoch = epoch
        self.base_criterion = base_criterion
        self.optimizer = optimizer
        self.lr = lr
        self.alpha = alpha
        self.intermidiate_shape = intermidiate_shape
        self.dist_of_noise = dist_of_noise

        self.client = self.splitnn.client
        self.model_noise = ShredderNoisyActivation(intermidiate_shape,
                                                   dist_of_noise=dist_of_noise)

        self.sampled_noises = []

    def fit(self, train_dataloader):

        self.splitnn.eval()
        for _ in range(self.num_of_distribution):

            temp_model_noise = ShredderNoisyActivation(
                self.intermidiate_shape,
                dist_of_noise=self.dist_of_noise
            )
            temp_optimizer = self.optimizer(
                temp_model_noise.parameters(), lr=self.lr)
            temp_model_noise.train()

            for nitr in range(self.epoch):
                for i, data in enumerate(train_dataloader):
                    temp_optimizer.zero_grad()
                    x, y = data[0], data[1]

                    with torch.no_grad():
                        intermidiate = self.client.client_model(x)
                        outputs = self.splitnn(x)

                    noised_intermidiate = temp_model_noise(intermidiate)
                    norm_noised_intermidiate = noised_intermidiate.norm()

                    # basic shredder
                    base_loss = self.base_criterion(outputs, y)
                    loss = base_loss + self.alpha * norm_noised_intermidiate

                    # update the model_noise
                    loss.backward()
                    temp_optimizer.step()

            self.sampled_noises.append(temp_model_noise.eval_noise[0])
            del temp_model_noise
            del temp_optimizer

        self.model_noise.eval_noise = self.sampled_noises

    def update_client(self):
        self.client.client_model = nn.Seuquential(
            self.client.client_model, self.model_noise)
        return self.client
