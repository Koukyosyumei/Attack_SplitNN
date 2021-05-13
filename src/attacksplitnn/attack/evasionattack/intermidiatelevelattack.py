import torch

from ..attacker import AbstractAttacker

# this code is mainly based on the greates repository
# https://github.com/CUAI/Intermediate-Level-Attack/blob/master/attacks.py


class IntermidiateLevelAttack(AbstractAttacker):
    def __init__(self, splitnn, epochs,
                 lr, alpha, epsilon,
                 with_projection=False):
        super().__init__(splitnn)
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.with_projection = with_projection

        self.client_model = splitnn.client.client_model

    def attack(self, original_data, adversarial_example):
        original_data = original_data.detach()
        new_adversarial_example = adversarial_example.clone()
        mid_original_data = self.splitnn.client(original_data).detach()
        delta_y_prime = self.client_model(adversarial_example).detach() - \
            mid_original_data

        for _ in range(self.epochs):
            new_adversarial_example = new_adversarial_example.detach()
            new_adversarial_example.requires_grad = True
            delta_y_doubleprime = self.client_model(
                new_adversarial_example) - mid_original_data

            if self.with_projection:
                pass
            else:
                loss = self.ila_flexible_loss(
                    delta_y_prime, delta_y_doubleprime)
            loss.backward()

            grad_new_adversarial_example_L = new_adversarial_example.grad.\
                detach()
            new_adversarial_example = new_adversarial_example.detach()
            new_adversarial_example += self.lr * grad_new_adversarial_example_L
            new_adversarial_example = self._clip_epsilon(
                new_adversarial_example, original_data, self.epsilon) +\
                original_data
            new_adversarial_example = self._clip_image_range(
                new_adversarial_example)

        return new_adversarial_example

    def ila_flexible_loss(self, delta_y_prime, delta_y_doubleprime):

        delta_y_prime_norm = delta_y_prime.pow(2).sqrt().sum()
        delta_y_doubleprime_norm = delta_y_doubleprime.pow(
            2).sqrt().sum()

        loss = torch.mm((delta_y_doubleprime / delta_y_doubleprime_norm),
                        (delta_y_prime / delta_y_prime_norm)
                        .transpose(0, 1)) +\
            self.alpha * (delta_y_doubleprime_norm / delta_y_prime_norm)

        return loss

    def proj_loss(self):
        pass

    def _clip_epsilon(self, x, y, epsilon):
        return (x.clone().detach() - y.clone().detach()).\
            clamp(-epsilon, epsilon)

    def _clip_image_range(self, x):
        return x
