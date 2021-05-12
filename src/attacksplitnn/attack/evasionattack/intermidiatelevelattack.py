import torch

from ..attacker import AbstractAttacker

# this code is mainly based on the greates repository
# https://github.com/CUAI/Intermediate-Level-Attack/blob/master/attacks.py


class Proj_Loss(torch.nn.Module):
    def __init__(self):
        super(Proj_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss


class Mid_layer_target_Loss(torch.nn.Module):
    def __init__(self):
        super(Mid_layer_target_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)

        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * coeff


class IntermidiateLevelAttack(AbstractAttacker):
    def __init__(self, splitnn, epochs,
                 lr, alpha, epsilon,
                 with_projection=True):
        super().__init__(splitnn)
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.with_projection = with_projection

        self.client = self.splitnn.client.client_model

    def attack(self, X, X_attack):

        X = X.detach()
        X_pert = torch.zeros(X.size())
        X_pert.copy_(X).detach()
        X_pert.requires_grad = True

        mid_original = self.client(X)
        mid_attack_original = self.client(X_attack)

        for _ in range(self.epochs):
            mid_output = self.client(X_pert)
            # generate adversarial example by max middle layer pertubation
            # in the direction of increasing loss
            if self.with_projection:
                loss = Proj_Loss()(
                    mid_attack_original.detach(),
                    mid_output, mid_original.detach(),
                    self.alpha
                )
            else:
                loss = Mid_layer_target_Loss()(
                    mid_attack_original.detach(),
                    mid_output, mid_original.detach(),
                    self.alpha
                )

            loss.backward()
            pert = self.lr * X_pert.grad.detach().sign()

            # minimize loss
            X_pert = X_pert.detach() + pert
            X_pert.requires_grad = True

            # make sure we don't modify the original image beyond epsilon
            X_pert = self._clip_epsilon(X, X_pert, self.epsilon)
            X_pert.requires_grad = True

        return X_pert

    def ila_flexible_loss(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)

        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * coeff

    def proj_loss(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss

    def _clip_epsilon(self, x, y, epsilon):
        return (x.clone().detach() - y.clone().detach()).\
            clamp(-epsilon, epsilon)

    def _clip_image_range(self, x):
        return x
