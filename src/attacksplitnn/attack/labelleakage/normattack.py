import torch
from sklearn.metrics import roc_auc_score

from ..attacker import AbstractAttacker


class NormAttack(AbstractAttacker):
    def __init__(self, splitnn):
        """Class that implement normattack

        Args:
            splitnn (attack_splitnn.splitnn.SplitNN): target splotnn model
        """
        super().__init__(splitnn)
        self.splitnn = splitnn

    def attack(self, dataloader, criterion, device):
        """Culculate leak_auc on the given SplitNN model
           reference: https://arxiv.org/abs/2102.08504

        Args:
            dataloader (torch dataloader): dataloader for evaluation
            criterion: loss function for training
            device: cpu or GPU

        Returns:
            score: culculated leak auc
        """
        epoch_labels = []
        epoch_g_norm = []
        for i, data in enumerate(dataloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = self.splitnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.splitnn.backward()

            grad_from_server = self.splitnn.client.grad_from_server
            g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
            epoch_labels.append(labels)
            epoch_g_norm.append(g_norm)

        epoch_labels = torch.cat(epoch_labels)
        epoch_g_norm = torch.cat(epoch_g_norm)
        score = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
        return score
