import torch
from sklearn.metrics import roc_auc_score


class NormAttack:
    def __init__(self, splitnn):
        self.splitnn = splitnn

    def attack(self, dataloader):
        """culculate leak_auc on the given SplitNN model
        reference: https://arxiv.org/abs/2102.08504

        Args:
            splitnn (attack_splitnn.splitnn.SplitNN): target system
            dataloader (torch dataloader):

        Returns:
            score: culculated leak auc
        """
        epoch_labels = []
        epoch_g_norm = []
        for i, data in enumerate(dataloader, 0):

            inputs, labels = data
            inputs = inputs.to(self.splitnn.device)
            labels = labels.to(self.splitnn.device)

            intermidiate_to_server = self.splitnn.client._fit_client_forward(
                inputs)
            _, _, grad_to_client = self.splitnn.server._fit_server(
                intermidiate_to_server, labels)

            g_norm = grad_to_client.pow(2).sum(dim=1).sqrt()
            epoch_labels.append(labels)
            epoch_g_norm.append(g_norm)

        epoch_labels = torch.cat(epoch_labels)
        epoch_g_norm = torch.cat(epoch_g_norm)
        score = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
        return score
