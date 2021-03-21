import torch
from sklearn.metrics import roc_auc_score

from ..splitnn import SplitNN


class SplitNN_with_leak_auc(SplitNN):
    def __init__(self, client, server, device="cpu"):
        super().__init__(client, server, device="cpu")

    def fit(self, dataloader, epochs, metric=None):
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_labels = []
            epoch_outputs = []
            epoch_g_norm = []
            for i, data in enumerate(dataloader, 0):

                self.client.client_optimizer.zero_grad()
                self.server.server_optimizer.zero_grad()

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                intermidiate_to_server = self.client._fit_client_forward(
                    inputs, labels)
                outputs, loss, grad_to_client = self.server._fit_server(
                    intermidiate_to_server, labels)
                self.client._fit_client_backpropagation(grad_to_client)

                self.client.client_optimizer.step()
                self.server.server_optimizer.step()

                epoch_loss += loss / len(dataloader.dataset)
                epoch_outputs.append(outputs)
                epoch_labels.append(labels)

                g_norm = grad_to_client.pow(2).sum(dim=1).sqrt()
                epoch_g_norm.append(g_norm)

            epoch_outputs = torch.cat(epoch_outputs)
            epoch_labels = torch.cat(epoch_labels)
            epoch_g_norm = torch.cat(epoch_g_norm)

            if metric is not None:
                m = metric(epoch_labels, epoch_outputs)
            leak_auc = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
            print(
                f"epoch {epoch+1}, loss {epoch_loss: .5}, metric {m} " +
                f"leak AUC {leak_auc: .5}")
