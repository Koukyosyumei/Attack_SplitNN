import torch
from sklearn.metrics import roc_auc_score


def label_leak_auc(splitnn, dataloader):
    epoch_labels = []
    epoch_g_norm = []
    for i, data in enumerate(dataloader, 0):

        inputs, labels = data
        inputs = inputs.to(splitnn.device)
        labels = labels.to(splitnn.device)

        intermidiate_to_server = splitnn.client._fit_client_forward(
            inputs, labels)
        _, _, grad_to_client = splitnn.server._fit_server(
            intermidiate_to_server, labels)

        g_norm = grad_to_client.pow(2).sum(dim=1).sqrt()
        epoch_labels.append(labels)
        epoch_g_norm.append(g_norm)

    epoch_labels = torch.cat(epoch_labels)
    epoch_g_norm = torch.cat(epoch_g_norm)
    score = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
    return score
