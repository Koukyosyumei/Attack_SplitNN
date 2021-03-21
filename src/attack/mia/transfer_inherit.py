import torch


class mia_transfer_inherit:
    def __init__(self, shadow_client, server, device="cpu"):

        self.shadow_client = shadow_client
        self.server = server
        self.device = device

    def fit(self, shadowloader, epochs, metric=None):
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_labels = []
            epoch_outputs = []
            for i, data in enumerate(shadowloader, 0):

                self.shadow_client.client_optimizer.zero_grad()

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # execute client - feed forward network
                intermidiate_to_server =\
                    self.shadow_client._fit_client_forward(inputs, labels)
                # execute server side actions
                outputs, loss, grad_to_client = self.server._fit_server(
                    intermidiate_to_server, labels)
                # execute client - back propagation
                self.shadow_client._fit_client_backpropagation(grad_to_client)

                self.shadow_client.client_optimizer.step()

                epoch_loss += loss / len(shadowloader.dataset)
                epoch_outputs.append(outputs)
                epoch_labels.append(labels)

            epoch_outputs = torch.cat(epoch_outputs)
            epoch_labels = torch.cat(epoch_labels)

            self._print_metric(epoch, epoch_loss,
                               epoch_outputs, epoch_labels,
                               metric=metric)

    def predict(self):
        pass

    def _print_metric(self, epoch,
                      epoch_loss,
                      epoch_outputs, epoch_labels,
                      metric=None):
        if metric is not None:
            m = metric(epoch_labels, epoch_outputs)
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}, metric {m}")

        else:
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}")
