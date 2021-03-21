import torch


class Client:
    def __init__(self, client_model,
                 client_optimizer):
        self.client_model = client_model
        self.client_optimizer = client_optimizer
        self.client_side_intermidiate = None

    def _fit_client_forward(self, inputs, label):
        self.client_side_intermidiate = self.client_model(inputs)
        # send intermidiate tensor to the server
        intermidiate_to_server = self.client_side_intermidiate.detach()\
            .requires_grad_()

        return intermidiate_to_server

    def _fit_client_backpropagation(self, grad_to_client):
        self.client_side_intermidiate.backward(grad_to_client)


class Server:
    def __init__(self, server_model,
                 server_optimizer,
                 criterion):
        self.server_model = server_model
        self.server_optimizer = server_optimizer
        self.criterion = criterion

    def _fit_server(self, intermidiate_to_server, labels):
        outputs = self.server_model(intermidiate_to_server)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grad_to_client = intermidiate_to_server.grad.clone()
        return outputs, loss, grad_to_client


class SplitNN:
    def __init__(self, client, server, device="cpu"):

        self.client = client
        self.server = server
        self.device = device

    def fit(self, dataloader, epochs, metric=None):
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_labels = []
            epoch_outputs = []
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
