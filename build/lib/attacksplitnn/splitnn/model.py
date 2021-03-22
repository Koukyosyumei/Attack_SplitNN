import torch


class Client:
    def __init__(self, client_model,
                 client_optimizer):
        """class that expresses the Client on SplitNN

        Args:
            client_model (torch model): client-side model
            client_optimizer (torch optimizer): optimizer for client-side model

        Attributes:
            client_model (torch model): cliet-side model
            client_optimizer (torch optimizer): optimizer for client-side model
            client_side_intermidiate (torch.Tensor): output of
                                                     client-side model
        """

        self.client_model = client_model
        self.client_optimizer = client_optimizer
        self.client_side_intermidiate = None

    def _fit_client_forward(self, inputs):
        """client-side feed forward network

        Args:
            inputs (torch.Tensor): the input data

        Returns:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model which the client sent
                                                   to the server
        """

        self.client_side_intermidiate = self.client_model(inputs)
        # send intermidiate tensor to the server
        intermidiate_to_server = self.client_side_intermidiate.detach()\
            .requires_grad_()

        return intermidiate_to_server

    def _fit_client_backpropagation(self, grad_to_client):
        """client-side back propagation

        Args:
            grad_to_client
        """
        self.client_side_intermidiate.backward(grad_to_client)


class Server:
    def __init__(self, server_model,
                 server_optimizer,
                 criterion):
        """class that expresses the Server on SplitNN

        Args:
            server_model (torch model): server-side model
            server_optimizer (torch optimizer): optimizer for server-side model
            criterion (function): loss function for training

        Attributes:
            server_model (torch model): server-side model
            server_optimizer (torch optimizer): optimizer for server-side model
            criterion (function): loss function for training
        """
        self.server_model = server_model
        self.server_optimizer = server_optimizer
        self.criterion = criterion

    def _fit_server(self, intermidiate_to_server, labels):
        """server-side training

        Args:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model
            labels (torch.Tensor): ground-truth label

        Returns:
            outputs (torch.Tensor): outputs of server-side model
            loss:
            grad_to_client (torch.Tensor): the gradient of loss with respect to
                                           the input of the server-side model
                                        (corresponds to intermidiate_to_server)
        """
        outputs = self.server_model(intermidiate_to_server)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grad_to_client = intermidiate_to_server.grad.clone()
        return outputs, loss, grad_to_client


class SplitNN:
    def __init__(self, client, server, device="cpu"):
        """class that expresses the whole architecture of SplitNN

        Args:
            client (attack_splitnn.splitnn.Client):
            server (attack_splitnn.splitnn.Server):
            device (str): device type (default 'cpu')

        Attributes:
            client (attack_splitnn.splitnn.Client):
            server (attack_splitnn.splitnn.Server):
            device (str): device type (default 'cpu')

        Examples:
            model_1 = FirstNet()
            model_1 = model_1.to(device)

            model_2 = SecondNet()
            model_2 = model_2.to(device)

            opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)
            opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)

            criterion = nn.BCELoss()

            client = Client(model_1, opt_1)
            server = Server(model_2, opt_2, criterion)

            sn = SplitNN(client, server, device=device)
            sn.fit(train_loader, 3, metric=torch_auc)
        """
        self.client = client
        self.server = server
        self.device = device

    def fit(self, dataloader, epochs, metric=None):
        """train SplitNN

        Args:
            dataloader (torch Dataloaser)
            epochs (int):
            metric (function):
        """
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

                # execute client - feed forward network
                intermidiate_to_server = self.client._fit_client_forward(
                    inputs)
                # execute server side actions
                outputs, loss, grad_to_client = self.server._fit_server(
                    intermidiate_to_server, labels)
                # execute client - back propagation
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
        """culculate the given metric and print the log of each epoch

        Args:
            epoch (int):
            epoch_loss (float):
            epoch_outpus (torch.Tensor)
            epoch_labels (torch.Tensor)
            metric (function)
        """
        if metric is not None:
            m = metric(epoch_labels, epoch_outputs)
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}, metric {m}")

        else:
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}")
