import torch


class Client(torch.nn.Module):
    def __init__(self, client_model):
        super().__init__()
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
        self.client_side_intermidiate = None

    def forward(self, inputs):
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

    def backward(self, grad_to_client):
        """client-side back propagation

        Args:
            grad_to_client
        """
        self.client_side_intermidiate.backward(grad_to_client)

    def train(self):
        self.client_model.train()

    def eval(self):
        self.client_model.eval()


class Server(torch.nn.Module):
    def __init__(self, server_model):
        super().__init__()
        """class that expresses the Server on SplitNN

        Args:
            server_model (torch model): server-side model
            server_optimizer (torch optimizer): optimizer for server-side model
            criterion (function): loss function for training

        Attributes:
            server_model (torch model): server-side model
            server_optimizer (torch optimizer): optimizer for server-side model
        """
        self.server_model = server_model

        self.intermidiate_to_server = None

    def forward(self, intermidiate_to_server):
        """server-side training

        Args:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model

        Returns:
            outputs (torch.Tensor): outputs of server-side model
        """
        self.intermidiate_to_server = intermidiate_to_server
        outputs = self.server_model(intermidiate_to_server)

        return outputs

    def backward(self):
        grad_to_client = self.intermidiate_to_server.grad.clone()
        return grad_to_client

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()


class SplitNN(torch.nn.Module):
    def __init__(self, client, server,
                 client_optimizer, server_optimizer,
                 ):
        super().__init__()
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
        self.client_optimizer = client_optimizer
        self.server_optimizer = server_optimizer

    def forward(self, inputs):
        # execute client - feed forward network
        intermidiate_to_server = self.client.forward(inputs)
        # execute server - feed forward netwoek
        outputs = self.server.forward(intermidiate_to_server)

        return outputs

    def backward(self):
        # execute server - back propagation
        grad_to_client = self.server.backward()
        # execute client - back propagation
        self.client.backward(grad_to_client)

    def zero_grads(self):
        self.client_optimizer.zero_grad()
        self.server_optimizer.zero_grad()

    def step(self):
        self.client_optimizer.step()
        self.server_optimizer.step()

    def train(self):
        self.client.train()
        self.server.train()

    def eval(self):
        self.client.eval()
        self.server.eval()
