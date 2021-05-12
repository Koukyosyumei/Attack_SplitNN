import torch


class Client(torch.nn.Module):
    def __init__(self, client_model):
        super().__init__()
        """class that expresses the Client on SplitNN

        Args:
            client_model (torch model): client-side model

        Attributes:
            client_model (torch model): cliet-side model
            client_side_intermidiate (torch.Tensor): output of
                                                     client-side model
            grad_from_server
        """

        self.client_model = client_model
        self.client_side_intermidiate = None
        self.grad_from_server = None

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

    def client_backward(self, grad_from_server):
        """client-side back propagation

        Args:
            grad_from_server: gradient which the server send to the client
        """
        self.grad_from_server = grad_from_server
        self.client_side_intermidiate.backward(grad_from_server)

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

        Attributes:
            server_model (torch model): server-side model
            intermidiate_to_server:
            grad_to_client
        """
        self.server_model = server_model

        self.intermidiate_to_server = None
        self.grad_to_client = None

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

    def server_backward(self):
        self.grad_to_client = self.intermidiate_to_server.grad.clone()
        return self.grad_to_client

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
            clietn_optimizer
            server_optimizer

        Attributes:
            client (attack_splitnn.splitnn.Client):
            server (attack_splitnn.splitnn.Server):
            clietn_optimizer
            server_optimizer
        """
        self.client = client
        self.server = server
        self.client_optimizer = client_optimizer
        self.server_optimizer = server_optimizer

        self.intermidiate_to_server = None

    def forward(self, inputs):
        # execute client - feed forward network
        self.intermidiate_to_server = self.client(inputs)
        # execute server - feed forward netwoek
        outputs = self.server(self.intermidiate_to_server)

        return outputs

    def backward(self):
        # execute server - back propagation
        grad_to_client = self.server.server_backward()
        # execute client - back propagation
        self.client.client_backward(grad_to_client)

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
