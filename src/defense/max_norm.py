import torch

from ..splitnn import Server


class Server_with_max_norm(Server):
    def __init__(self, server_model,
                 server_optimizer,
                 criterion):
        super().__init__(server_model,
                         server_optimizer,
                         criterion)

    def _fit_server(self, intermidiate_to_server, labels):
        outputs = self.server_model(intermidiate_to_server)
        loss = self.criterion(outputs, labels)
        loss.backward()

        grad_to_client = intermidiate_to_server.grad.clone()
        grad_to_client = self._max_norm(grad_to_client)
        return outputs, loss, grad_to_client

    def _max_norm(self, grad):
        """server-side heuristic approach to prevent label leakage attacks
        https://arxiv.org/abs/2102.08504

        Args:
            grad (torch.Tensor): the gradient of L with respect to the
                                input of the function h

                                ---
                                L : the loss function
                                f : the client side model
                                h : the server side model
                                the whole model can be expressed as h ◦ f


        Returns
            pertubated_gard (torch.Tensor): noised gradient which is
                                            supposed to be sent to the client
        """

        g_norm = grad.pow(2).sum(dim=list(range(1, len(grad.shape)))).sqrt()
        # maximum gradient norm among the mini-batch
        g_max = g_norm[torch.argmax(g_norm)]
        # the standard deviation to be determined
        sigma = torch.sqrt(g_max / g_norm - 1)
        # gausiaan noise
        perturbation = torch.normal(torch.zeros_like(sigma), sigma)
        # expand dimension
        perturbation = perturbation.expand(list(grad.shape)[::-1]).T
        # perturbed gradient
        pertubated_gard = grad + perturbation

        return pertubated_gard
