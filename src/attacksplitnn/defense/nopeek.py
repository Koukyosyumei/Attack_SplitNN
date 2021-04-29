import torch


class NoPeekLoss(torch.nn.modules.loss._Loss):
    def __init__(self,
                 alpha: float = 0.1,
                 base_loss: torch.nn.Module = torch.nn.CrossEntropyLoss())\
            -> None:
        super().__init__()
        self.alpha = alpha
        self.base_loss_func = base_loss

        self.dcor_loss_func = DistanceCorrelationLoss()

    def forward(self, inputs, intermediates, outputs, targets):
        base_loss = self.base_loss_func(outputs, targets)
        dcor_loss = self.dcor_loss_func(inputs, intermediates)
        nopeekloss = (1 - self.alpha) * base_loss + \
            self.alpha * dcor_loss
        return nopeekloss


class DistanceCorrelationLoss(torch.nn.modules.loss._Loss):
    def forward(self, input_data, intermediate_data):
        return self._dist_corr(input_data, intermediate_data)

    def _pairwise_dist(self, data):
        """culculate pairwise distance within data
           modified from https://github.com/TTitcombe/NoPeekNN

        Args:
            data: target data

        Returns:
            distance_matrix (torch.Tensor): pairwise distance matrix
        """
        n = data.size(0)
        distance_matrix = torch.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                distance_matrix[i, j] = (data[i] - data[j]).square().sum()
                distance_matrix[j, i] = distance_matrix[i, j]

        return distance_matrix

    def _dist_corr(self, X, Y):
        """culculate distance correlation between X and Y
        modified from https://github.com/tremblerz/nopeek

        Args:
            X (torch.Tensor): target data
            Y (torch.Tensor): target data

        Returns:
            dCorXY (torch.Tensor): distance correlation between X and Y
        """
        n = X.shape[0]
        a = self._pairwise_dist(X)
        b = self._pairwise_dist(Y)
        A = a - a.mean(dim=1) - a.mean(dim=1).unsqueeze(dim=1) + a.mean()
        B = b - b.mean(dim=1) - b.mean(dim=1).unsqueeze(dim=1) + b.mean()
        dCovXY = torch.sqrt((A*B).sum() / (n**2))
        dVarXX = torch.sqrt((A*A).sum() / (n**2))
        dVarYY = torch.sqrt((A*A).sum() / (n**2))
        dCorXY = dCovXY / torch.sqrt(dVarXX * dVarYY)

        return dCorXY
