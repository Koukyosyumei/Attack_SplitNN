import torch


class mia_transfer_inherit:
    def __init__(self, shadow_client,
                 server,
                 device="cpu"):

        self.shadow_client = shadow_client
        self.server = server
        self.device = device

        self.attacker_X_train_shadow = None
        self.attacker_X_test_shadow = None
        self.attacker_y_train_shadow = None
        self.attacker_y_test_shadow = None

    def fit(self, member_shadowloader, nonmember_shadowloader):
        pass

    def _fit_shadow_model(self, member_shadowloader, epochs, metric=None):
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_labels = []
            epoch_outputs = []
            for i, data in enumerate(member_shadowloader, 0):

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

                epoch_loss += loss / len(member_shadowloader.dataset)
                epoch_outputs.append(outputs)
                epoch_labels.append(labels)

            epoch_outputs = torch.cat(epoch_outputs)
            epoch_labels = torch.cat(epoch_labels)

            self._print_metric(epoch, epoch_loss,
                               epoch_outputs, epoch_labels,
                               metric=metric)

    def _create_dataset_for_attacker(self, member_shadowloader,
                                     nonmember_shadowloader):
        self.attacker_X_train_shadow = self._predict_shadow_model(
            member_shadowloader)
        self.attacker_X_test_shadow = self._predict_shadow_model(
            nonmember_shadowloader)
        self.attacker_y_train_shadow = torch.ones(
            self.attacker_X_train_shadow.shape[0])
        self.attacker_y_test_shadow = torch.zeros(
            self.attacker_X_test_shadow.shape[0])

    def _predict_shadow_model(self, dataloader):
        with torch.no_grad():
            train_shadow = []
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # execute client - feed forward network
                intermediate = self.shadow_client(inputs)
                remote_intermidiate = intermediate.detach().requires_grad_()
                # execute server - feed forward network
                output = self.server(remote_intermidiate)

                train_shadow.append(output)

        outputs = torch.cat(train_shadow)
        return outputs

    def _print_metric(self, epoch,
                      epoch_loss,
                      epoch_outputs, epoch_labels,
                      metric=None):
        if metric is not None:
            m = metric(epoch_labels, epoch_outputs)
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}, metric {m}")

        else:
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}")
