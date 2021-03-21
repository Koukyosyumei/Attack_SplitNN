import numpy as np
import torch
from sklearn.model_selection import train_test_split


class SplitMIA:
    def __init__(self,
                 shadow_client,
                 server,
                 attacker_clf,
                 device="cpu"):

        self.shadow_client = shadow_client
        self.server = server
        self.attacker_clf = attacker_clf
        self.device = device

        self.attacker_dataset_x = None
        self.attacker_dataset_y = None
        self.attacker_X_train = None
        self.attacker_X_test = None
        self.attacker_y_train = None
        self.attacker_y_test = None

    def fit(self,
            member_shadowloader,
            nonmember_shadowloader,
            shadow_epochs,
            shadow_metric=None,
            attack_dataset_split=0.3,
            random_state=None):

        # train shadow model
        print("start training shadow model")
        self._fit_shadow_model(member_shadowloader,
                               shadow_epochs,
                               metric=shadow_metric)
        # create dataset for attacker from shadow_model
        print("start creating dataset for attacker")
        self._create_dataset_for_attacker(member_shadowloader,
                                          nonmember_shadowloader,
                                          test_size=attack_dataset_split,
                                          random_state=random_state)
        # train attacker classifier
        print("start training attacker")
        self._fit_attacker_clf()

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

    def _create_dataset_for_attacker(self, member_shadowloader,
                                     nonmember_shadowloader,
                                     test_size=0.3,
                                     random_state=None):
        attacker_X_train_shadow = self._predict_shadow_model(
            member_shadowloader)
        attacker_X_test_shadow = self._predict_shadow_model(
            nonmember_shadowloader)
        attacker_y_train_shadow = torch.ones(
            attacker_X_train_shadow.shape[0])
        attacker_y_test_shadow = torch.zeros(
            attacker_X_test_shadow.shape[0])

        # convert torch.Tensor to np.array
        attacker_X_train_shadow = np.array(attacker_X_train_shadow)
        attacker_X_test_shadow = np.array(attacker_X_test_shadow)
        attacker_y_train_shadow = np.array(attacker_y_train_shadow)
        attacker_y_test_shadow = np.array(attacker_y_test_shadow)

        # concatenate
        self.attacker_X = np.concatenate(
            [attacker_X_train_shadow, attacker_X_test_shadow])
        self.attacker_y = np.concatenate(
            [attacker_y_train_shadow, attacker_y_test_shadow])

        # split dataset
        self.attacker_X_train, self.attacker_X_test,\
            self.attacker_y_train, self.attacker_y_test = train_test_split(
                self.attacker_X,
                self.attacker_y,
                test_size=test_size,
                shuffle=True,
                random_state=random_state)

    def _fit_attacker_clf(self):
        self.attacker_clf.fit(self.attacker_X_train, self.attacker_y_train)

    def _predict_proba_attacker_clf(self, x):
        pred_proba = self.attacker_clf.predict_proba(x)
        return pred_proba

    def _print_metric(self, epoch,
                      epoch_loss,
                      epoch_outputs, epoch_labels,
                      metric=None):
        if metric is not None:
            m = metric(epoch_labels, epoch_outputs)
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}, metric {m}")

        else:
            print(f"epoch {epoch+1}, loss {epoch_loss:.5}")

    def save(self):
        pass
