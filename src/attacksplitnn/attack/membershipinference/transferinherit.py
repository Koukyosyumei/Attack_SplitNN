import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ...splitnn.model import SplitNN
from ..attacker import AbstractAttacker


class TransferInherit(AbstractAttacker):
    def __init__(self,
                 splitnn,
                 shadow_client,
                 shadow_client_optimizer,
                 attacker_clf,
                 device="cpu"):
        """class to execure membership inference attack against SplitNN
           reference https://ieeexplore.ieee.org/document/9302683

        Args:
            splitnn: target splitnn model
            shadow_client (attack_splitnn.splitnn.Client): shadow client that
                                   the server prepares to mimic victim client
            shadow_client_optimizer (torch optimizer):
            attacker_clf (sklearn classfier): attacker's classifier for binary
                               classification for membership inference attack
            device (str): device type (default 'cpu')

        Attributes:
            shadow_client (attack_splitnn.splitnn.Client): shadow client that
                                   the server prepares to mimic victim client
            server (attack_splitnn.splitnn.Server): the server who want to
                                   execute membership inference attack
            shadow_client_optimizer (torch optimizer):
            attacker_clf (sklearn classfier): attacker's classifier for binary
                               classification for membership inference attack
            device (str): device type (default 'cpu')

            attacker_X (np.array):
            attacker_y (np.array):
            attacker_X_train (np.array):
            attacker_X_test (np.array):
            attacker_y_train (np.array):
            attacker_y_test (np.array):

        Examples
        """
        super().__init__(splitnn)
        self.shadow_client = shadow_client
        self.shadow_client_optimizer = shadow_client_optimizer
        self.server = splitnn.server
        self.attacker_clf = attacker_clf
        self.device = device

        self.shadow_splitnn = SplitNN(self.shadow_client,
                                      self.server,
                                      self.shadow_client_optimizer,
                                      None)

        self.attacker_X = None
        self.attacker_y = None
        self.attacker_X_train = None
        self.attacker_X_test = None
        self.attacker_y_train = None
        self.attacker_y_test = None

    def fit(self,
            member_shadowloader,
            nonmember_shadowloader,
            shadow_epochs,
            shadow_criterion,
            shadow_metric=None,
            attack_dataset_split=0.3,
            random_state=None):
        """execure whole process of membership inference attack

        Args:
            member_shadowloader (torch dataloader):
            nonmember_shadowloader (torch dataloader):
            shadow_epochs (int):
            shadow_criterion (function)
            shadow_metric (function):
            attack_dataset_split (float):
            random_state (int):
        """

        # train shadow model
        print("start training shadow model")
        self._fit_shadow_model(member_shadowloader,
                               shadow_criterion,
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

        print("Done")

    def attack(self, x):
        pred_proba = self.attacker_clf.predict_proba(x)
        return pred_proba

    def _fit_shadow_model(self, member_shadowloader,
                          criterion,
                          epochs, metric=None):
        """train shadow model

        Args:
            member_shadowloader (torch dataloader):
            epochs (int):
            metric (function):
        """
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_labels = []
            epoch_outputs = []
            for i, data in enumerate(member_shadowloader, 0):

                self.shadow_splitnn.client_optimizer.zero_grad()

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.shadow_splitnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                epoch_loss += loss / len(member_shadowloader.dataset)
                epoch_outputs.append(outputs)
                epoch_labels.append(labels)

                self.shadow_splitnn.backward()
                self.shadow_splitnn.client_optimizer.step()

            epoch_outputs = torch.cat(epoch_outputs)
            epoch_labels = torch.cat(epoch_labels)

            self._print_metric(epoch, epoch_loss,
                               epoch_outputs, epoch_labels,
                               metric=metric)

    def _predict_shadow_model(self, dataloader):
        """predict by shadow_model

        Args:
            dataloader

        Returns:
            outputs
        """
        with torch.no_grad():
            train_shadow = []
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.shadow_splitnn(inputs)
                train_shadow.append(output)

        outputs = torch.cat(train_shadow)
        return outputs

    def _create_dataset_for_attacker(self, member_shadowloader,
                                     nonmember_shadowloader,
                                     test_size=0.3,
                                     random_state=None):
        """create membership dataset for attacker

        Args:
            member_shadowloader
            nonmember_shadowloader
            test_size
            random_state
        """
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
        """train attacker's classifier
        """
        self.attacker_clf.fit(self.attacker_X_train, self.attacker_y_train)

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
