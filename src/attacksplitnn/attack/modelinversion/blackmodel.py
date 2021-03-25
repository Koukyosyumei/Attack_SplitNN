import torch


class Black_Box_Model_Inversion:
    def __init__(self, attacker_model, attacker_optimizer):
        """class that implement black box model inversion

        Args:
            attacker_model (torch model):
            attacker_optimizer (torch optimizer):

        Attributes:
            attacker_model (torch model):
            attacker_optimizer (torch optimizer):
        """

        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer

    def fit(self, splitnn, dataloader_for_attacker, epoch):
        client_side_model = splitnn.client.client_model

        for i in range(epoch):
            for data, _ in dataloader_for_attacker:
                self.attacker_optimizer.zero_grad()

                target_outputs = client_side_model(data)

                attack_outputs = self.attacker_model(target_outputs)

                loss = ((data - attack_outputs)**2).mean()

                loss.backward()
                self.attacker_optimizer.step()

            print(f"epoch {i}: reconstruction_loss {loss.item()}")

    def attack(self, splitnn, dataloader_target):
        client_side_model = splitnn.client.client_model
        attack_results = []

        for data, _ in dataloader_target:
            target_outputs = client_side_model(data)
            recreated_data = self.attacker_model(target_outputs)
            attack_results.append(recreated_data)

        return torch.cat(attack_results)
