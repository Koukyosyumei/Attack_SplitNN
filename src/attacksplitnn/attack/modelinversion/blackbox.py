import torch


class Black_Box_Model_Inversion:
    def __init__(self, splitnn, attacker_model, attacker_optimizer):
        """class that implement black box model inversion

        Args:
            attacker_model (torch model):
            attacker_optimizer (torch optimizer):

        Attributes:
            attacker_model (torch model):
            attacker_optimizer (torch optimizer):
        """
        self.splitnn = splitnn
        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer

    def attack(self, dataloader_for_attacker, epoch):

        for i in range(epoch):
            for data, _ in dataloader_for_attacker:
                self.attacker_optimizer.zero_grad()

                target_outputs = self.splitnn.client(data)

                attack_outputs = self.attacker_model(target_outputs)

                loss = ((data - attack_outputs)**2).mean()

                loss.backward()
                self.attacker_optimizer.step()

            print(f"epoch {i}: reconstruction_loss {loss.item()}")

    def predict(self, dataloader_target):
        attack_results = []

        for data, _ in dataloader_target:
            target_outputs = self.splitnn.client(data)
            recreated_data = self.attacker_model(target_outputs)
            attack_results.append(recreated_data)

        return torch.cat(attack_results)
