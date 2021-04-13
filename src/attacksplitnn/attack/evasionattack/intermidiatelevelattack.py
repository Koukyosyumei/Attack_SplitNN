import copy


class IntermidiateLevelAttack:
    def __init__(self, splitnn):
        self.splitnn = splitnn

    def attack(self,
               original_dataloader,
               adversarial_example_from_original_dataloader,
               epochs, lr, alpha, epsilon):
        new_adversarial_example = copy.copy(
            adversarial_example_from_original_dataloader)

        for i in range(epochs):
            for original_data, adversarial_example, new_adversarial_example in\
                zip(original_dataloader,
                    adversarial_example_from_original_dataloader,
                    new_adversarial_example):

                delta_y_prime = self.splitnn.client(adversarial_example) - \
                    self.splitnn.client(original_data)
                delta_y_doubleprime = self.splitnn.client() - \
                    self.splitnn.client(original_data)

                delta_y_prime_norm = delta_y_prime.pow(2).sqrt().sum()
                delta_y_doubleprime_norm = delta_y_doubleprime.pow(
                    2).sqrt().sum()

                L = (delta_y_doubleprime / delta_y_doubleprime_norm) *\
                    (delta_y_prime / delta_y_prime_norm) +\
                    alpha * (delta_y_doubleprime_norm / delta_y_prime_norm)
                L.backward()

                grad_new_adversarial_example_L = new_adversarial_example.grad()
                new_adversarial_example += lr * grad_new_adversarial_example_L
                new_adversarial_example = self._clip_epsilon(
                    new_adversarial_example, original_data, epsilon) +\
                    original_data

    def _clip_epsilon(self, x, y, epsilon):
        diff = x - y
        return diff
