from abc import ABCMeta, abstractmethod


class Attacker(metaclass=ABCMeta):
    def __init__(self, splitnn):
        """attacker against SplitNN

        Args:
            splitnn: SplitNN
        """
        self.splitnn = splitnn

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def attack(self):
        pass
