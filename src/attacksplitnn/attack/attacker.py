from abc import ABCMeta, abstractmethod


class AbstractAttacker(metaclass=ABCMeta):
    def __init__(self, splitnn):
        """attacker against SplitNN

        Args:
            splitnn: SplitNN
        """
        self.splitnn = splitnn

    def fit(self):
        pass

    @abstractmethod
    def attack(self):
        pass
