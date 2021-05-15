# Attacking SplitNN

Attacking_SplitNN is a Python library, and it allows you to easily experiment with various combinations of attack and defense algorithms against SplitNN within PyTorch and scikit-learn.

## Install

        pip install git+https://github.com/Koukyosyumei/Attack_SplitNN

## SplitNN

You can easily create two-SplitNN with this package as follows.\
The client only has input data, and the server has only labels.
This package implements SplitNN as the custom torch.nn.modules, so you
can train SplitNN like the normal torch models.



        Examples:
                model_1 = FirstNet()
                model_1 = model_1.to(device)

                model_2 = SecondNet()
                model_2 = model_2.to(device)

                opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)
                opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)

                criterion = nn.BCELoss()

                client = Client(model_1)
                server = Server(model_2)

                splitnn = SplitNN(client, server, opt_1, opt_2)

                splitnn.train()
                for epoch in range(3):
                epoch_loss = 0
                epoch_outputs = []
                epoch_labels = []
                for i, data in enumerate(train_loader):
                        splitnn.zero_grads()
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = splitnn(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        epoch_loss += loss.item() / len(train_loader.dataset)

                        epoch_outputs.append(outputs)
                        epoch_labels.append(labels)

                        splitnn.backward()
                        splitnn.step()

                print(epoch_loss, torch_auc(torch.cat(epoch_labels),
                                                torch.cat(epoch_outputs)))

## Attack

|                                  | type                        | example                                                |
| -------------------------------- | --------------------------- | ------------------------------------------------------ |
| Intermidiate Level Attack        | evasion attack              | [notebook](examples/IntermidiateLevelAttack.ipynb)     |
| Norm Attack                      | label leakage attack        | [notebook](examples/Label_Leakage.ipynb)               |
| Transfer Inherit Attack          | membership inference attack | [notebook](examples/Membershio_Inference_Attack.ipynb) |
| Black Box Model Inversion Attack | model inversion attack      | [notebook](examples/Black_Box_Model_Inversion.ipynb)   |


## Defense

|          | example                                  |
| -------- | ---------------------------------------- |
| Max Norm | [notebook](examples/Label_Leakage.ipynb) |
| NoPeek   | [notebook](examples/NoPeekLoss.ipynb)    |

## License

This software is released under the MIT License, see LICENSE.txt.
