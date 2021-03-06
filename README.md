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

## Label Leakage

Label Leakage (Oscar et al.) is one of the weaknesses in SplitNN, and it means that the intermediate gradient which the server sends to the client may be able to allow the client to extract the private ground-truth labels that the server has. We currently support measuring leak_auc that measures how well the l2 norm of the communicated gradient can predict y by the AUC of the ROC curve. Also, we allow you to avoid this leakage with the defense method called max norm.

*Oscar Li ??? Jiankai Sun ??? Xin Yang ??? Weihao Gao ??? Hongyi Zhang ??? Junyuan Xie ??? Virginia Smith ??? Chong Wang. Label Leakage and Protection in Two-party Split Learning, https://arxiv.org/abs/2102.08504, 2021*

[notebook](examples/Label_Leakage.ipynb)

## Membership Inference Attack

It is proved that the attacker can determine whether the victim used a record to train the target model, and this attack is called Membership Inference Attack (MIA). In SplitNN, one of the possible situations is that the malicious server executes a membership inference attack against the client.

 *H. Chen et al., (2020). Practical Membership Inference Attack Against Collaborative Inference in Industrial IoT, in IEEE Transactions on Industrial Informatics, DOI: 10.1109/TII.2020.3046648. https://ieeexplore.ieee.org/document/9302683*

[notebook](examples/Membershio_Inference_Attack.ipynb)

## Model Inversion

The general purpose of Model Inversion is to reconstruct the training data from models. FSHA is a kind of Model Inversion Attack for SplitNN, and it allows the server to rebuild the training data that the client wants to hide. Also, the malicious client uses this method to fetch other clients' training data.

 *D Pasquini, G Ateniese, M Bernaschi. (2020). Unleashing the Tiger: Inference Attacks on Split Learning. https://arxiv.org/abs/2012.02670*

[notebook](examples/FSHA_Model_Inversion_FSHA.ipynb)

*note*\
We currently don't support the same interface (Client, Server, SplitNN, etc.) like other attacks for this method.

### Server side

### Client side

## Poisoning Attack

## Evasion Attack

## License

This software is released under the MIT License, see LICENSE.txt.
