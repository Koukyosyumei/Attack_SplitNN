# Attacking SplitNN

Attacking_SplitNN is a Python library, and it allows you to easily experiment with various combinations of attack and defense algorithms against SplitNN within PyTorch and scikit-learn.

## Install

        pip install git+https://github.com/Koukyosyumei/Attack_SplitNN

## SplitNN

You can easily create two-SplitNN with this package as follows.\
The client only has input data, and the server has only labels.


        Examples:
            model_1 = FirstNet()
            model_1 = model_1.to(device)

            model_2 = SecondNet()
            model_2 = model_2.to(device)

            opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)
            opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)

            criterion = nn.BCELoss()

            client = Client(model_1, opt_1)
            server = Server(model_2, opt_2, criterion)

            sn = SplitNN(client, server, device=device)
            sn.fit(train_loader, 3, metric=torch_auc)

## Label Leakage

Label Leakage (Oscar et al.) is one of the weaknesses in SplitNN, and it means that the intermediate gradient which the server sends to the client may be able to allow the client to extract the private ground-truth labels that the server has. We currently support measuring leak_auc that measures how well the l2 norm of the communicated gradient can predict y by the AUC of the ROC curve. Also, we allow you to avoid this leakage with the defense method called max norm.

[notebook](examples/Label_Leakage.ipynb)\
[paper](https://arxiv.org/abs/2102.08504)

## Membership Inference Attack

It is proved that the attacker can determine whether the victim used a record to train the target model, and this attack is called Membership Inference Attack (MIA). In SplitNN, one of the possible situations is that the malicious server executes a membership inference attack against the client.

[notebook](examples/Membershio_Inference_Attack.ipynb)\
[paper](https://ieeexplore.ieee.org/document/9302683)

## Model Inversion

The general purpose of Model Inversion is to reconstruct the training data from models. FSHA is a kind of Model Inversion Attack for SplitNN, and it allows the server to rebuild the training data that the client wants to hide. Also, the malicious client uses this method to fetch other clients' training data.

[notebook](examples/FSHA_Model_Inversion_FSHA.ipynb)\
[paper](https://arxiv.org/abs/2012.02670)

*note*\
We currently don't support the same interface (Client, Server, SplitNN, etc.) like other attacks for this method.

### Server side

### Client side

## Poisoning Attack

## Evasion Attack