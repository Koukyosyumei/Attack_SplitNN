## Label Leakage

Label Leakage (Oscar et al.) is one of the weaknesses in SplitNN, and it means that the intermediate gradient which the server sends to the client may be able to allow the client to extract the private ground-truth labels that the server has. We currently support measuring leak_auc that measures how well the l2 norm of the communicated gradient can predict y by the AUC of the ROC curve. Also, we allow you to avoid this leakage with the defense method called max norm.

*Oscar Li • Jiankai Sun • Xin Yang • Weihao Gao • Hongyi Zhang • Junyuan Xie • Virginia Smith • Chong Wang. Label Leakage and Protection in Two-party Split Learning, https://arxiv.org/abs/2102.08504, 2021*

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
