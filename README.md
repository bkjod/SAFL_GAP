# FedPatch: Bridging the Performance Gap between Gradient and Model Aggregation in Semi-Asynchronous Federated Learning

The repository is the implementation of the PAPER: `FedPatch: Bridging the Performance Gap between Gradient and Model Aggregation in Semi-Asynchronous Federated Learning`. 
Main functionalities include: 
1. splitting the `CIFAR-10` dataset, `CIFAR-100` dataset, `FEMNIST` dataset, `Sentiment140` dataset and `Shakespeare` dataset into IID and non-IID distributions in the federated setting.
2. implementation of FedAvg and FedSGD under Synchronous Fedrated Learning
3. implementation of FedAvg and FedSGD under Semi-Asynchronous Fedrated Learnin
4. implementation of FedPatch
5. implementation of SOTA algorithms including WKAFL[1] and M-step-FedAsync[2]

## Preparation
### Choose Strategies
Getting started with choose a strategy in the `utils` such as `afl_avg.py` and copy it to your current folder.
### Modify the configure file
Change the configure file based on your tasks, where:
```
"model_name": The federated model you choose. We provide four different models to choose from, including `resnet18`/`vgg16`/`CNN`/`LSTM`. Remark: you can change the model structure in he file `resnet_model.py`.
"no_models": The number of clients, default by 100.
"type": The datasets used for the training tasks. We provide four different datasets to choose from, including `cifar10`/`cifar100`/`mnist`/`Shakespeare`. Remark: if you want to use `Sentiment140` dataset, please choose strategy file such as `afl_avg_senti.py` in utils.

```

[1]: Z. Zhou, Y. Li, X. Ren, and S. Yang, “Towards efficient and stable k-asynchronous federated learning with unbounded stale gradients on non-iid data,” IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 12, pp. 3291–3305, 2022

[2]: X. Wu and C.-L. Wang, “Kafl: Achieving high training efficiency for fast-k asynchronous federated learning,” in 2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS). IEEE, 2022, pp. 873–883.
