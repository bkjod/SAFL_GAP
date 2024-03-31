# FedPatch: Bridging the Performance Gap between Gradient and Model Aggregation in Semi-Asynchronous Federated Learning

The repository is the implementation of the PAPER: 'FedPatch: Bridging the Performance Gap between Gradient and Model Aggregation in Semi-Asynchronous Federated Learning'. 
Main functionalities include: 
1. splitting the 'CIFAR-10' dataset, 'CIFAR-100' dataset, 'FEMNIST' dataset, 'Sentiment140' dataset and 'Shakespeare' dataset into IID and non-IID distributions in the federated setting.
2. implementation of FedAvg and FedSGD under Synchronous Fedrated Learning
3. implementation of FedAvg and FedSGD under Semi-Asynchronous Fedrated Learnin
4. implementation of FedPatch
5. implementation of SOTA algorithms including WKAFL[@Zhou] and M-step-FedAsync[@Wu]

[@Zhou]: Z. Zhou, Y. Li, X. Ren, and S. Yang, “Towards efficient and stable k-asynchronous federated learning with unbounded stale gradients on non-iid data,” IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 12, pp. 3291–3305, 2022
[@Wu]: X. Wu and C.-L. Wang, “Kafl: Achieving high training efficiency for fast-k asynchronous federated learning,” in 2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS). IEEE, 2022, pp. 873–883.
