# FedPatch: Bridging the Performance Gap between Gradient and Model Aggregation in Semi-Asynchronous Federated Learning

The repository is the implementation of the PAPER: `FedPatch: Bridging the Performance Gap between Gradient and Model Aggregation in Semi-Asynchronous Federated Learning`. 
Main functionalities include: 
1. splitting the `CIFAR-10` dataset, `CIFAR-100` dataset, `FEMNIST` dataset, `Sentiment140` dataset and `Shakespeare` dataset into IID and non-IID distributions in the federated setting.
2. implementation of FedAvg and FedSGD under Synchronous Fedrated Learning
3. implementation of FedAvg and FedSGD under Semi-Asynchronous Fedrated Learnin
4. implementation of FedPatch
5. implementation of SOTA algorithms including WKAFL[1] and M-step-FedAsync[2]

Here is a simple user manual. Please **feel free** to ask any questions if you encounter any problems during use.

## Preparation
### Installing necessary Python libraries and their versions
1. numpy==1.19.5
2. torch==1.10.0+cu102
3. torchtext==0.11.0
4. torchvision==0.11.0+cu102

### Choose Strategies
Getting started with choose a strategy in the `utils` such as `afl_avg.py` and copy it to your current folder.

### Modify the configuration file
Change the configure file `conf.json` based on your tasks, where:
```
"model_name": The federated model you choose. We provide four different models to choose from, including `resnet18`/`vgg16`/`CNN`/`LSTM`.
              *Remark: you can change the model structure in he file `resnet_model.py`.
"no_models": The number of clients, default by 100.
"type": The datasets used for the training tasks. We provide four different datasets to choose from, including `cifar10`/`cifar100`/`mnist`/`Shakespeare`.
        *Remark1: if you want to use `Sentiment140` dataset, please choose strategy file such as `afl_avg_senti.py` in utils.
        *Remark2: `mnist` refers to FEMNIST dataset.
"CLASS_NUM": The total number of labels in the dataset used for training tasks. Eg: `10` for `cifar10`, `100` for `cifar100`, `62` for `mnist` and so on.
"clip": Clip bound when training models, default by 20.
"global_epochs": The number of epoches required for the whole training, default by 400.
"local_epochs": The number of epoches required for local training for each client, default by 2.
"k": For SFL, this parameter represents the number of activated clients; for SAFL, it represents the amount of data needed for aggregation.
"batch_size": The size of batches for dataloader, default by 50.
"non_iid": The distributions used for pre-processing data. We provide four different datasets to choose from, including `HeteroDiri`/`Shards`/`Unbalance_Diri`/`iid`.
"local_lr": The local learning rate for local training, default by 1e-1.
"global_lr": The global learning rate for local training under gradient aggregation such as FedSGD and WKAFL, default by 1e-1.
"local_momentum": The local momentum for local training, default by 0.
"redistribution": Whether you want to redistribute the dataset. Please set it as "y" when you first use a specific distribution for a specific dataset.
"alpha": The distribution parameter.
"resource_max": The maximum resource among clients.
"server_eval_size": The hyper-parameter using in FedPatch.
"stage_bound": The hyper-parameter using in WKAFL.
"sim_bound": The hyper-parameter using in WKAFL.
"beta": The hyper-parameter using in WKAFL.
"norm_bound" The hyper-parameter using in WKAFL.
```

### Download the dataset
If you choose dataset as `cifar10`/`cifar100`/`mnist`, you need to ensure that the parent directory of the files you run is accessible and modifiable. 

We will invoke PyTorch to create a folder named "data" in the parent directory and download the necessary training/testing data.

If you choose dataset as `Shakespeare`, please unzip `all_data.zip`(you can find it in folder `data_partition`) in your current directory.

If you choose dataset as `Sentiment140`, please unzip `train-processed.csv`(you can find it in folder `data_partition`) in your current directory.

### Generate the resources for each client
After modifying the configuration file, please ensure that you have run `resource_generate.py` to generate the resources for each client when you choose **SAFL**.
What's more, you need to change the path in utils file (such as `afl_avg.py`) into your own resource file.


Meanwhile, we provide a 
## Start the federated learning

## Get results and evaluation


[1]: Z. Zhou, Y. Li, X. Ren, and S. Yang, “Towards efficient and stable k-asynchronous federated learning with unbounded stale gradients on non-iid data,” IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 12, pp. 3291–3305, 2022

[2]: X. Wu and C.-L. Wang, “Kafl: Achieving high training efficiency for fast-k asynchronous federated learning,” in 2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS). IEEE, 2022, pp. 873–883.
