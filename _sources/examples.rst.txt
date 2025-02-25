Examples
========

Load Dataset
------------

- load data as graphs in ``pytorch_geometric`` format:

  .. code-block:: python

    from flowbench.dataset import FlowDataset
    dataset = FlowDataset(root="./", name="montage")
    data = dataset[0]

  The ``data`` contains the structural information by accessing ``data.edge_index``, and node feature information ``data.x``.

- load data as tabular data in ``pytorch`` format:

  .. code-block:: python

    from flowbench.dataset import FlowDataset
    dataset = FlowDataset(root="./", name="montage")
    data = dataset[0]
    Xs = data.x
    ys = data.y

  Unlike the graph ``pyg.data``, the ``data`` only contains the node features.

- load data as tabular data in ``numpy`` format:

  .. code-block:: python

    from flowbench.dataset import FlowDataset
    dataset = FlowDataset(root="./", name="montage")
    data = dataset[0]
    Xs = data.x.numpy()
    ys = data.y.numpy()

  This is the same as the previous one, but the data is in ``numpy`` format, which is typically used in the models from ``sklearn`` and ``xgboost``.

- load text data with ``huggingface`` interface.
  We have uploaded our parsed text data in the ``huggingface`` dataset. You can load the data with the following code:
  
  .. code-block:: python

    from datasets import load_dataset
    dataset = load_dataset("cshjin/poseidon", "1000genome")

  The dataset is in the format of ``dict`` with keys ``train``, ``test``, and ``validation``. 

PyOD Models
-----------

===================  ================  ======================================================================================================  =====  =================================================== 
Type                 Abbr              Algorithm                                                                                               Year   Class                                               
===================  ================  ======================================================================================================  =====  =================================================== 
Probabilistic        ABOD              Angle-Based Outlier Detection                                                                           2008   :class:`flowbench.unsupervised.pyod.ABOD`
Probabilistic        KDE               Outlier Detection with Kernel Density Functions                                                         2007   :class:`flowbench.unsupervised.pyod.KDE`
Probabilistic        GMM               Probabilistic Mixture Modeling for Outlier Analysis                                                            :class:`flowbench.unsupervised.pyod.GMM`
Linear Model         PCA               Principal Component Analysis (the sum of weighted projected distances to the eigenvector hyperplanes)   2003   :class:`flowbench.unsupervised.pyod.PCA`
Linear Model         OCSVM             One-Class Support Vector Machines                                                                       2001   :class:`flowbench.unsupervised.pyod.OCSVM`
Linear Model         LMDD              Deviation-based Outlier Detection (LMDD)                                                                1996   :class:`flowbench.unsupervised.pyod.LMDD`
Proximity-Based      LOF               Local Outlier Factor                                                                                    2000   :class:`flowbench.unsupervised.pyod.LOF`
Proximity-Based      CBLOF             Clustering-Based Local Outlier Factor                                                                   2003   :class:`flowbench.unsupervised.pyod.CBLOF`
Proximity-Based      kNN               k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score)                 2000   :class:`flowbench.unsupervised.pyod.KNN`
Outlier Ensembles    IForest           Isolation Forest                                                                                        2008   :class:`flowbench.unsupervised.pyod.IForest`
Outlier Ensembles    INNE              Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles                                      2018   :class:`flowbench.unsupervised.pyod.INNE`
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                       2019   :class:`flowbench.unsupervised.pyod.LSCP`
===================  ================  ======================================================================================================  =====  =================================================== 

- Example of using `GMM`

  .. code-block:: python

    from flowbench.pyod import GMM
    from flowbench.dataset import FlowDataset
    dataset = FlowDataset(root="./", name="1000genome")
    Xs = ds.x.numpy()
    clf = GMM()
    clf.fit(Xs)
    y_pred = clf.predict(Xs)

  - Detailed example in ``example/demo_pyod.py``

PyGOD Models
------------

=========== ==================  =====    ==============================================
Type        Abbr                Year        Class
=========== ==================  =====    ==============================================
Clustering  SCAN                2007              :class:`flowbench.unsupervised.pygod.SCAN`
GNN+AE      GAE                 2016             :class:`flowbench.unsupervised.pygod.GAE`
MF          Radar               2017              :class:`flowbench.unsupervised.pygod.Radar`
MF          ANOMALOUS           2018              :class:`flowbench.unsupervised.pygod.ANOMALOUS`
MF          ONE                 2019              :class:`flowbench.unsupervised.pygod.ONE`
GNN+AE      DOMINANT            2019             :class:`flowbench.unsupervised.pygod.DOMINANT`
MLP+AE      DONE                2020             :class:`flowbench.unsupervised.pygod.DONE`
MLP+AE      AdONE               2020             :class:`flowbench.unsupervised.pygod.AdONE`
GNN+AE      AnomalyDAE          2020             :class:`flowbench.unsupervised.pygod.AnomalyDAE`
GAN         GAAN                2020             :class:`flowbench.unsupervised.pygod.GAAN`
GNN+AE      DMGD                2020             :class:`flowbench.unsupervised.pygod.DMGD`
GNN         OCGNN               2021             :class:`flowbench.unsupervised.pygod.OCGNN`
GNN+AE+SSL  CoLA                2021             :class:`flowbench.unsupervised.pygod.CoLA`
GNN+AE      GUIDE               2021             :class:`flowbench.unsupervised.pygod.GUIDE`
GNN+AE+SSL  CONAD               2022             :class:`flowbench.unsupervised.pygod.CONAD`
GNN+AE      GADNR               2024             :class:`flowbench.unsupervised.pygod.GADNR`
=========== ==================  =====    ==============================================


- Example of using `GMM`

  .. code-block:: python

    from flowbench.unsupervised.pygod import GAE
    from flowbench.dataset import FlowDataset
    dataset = FlowDataset(root="./", name="1000genome")
    data = dataset[0]
    clf = GAE()
    clf.fit(data)

  - Detailed example in ``example/demo_pygod.py``


Supervised Models
-----------------

- Example of using `MLP`
  
    .. code-block:: python
  
      from flowbench.supervised.mlp import MLPClassifier
      from flowbench.dataset import FlowDataset
      dataset = FlowDataset(root="./", name="1000genome")
      data = dataset[0]
      clf = MLPClassifier()
      clf.fit(data)
  
    - Detailed example in ``example/demo_supervised.py``

Supervised fine-tuned LLMs
--------------------------

- Example of using LoRA (Low-rank Adaptation) for supervised fine-tuned LLMs:

  .. code-block:: python

    from peft import LoraConfig
    dataset = load_dataset("cshjin/poseidon", "1000genome")
    # data processing
    ...
    # LoRA config
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    training_args = TrainingArgument(...)
    # LoRA trainer
    trainer = Trainer(peft_model, ...)
    trainer.train()
    ...

  - Detailed example in ``example/demo_sft_lora.py``