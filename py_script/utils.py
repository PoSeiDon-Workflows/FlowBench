import json
import os
import os.path as osp


def create_dir(path):
    r""" Create a dir where the processed data will be stored

    Args:
        path (str): Path to create the folder.
    """
    dir_exists = os.path.exists(path)

    if not dir_exists:
        try:
            os.makedirs(path)
            print("The {} directory is created.".format(path))
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)


def parse_adj(workflow):
    r""" Processing adjacency file.

    Args:
        workflow (str): Workflow name.

    Raises:
        NotImplementedError: No need to process the workflow `all`.

    Returns:
        tuple: (dict, list)
            dict: Dictionary of nodes.
            list: List of directed edges.
    """
    adj_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "adjacency_list_dags")
    if workflow == "all":
        raise NotImplementedError
    else:
        adj_file = osp.join(adj_folder, f"{workflow.replace('-', '_')}.json")
    adj = json.load(open(adj_file))

    if workflow == "predict_future_sales":
        nodes = {}
        for idx, node_name in enumerate(adj.keys()):
            nodes[node_name] = idx

        edges = []
        for u in adj:
            for v in adj[u]:
                edges.append((nodes[u], nodes[v]))
    else:
        # build dict of node: {node_name: idx}
        nodes = {}
        for idx, node_name in enumerate(adj.keys()):
            if node_name.startswith("create_dir_") or node_name.startswith("cleanup_"):
                node_name = node_name.split("-")[0]
                nodes[node_name] = idx
            else:
                nodes[node_name] = idx

        # build list of edges: [(target, source)]
        edges = []
        for u in adj:
            for v in adj[u]:
                if u.startswith("create_dir_") or u.startswith("cleanup_"):
                    u = u.split("-")[0]
                if v.startswith("create_dir_") or v.startswith("cleanup_"):
                    v = v.split("-")[0]
                edges.append((nodes[u], nodes[v]))

    return nodes, edges


def init_model(args):
    r""" Initiate model for PyGOD

    Args:
        args (dict): Args from argparser.

    Returns:
        object: Model object.
    """
    from random import choice

    from pygod.models import (ANOMALOUS, CONAD, DOMINANT, DONE, GAAN, GCNAE,
                              GUIDE, MLPAE, SCAN, AdONE, AnomalyDAE, Radar)
    from pyod.models.lof import LOF
    from sklearn.ensemble import IsolationForest
    if not isinstance(args, dict):
        args = vars(args)
    dropout = [0, 0.1, 0.3]
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01

    if args['dataset'] == 'inj_flickr':
        # sampling and minibatch training on large dataset flickr
        batch_size = 64
        num_neigh = 3
        epoch = 2
    else:
        batch_size = 0
        num_neigh = -1
        epoch = 300

    model_name = args['model']
    gpu = args['gpu']

    # if hasattr(args, 'epoch'):
    epoch = args.get('epoch', 200)

    if args['dataset'] == 'reddit':
        # for the low feature dimension dataset
        hid_dim = [32, 48, 64]
    else:
        hid_dim = [32, 64, 128, 256]

    if args['dataset'][:3] == 'inj':
        # auto balancing on injected dataset
        alpha = [None]
    else:
        alpha = [0.8, 0.5, 0.2]

    if model_name == "adone":
        return AdONE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'anomalydae':
        hd = choice(hid_dim)
        return AnomalyDAE(embed_dim=hd,
                          out_dim=hd,
                          weight_decay=weight_decay,
                          dropout=choice(dropout),
                          theta=choice([10., 40., 90.]),
                          eta=choice([3., 5., 8.]),
                          lr=choice(lr),
                          epoch=epoch,
                          gpu=gpu,
                          alpha=choice(alpha),
                          batch_size=batch_size,
                          num_neigh=num_neigh)
    elif model_name == 'conad':
        return CONAD(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'dominant':
        return DOMINANT(hid_dim=choice(hid_dim),
                        weight_decay=weight_decay,
                        dropout=choice(dropout),
                        lr=choice(lr),
                        epoch=epoch,
                        gpu=gpu,
                        alpha=choice(alpha),
                        batch_size=batch_size,
                        num_neigh=num_neigh)
    elif model_name == 'done':
        return DONE(hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gaan':
        return GAAN(noise_dim=choice([8, 16, 32]),
                    hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    alpha=choice(alpha),
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gcnae':
        return GCNAE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'guide':
        return GUIDE(a_hid=choice(hid_dim),
                     s_hid=choice([4, 5, 6]),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh,
                     cache_dir='./tmp')
    elif model_name == "mlpae":
        return MLPAE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size)
    elif model_name == 'lof':
        return LOF()
    elif model_name == 'if':
        return IsolationForest()
    elif model_name == 'radar':
        return Radar(lr=choice(lr), gpu=gpu)
    elif model_name == 'anomalous':
        return ANOMALOUS(lr=choice(lr), gpu=gpu)
    elif model_name == 'scan':
        return SCAN(eps=choice([0.3, 0.5, 0.8]), mu=choice([2, 5, 10]))
