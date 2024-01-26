import os.path as osp
from flowbench.dataset_v2 import FlowBench

ROOT = osp.join("/tmp", "data", "poseidon")


def test_1000genome():
    dataset = FlowBench(ROOT, "1000genome", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.name == "1000genome"


def test_montage():
    dataset = FlowBench(ROOT, "montage", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.name == "montage"


def test_predict_future_sales():
    dataset = FlowBench(ROOT, "predict_future_sales", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.name == "predict_future_sales"


def test_casa_wind_full():
    dataset = FlowBench(ROOT, "casa_wind_full", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.name == "casa_wind_full"


def test_casa_nowcast_full():
    dataset = FlowBench(ROOT, "casa_nowcast_full", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.name == "casa_nowcast_full"


def test_variant_calling():
    dataset = FlowBench(ROOT, "variant_calling", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.name == "variant_calling"


def test_force_reprocess():
    dataset = FlowBench(ROOT, "1000genome", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.force_reprocess

    dataset = FlowBench(ROOT, "1000genome", force_reprocess=False, log=True)
    print(dataset)
    assert not dataset.force_reprocess


def test_binary_labels():
    dataset = FlowBench(ROOT, "1000genome", binary_labels=True, log=True)
    print(dataset)
    assert dataset.binary_labels

    dataset = FlowBench(ROOT, "1000genome", binary_labels=False, log=True)
    print(dataset)
    assert not dataset.binary_labels


def test_log():
    dataset = FlowBench(ROOT, "1000genome", log=True)
    assert dataset.log


def test_node_level():
    dataset = FlowBench(ROOT, "1000genome", node_level=True)
    assert dataset.node_level
    dataset = FlowBench(ROOT, "1000genome", node_level=False)
    assert not dataset.node_level


def test_pre_filter():
    # TODO: add test case to filter data by label
    pass


# test_1000genome()
# test_montage()
# test_predict_future_sales()
# test_casa_wind_full()
# test_casa_nowcast_full()
# test_variant_calling()

# # test_casa_wind_full()
# dataset = FlowBench(ROOT, "1000genome", force_reprocess=True, log=False,
#                     pre_filter=lambda data: "cpu" in data.label or "normal" in data.label)
# print(dataset)
# print(dataset.x)
# print(dataset.y)

# print(torch.bincount(dataset.y))
# data, y = dataset.to_graph_level()
# print(data)
# print(y)