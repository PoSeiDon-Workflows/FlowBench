from flowbench.supervised import MLPClassifier
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def test_mlp_classifier_initialization():
    in_channels = 10
    out_channels = 2
    hidden_channels = 64
    num_layers = 4
    lr = 0.01
    dropout = 0.2

    mlp_classifier = MLPClassifier(
        in_channels,
        out_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        lr=lr,
        dropout=dropout)

    assert mlp_classifier.in_channels == in_channels
    assert mlp_classifier.out_channels == out_channels
    assert mlp_classifier.hidden_channels == hidden_channels
    assert mlp_classifier.num_layers == num_layers
    assert mlp_classifier.lr == lr
    assert mlp_classifier.dropout == dropout
