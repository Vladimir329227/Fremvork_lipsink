import torch
import torch.nn as nn

from lipsync.activations import build_activation
from lipsync.losses import build_loss
from lipsync.optimizers import build_optimizer


def test_activation_factory():
    x = torch.randn(2, 4)
    act = build_activation("relu")
    y = act(x)
    assert y.shape == x.shape


def test_loss_factory_classification():
    loss = build_loss("cross_entropy")
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    v = loss(logits, targets)
    assert v.item() >= 0


def test_loss_factory_regression():
    loss = build_loss("huber", delta=1.0)
    p = torch.randn(8, 2)
    t = torch.randn(8, 2)
    v = loss(p, t)
    assert v.item() >= 0


def test_optimizer_factory():
    m = nn.Linear(4, 2)
    opt = build_optimizer(m.parameters(), "momentum_sgd", lr=1e-3, momentum=0.9)
    x = torch.randn(4, 4)
    y = m(x)
    loss = y.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
