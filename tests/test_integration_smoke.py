import torch

from lipsync.nn import Linear, Sequential


def test_train_eval_predict_smoke():
    X = torch.randn(128, 4)
    y = torch.randint(0, 3, (128,))

    model = Sequential([
        Linear(4, 16, activation="relu"),
        Linear(16, 3),
    ])
    model.compile(optimizer="adamw", loss="cross_entropy", lr=1e-3)
    hist = model.fit(X, y, epochs=2, batch_size=16, verbose=False)

    assert len(hist) == 2
    m = model.evaluate(X, y)
    assert "loss" in m
    preds = model.predict_classes(X[:10])
    assert preds.shape[0] == 10
