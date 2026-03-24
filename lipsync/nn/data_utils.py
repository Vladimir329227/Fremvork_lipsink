"""Dataset utilities: map, minibatching, shuffle, split, normalise, and more.

Example::

    from lipsync.nn import Dataset

    ds = Dataset(X, y)
    ds = ds.map(lambda x, y: (x / 255.0, y))   # normalise
    ds = ds.shuffle(seed=42)
    train_ds, val_ds = ds.split(0.8)

    for xb, yb in train_ds.batch(32):
        ...
"""
from __future__ import annotations

import random
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    """Lightweight tensor dataset with functional-style transforms.

    Args:
        X: Feature tensor of shape (N, ...).
        y: Optional label tensor of shape (N, ...).
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:
        self.X = X
        self.y = y

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def map(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor | None],
                     tuple[torch.Tensor, torch.Tensor | None]],
    ) -> "Dataset":
        """Apply *fn* element-wise and return a new Dataset.

        Args:
            fn: Callable(x_i, y_i) → (x_i_new, y_i_new).
                y_i is None when the dataset has no labels.

        Example::

            ds = ds.map(lambda x, y: (x.float() / 255.0, y))
        """
        xs, ys = [], []
        for i in range(len(self)):
            xi = self.X[i]
            yi = self.y[i] if self.y is not None else None
            xi_new, yi_new = fn(xi, yi)
            xs.append(xi_new)
            ys.append(yi_new)
        new_X = torch.stack(xs)
        new_y = torch.stack(ys) if ys[0] is not None else None
        return Dataset(new_X, new_y)

    def map_x(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Dataset":
        """Apply *fn* to every feature vector (labels unchanged).

        Example::

            ds = ds.map_x(lambda x: (x - mean) / std)
        """
        return Dataset(torch.stack([fn(self.X[i]) for i in range(len(self))]), self.y)

    def map_y(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Dataset":
        """Apply *fn* to every label (features unchanged)."""
        if self.y is None:
            raise ValueError("Dataset has no labels")
        return Dataset(self.X, torch.stack([fn(self.y[i]) for i in range(len(self))]))

    def filter(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor | None], bool],
    ) -> "Dataset":
        """Keep only samples for which *fn(x_i, y_i)* returns True."""
        indices = [
            i for i in range(len(self))
            if fn(self.X[i], self.y[i] if self.y is not None else None)
        ]
        new_y = self.y[indices] if self.y is not None else None
        return Dataset(self.X[indices], new_y)

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def shuffle(self, seed: int | None = None) -> "Dataset":
        """Return a shuffled copy of the dataset."""
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        idx = torch.randperm(len(self), generator=gen)
        new_y = self.y[idx] if self.y is not None else None
        return Dataset(self.X[idx], new_y)

    def sort_by_label(self) -> "Dataset":
        """Return dataset sorted by label (useful for visualisation)."""
        if self.y is None:
            raise ValueError("Dataset has no labels")
        idx = self.y.argsort()
        return Dataset(self.X[idx], self.y[idx])

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(
        self,
        ratio: float = 0.8,
        shuffle_first: bool = True,
        seed: int = 42,
    ) -> tuple["Dataset", "Dataset"]:
        """Split into train and val datasets.

        Args:
            ratio: Fraction of data to use for training (0 < ratio < 1).
            shuffle_first: Shuffle before splitting.
            seed: Random seed for reproducibility.

        Returns:
            (train_dataset, val_dataset)
        """
        ds = self.shuffle(seed) if shuffle_first else self
        n = int(len(ds) * ratio)
        train_y = ds.y[:n] if ds.y is not None else None
        val_y = ds.y[n:] if ds.y is not None else None
        return Dataset(ds.X[:n], train_y), Dataset(ds.X[n:], val_y)

    def k_fold(self, k: int = 5, seed: int = 42) -> Iterator[tuple["Dataset", "Dataset"]]:
        """Yield (train, val) splits for k-fold cross-validation.

        Args:
            k: Number of folds.

        Yields:
            (train_dataset, val_dataset) for each fold.
        """
        ds = self.shuffle(seed)
        fold_size = len(ds) // k
        for i in range(k):
            val_start = i * fold_size
            val_end = val_start + fold_size
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, len(ds)))
            val_y = ds.y[val_idx] if ds.y is not None else None
            train_y = ds.y[train_idx] if ds.y is not None else None
            yield Dataset(ds.X[train_idx], train_y), Dataset(ds.X[val_idx], val_y)

    # ------------------------------------------------------------------
    # Batching and iteration
    # ------------------------------------------------------------------

    def batch(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield mini-batches as (X_batch,) or (X_batch, y_batch) tuples.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Shuffle before batching.
            drop_last: Drop incomplete last batch.

        Example::

            for xb, yb in ds.batch(32, shuffle=True):
                pred = model(xb)
        """
        ds = self.shuffle() if shuffle else self
        N = len(ds)
        for start in range(0, N, batch_size):
            end = start + batch_size
            if drop_last and end > N:
                break
            xb = ds.X[start:end]
            if ds.y is not None:
                yield xb, ds.y[start:end]
            else:
                yield (xb,)

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Wrap in a PyTorch DataLoader.

        Args:
            batch_size: Mini-batch size.
            shuffle: Shuffle samples each epoch.
            num_workers: Background workers for data loading.
        """
        if self.y is not None:
            td = TensorDataset(self.X, self.y)
        else:
            td = TensorDataset(self.X)
        return DataLoader(td, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def normalise(self, mean: torch.Tensor | None = None,
                  std: torch.Tensor | None = None) -> "Dataset":
        """Z-score normalise features; computes statistics from data if not given."""
        m = mean if mean is not None else self.X.mean(dim=0)
        s = std if std is not None else self.X.std(dim=0).clamp(min=1e-8)
        return Dataset((self.X - m) / s, self.y)

    def one_hot(self, num_classes: int) -> "Dataset":
        """One-hot encode integer labels."""
        if self.y is None:
            raise ValueError("Dataset has no labels")
        y_oh = torch.zeros(len(self), num_classes)
        y_oh.scatter_(1, self.y.long().unsqueeze(1), 1.0)
        return Dataset(self.X, y_oh)

    def to_device(self, device: str | torch.device) -> "Dataset":
        """Move all tensors to *device*."""
        new_y = self.y.to(device) if self.y is not None else None
        return Dataset(self.X.to(device), new_y)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)

    def __repr__(self) -> str:
        y_info = f", y={tuple(self.y.shape)}" if self.y is not None else ""
        return f"Dataset(X={tuple(self.X.shape)}{y_info})"

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(cls, X, y=None) -> "Dataset":
        """Create from numpy arrays."""
        import numpy as np
        Xt = torch.from_numpy(np.asarray(X)).float()
        yt = torch.from_numpy(np.asarray(y)) if y is not None else None
        return cls(Xt, yt)

    @classmethod
    def concat(cls, *datasets: "Dataset") -> "Dataset":
        """Concatenate multiple datasets along the sample axis."""
        Xs = torch.cat([d.X for d in datasets])
        if all(d.y is not None for d in datasets):
            ys = torch.cat([d.y for d in datasets])  # type: ignore[arg-type]
        else:
            ys = None
        return cls(Xs, ys)
