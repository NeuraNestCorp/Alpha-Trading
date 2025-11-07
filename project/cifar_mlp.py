"""Train simple MLP classifiers on CIFAR-10 images flattened to 3072-length vectors."""
from __future__ import annotations

import pickle
import tarfile
import urllib.request
from urllib.error import URLError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


NUM_PIXELS = 32 * 32 * 3  # 3072
NUM_CLASSES = 10
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


@dataclass
class TrainingConfig:
    """Configuration for the CIFAR-10 MLP trainer."""

    hidden_layers: Sequence[int]
    learning_rate: float = 0.05
    batch_size: int = 256
    epochs: int = 10
    weight_scale: float = 0.05
    data_dir: Path = field(default_factory=lambda: Path("data/cifar10"))

    def describe_architecture(self) -> str:
        if not self.hidden_layers:
            return "No hidden layer"
        layer_desc = "-".join(str(units) for units in self.hidden_layers)
        return f"Hidden layers: {layer_desc}"


def ensure_cifar10(data_dir: Path) -> Path:
    """Download and extract CIFAR-10 if needed, returning the directory with batches."""
    batches_dir = data_dir / "cifar-10-batches-py"
    if batches_dir.exists():
        return batches_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "cifar-10-python.tar.gz"

    if archive_path.exists():
        print("Found existing CIFAR-10 archive. Skipping download.")
    else:
        print("Downloading CIFAR-10 dataset...")
        try:
            urllib.request.urlretrieve(CIFAR_URL, archive_path)
        except URLError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "Unable to download CIFAR-10 archive. "
                "Please download it manually from https://www.cs.toronto.edu/~kriz/cifar.html "
                f"and place it at {archive_path}."
            ) from exc
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    return batches_dir


def load_cifar10(data_dir: Path) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 batches into NumPy arrays."""
    batches_dir = ensure_cifar10(data_dir)

    def load_batch(batch_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with batch_path.open("rb") as fp:
            batch = pickle.load(fp, encoding="latin1")
        return batch["data"], np.array(batch["labels"], dtype=np.int64)

    x_train_list: List[np.ndarray] = []
    y_train_list: List[np.ndarray] = []
    for i in range(1, 6):
        data, labels = load_batch(batches_dir / f"data_batch_{i}")
        x_train_list.append(data)
        y_train_list.append(labels)

    x_train = np.concatenate(x_train_list, axis=0).astype(np.float32) / 255.0
    y_train = np.concatenate(y_train_list, axis=0)

    x_test, y_test = load_batch(batches_dir / "test_batch")
    x_test = x_test.astype(np.float32) / 255.0

    y_train_one_hot = np.eye(NUM_CLASSES)[y_train]
    y_test_one_hot = np.eye(NUM_CLASSES)[y_test]

    return (x_train, y_train_one_hot), (x_test, y_test_one_hot)


class DenseLayer:
    """A single dense (fully-connected) layer with optional ReLU activation."""

    def __init__(self, input_dim: int, output_dim: int, activation: str | None, weight_scale: float = 0.05):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        rng = np.random.default_rng()
        self.weights = rng.normal(0.0, weight_scale, size=(input_dim, output_dim)).astype(np.float32)
        self.bias = np.zeros(output_dim, dtype=np.float32)
        self.input_cache: np.ndarray | None = None
        self.linear_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        z = x @ self.weights + self.bias
        self.linear_output = z
        if self.activation == "relu":
            return np.maximum(0.0, z)
        return z

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.activation == "relu" and self.linear_output is not None:
            grad_output = grad_output * (self.linear_output > 0)

        assert self.input_cache is not None
        batch_size = self.input_cache.shape[0]
        grad_weights = self.input_cache.T @ grad_output / batch_size
        grad_bias = grad_output.mean(axis=0)
        grad_input = grad_output @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input


class SequentialModel:
    """Minimal sequential container managing dense layers."""

    def __init__(self, layers: Sequence[DenseLayer]) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray, learning_rate: float) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute softmax probabilities, cross-entropy loss, and gradient w.r.t logits."""
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(logits_shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    batch_size = logits.shape[0]
    loss = -np.sum(targets * np.log(probs + 1e-12)) / batch_size
    grad_logits = (probs - targets) / batch_size
    return loss, grad_logits, probs


@dataclass
class EpochHistory:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class CIFARMLPTrainer:
    """Train sequential MLPs on the flattened CIFAR-10 dataset using SGD."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model: SequentialModel | None = None
        self.history: List[EpochHistory] = []

    def load_data(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        return load_cifar10(self.config.data_dir)

    def build_model(self) -> SequentialModel:
        layers: List[DenseLayer] = []
        input_dim = NUM_PIXELS
        for units in self.config.hidden_layers:
            layers.append(DenseLayer(input_dim, units, activation="relu", weight_scale=self.config.weight_scale))
            input_dim = units
        layers.append(DenseLayer(input_dim, NUM_CLASSES, activation=None, weight_scale=self.config.weight_scale))
        self.model = SequentialModel(layers)
        return self.model

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        assert self.model is not None
        logits = self.model.forward(x)
        loss, _, probs = softmax_cross_entropy(logits, y)
        predictions = np.argmax(probs, axis=1)
        targets = np.argmax(y, axis=1)
        accuracy = (predictions == targets).mean()
        return loss, accuracy

    def train(self, train_data: tuple[np.ndarray, np.ndarray], val_data: tuple[np.ndarray, np.ndarray]) -> List[EpochHistory]:
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        x_train, y_train = train_data
        x_val, y_val = val_data
        num_samples = x_train.shape[0]

        for epoch in range(1, self.config.epochs + 1):
            permutation = np.random.permutation(num_samples)
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for start in range(0, num_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_x = x_train_shuffled[start:end]
                batch_y = y_train_shuffled[start:end]
                logits = self.model.forward(batch_x)
                loss, grad_logits, _ = softmax_cross_entropy(logits, batch_y)
                self.model.backward(grad_logits, self.config.learning_rate)

            train_loss, train_acc = self.evaluate(x_train, y_train)
            val_loss, val_acc = self.evaluate(x_val, y_val)

            epoch_history = EpochHistory(epoch, train_loss, train_acc, val_loss, val_acc)
            self.history.append(epoch_history)
            print(
                f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
        return self.history

    @staticmethod
    def plot_history(histories: Iterable[tuple[str, List[EpochHistory]]], output_path: Path) -> None:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for label, history in histories:
            epochs = [h.epoch for h in history]
            plt.plot(epochs, [h.train_loss for h in history], label=f"{label} Train")
            plt.plot(epochs, [h.val_loss for h in history], linestyle="--", label=f"{label} Val")
        plt.title("Loss vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        for label, history in histories:
            epochs = [h.epoch for h in history]
            plt.plot(epochs, [h.train_accuracy for h in history], label=f"{label} Train")
            plt.plot(epochs, [h.val_accuracy for h in history], linestyle="--", label=f"{label} Val")
        plt.title("Accuracy vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()


def run_experiments(output_dir: Path | None = None) -> list[tuple[str, CIFARMLPTrainer]]:
    output_dir = output_dir or Path("outputs")

    base_config = TrainingConfig(hidden_layers=[], epochs=10, learning_rate=0.1, batch_size=512)
    hidden_config = TrainingConfig(hidden_layers=[512], epochs=10, learning_rate=0.1, batch_size=512)

    trainers: List[tuple[str, CIFARMLPTrainer]] = [
        ("No Hidden", CIFARMLPTrainer(base_config)),
        ("Hidden 512", CIFARMLPTrainer(hidden_config)),
    ]

    (x_train, y_train), (x_test, y_test) = trainers[0][1].load_data()

    histories: list[tuple[str, List[EpochHistory]]] = []
    for label, trainer in trainers:
        trainer.build_model()
        history = trainer.train((x_train, y_train), (x_test, y_test))
        histories.append((label, history))

    plot_path = output_dir / "cifar_mlp_histories.png"
    CIFARMLPTrainer.plot_history(histories, plot_path)
    return trainers


if __name__ == "__main__":  # pragma: no cover - script entry point
    run_experiments()
