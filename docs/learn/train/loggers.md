# Training Loggers

RF-DETR supports integration with popular experiment tracking and visualization platforms. You can enable one or more loggers to monitor your training runs, compare experiments, and track metrics over time.

## CSV (always active)

A `CSVLogger` is always active regardless of any flags. It requires no extra packages and writes all metrics to `{output_dir}/metrics.csv` on every validation step.

---

## TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful toolkit for visualizing and tracking training metrics.

TensorBoard logging is enabled by default. Pass `tensorboard=False` to disable it.

!!! note "Missing package behaviour"

    If the `tensorboard` package is not installed, training continues without error — a
    `UserWarning` is emitted and TensorBoard logging is silently suppressed. Install
    `rfdetr[loggers]` to avoid this.

### Setup

Install the required packages:

```bash
pip install "rfdetr[loggers]"
```

### Usage

TensorBoard is active unless you explicitly disable it:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output",
    # tensorboard=True is the default; pass tensorboard=False to disable
)
```

### Viewing Logs

**Local environment:**

```bash
tensorboard --logdir output
```

Then open `http://localhost:6006/` in your browser.

**Google Colab:**

```ipython
%load_ext tensorboard
%tensorboard --logdir output
```

### Logged Metrics

All logged metric keys are listed in the [Logged Metrics Reference](customization.md#logged-metrics-reference).

---

## Weights and Biases

[Weights and Biases (W&B)](https://www.wandb.ai) is a cloud-based platform for experiment tracking and visualization.

### Setup

Install the required packages:

```bash
pip install "rfdetr[loggers]"
```

Log in to W&B:

```bash
wandb login
```

You can retrieve your API key at [wandb.ai/authorize](https://wandb.ai/authorize).

### Usage

Enable W&B logging in your training:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output",
    wandb=True,
    project="my-detection-project",
    run="experiment-001",
)
```

### Configuration

| Parameter | Description                             |
| --------- | --------------------------------------- |
| `project` | Groups related experiments together     |
| `run`     | Identifies individual training sessions |

If you don't specify a run name, W&B assigns a random one automatically.

### Features

Access your runs at [wandb.ai](https://wandb.ai). W&B provides:

- Real-time metric visualization
- Experiment comparison
- Hyperparameter tracking
- System metrics (GPU usage, memory)
- Training config logging

### Logged Metrics

All logged metric keys are listed in the [Logged Metrics Reference](customization.md#logged-metrics-reference).

---

## ClearML

[ClearML](https://clear.ml) is an open-source platform for managing, tracking, and automating machine learning experiments.

**ClearML is not yet integrated as a native PTL logger.** Passing `clearml=True` to `model.train()` emits a `UserWarning` and has no other effect — metrics are not logged to ClearML.

### Workaround: ClearML SDK auto-binding

ClearML's SDK captures PyTorch Lightning metrics automatically when a `Task` is initialised before training begins:

```python
from clearml import Task
from rfdetr import RFDETRMedium

# Initialise before model.train() — ClearML auto-binds to PTL logging
task = Task.init(project_name="my-detection-project", task_name="experiment-001")

model = RFDETRMedium()
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output",
    # Do NOT pass clearml=True — it does nothing
)
```

Alternatively, attach a ClearML callback directly using the [Custom Training API](#attaching-loggers-via-the-custom-training-api).

---

## MLflow

[MLflow](https://mlflow.org/) is an open-source platform for the machine learning lifecycle that helps track experiments, package code into reproducible runs, and share and deploy models.

### Setup

Install the required packages:

```bash
pip install "rfdetr[loggers]"
```

### Usage

Enable MLflow logging in your training:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output",
    mlflow=True,
    project="my-detection-project",
    run="experiment-001",
)
```

### Configuration

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `project` | Sets the experiment name in MLflow                  |
| `run`     | Sets the run name (auto-generated if not specified) |

### Custom Tracking Server

To use a custom MLflow tracking server, set environment variables:

```python
import os

# Set MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "https://your-mlflow-server.com"

# For authentication with tracking servers that require it
os.environ["MLFLOW_TRACKING_TOKEN"] = "your-auth-token"

# Then initialize and train your model
model = RFDETRMedium()
model.train(..., mlflow=True)
```

For teams using a hosted MLflow service (like Databricks), you'll typically need to set:

- `MLFLOW_TRACKING_URI`: The URL of your MLflow tracking server
- `MLFLOW_TRACKING_TOKEN`: Authentication token for your MLflow server

### Viewing Logs

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri <OUTPUT_PATH>
```

Then open `http://localhost:5000` in your browser to access the MLflow dashboard.

### Logged Metrics

All logged metric keys are listed in the [Logged Metrics Reference](customization.md#logged-metrics-reference).

---

## Using Multiple Loggers

You can enable multiple logging systems simultaneously:

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    tensorboard=True,
    wandb=True,
    mlflow=True,
    project="my-project",
    run="experiment-001",
)
```

This allows you to leverage the strengths of different platforms:

- **TensorBoard**: Local visualization and debugging
- **W&B**: Cloud-based collaboration and experiment comparison
- **MLflow**: Model registry and deployment tracking

Note: `clearml=True` is accepted but has no effect in the current version — the flag does not attach a ClearML logger. Use the [ClearML SDK workaround](#clearml) instead.

---

## Attaching loggers via the Custom Training API

`build_trainer` automatically creates loggers from `TrainConfig` flags. To attach a logger not listed above (for example Neptune, Comet, or a fully custom logger), build it separately and append it to `trainer.loggers` before calling `trainer.fit`:

```python
from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

model_config = RFDETRMediumConfig(num_classes=10)
train_config = TrainConfig(
    dataset_dir="path/to/dataset",
    epochs=100,
    output_dir="output",
    tensorboard=True,  # built-in loggers still work
)

module = RFDETRModelModule(model_config, train_config)
datamodule = RFDETRDataModule(model_config, train_config)
trainer = build_trainer(train_config, model_config)

# Attach any additional PTL-compatible logger
from pytorch_lightning.loggers import CSVLogger  # example — use any PTL logger

trainer.loggers.append(CSVLogger(save_dir="output", name="extra"))

trainer.fit(module, datamodule)
```

CSVLogger is always active (it requires no extra packages). All logged metric keys — `train/loss`, `val/mAP_50_95`, `val/F1`, `val/ema_mAP_50_95`, `val/AP/<class>`, etc. — are written to every logger in the list.

→ **[Full list of logged metrics](customization.md#logged-metrics-reference)**
