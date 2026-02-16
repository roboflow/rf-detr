# Training Loggers

RF-DETR supports integration with popular experiment tracking and visualization platforms. You can enable one or more loggers to monitor your training runs, compare experiments, and track metrics over time.

## TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful toolkit for visualizing and tracking training metrics.

### Setup

Install the required packages:

```bash
pip install "rfdetr[metrics]"
```

### Usage

Enable TensorBoard logging in your training:

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
    tensorboard=True,
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

TensorBoard tracks:

- Training and validation loss (total)
- Validation mAP
- EMA model metrics (when enabled)

---

## Weights and Biases

[Weights and Biases (W&B)](https://www.wandb.ai) is a cloud-based platform for experiment tracking and visualization.

### Setup

Install the required packages:

```bash
pip install "rfdetr[metrics]"
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

---

## ClearML

[ClearML](https://clear.ml) is an open-source platform for managing, tracking, and automating machine learning experiments.

### Setup

Install the required packages:

```bash
pip install "rfdetr[metrics]"
```

Initialize ClearML:

```bash
clearml-init
```

Follow the instructions to connect to your ClearML server (hosted or self-hosted).

### Usage

Enable ClearML logging in your training:

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
    clearml=True,
    project="my-detection-project",
    run="experiment-001",
)
```

### Configuration

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `project` | Groups related experiments together                 |
| `run`     | Identifies individual training sessions (task name) |

### Features

Access your experiments in the ClearML Web UI. ClearML provides:

- Real-time metric visualization
- Experiment comparison
- Hyperparameter tracking
- Artifact storage
- Model versioning

---

## MLflow

[MLflow](https://mlflow.org/) is an open-source platform for the machine learning lifecycle that helps track experiments, package code into reproducible runs, and share and deploy models.

### Setup

Install the required packages:

```bash
pip install "rfdetr[metrics]"
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

---

## Using Multiple Loggers

You can enable multiple logging systems simultaneously:

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    tensorboard=True,
    wandb=True,
    clearml=True,
    mlflow=True,
    project="my-project",
    run="experiment-001",
)
```

This allows you to leverage the strengths of different platforms:

- **TensorBoard**: Local visualization and debugging
- **W&B**: Cloud-based collaboration and experiment comparison
- **ClearML**: End-to-end MLOps pipeline automation
- **MLflow**: Model registry and deployment tracking
