# Train an RF-DETR Model

You can train RF-DETR object detection and segmentation models on a custom dataset using the `rfdetr` Python package, or in the cloud using Roboflow.

This guide describes how to train both an object detection and segmentation RF-DETR model.

### Dataset structure

RF-DETR expects the dataset to be in COCO format. Divide your dataset into three subdirectories: `train`, `valid`, and `test`. Each sub-directory should contain its own `_annotations.coco.json` file that holds the annotations for that particular split, along with the corresponding image files. Below is an example of the directory structure:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

[Roboflow](https://roboflow.com/annotate) allows you to create object detection datasets from scratch or convert existing datasets from formats like YOLO, and then export them in COCO JSON format for training. You can also explore [Roboflow Universe](https://universe.roboflow.com/) to find pre-labeled datasets for a range of use cases.

If you are training a segmentation model, your COCO JSON annotations should have a `segmentation` key with the polygon associated with each annotation.

## Start Training

You can fine-tune RF-DETR from pre-trained COCO checkpoints.

For object detection, the RF-DETR-B checkpoint is used by default. To get started quickly with training an object detection model, please refer to our fine-tuning Google Colab [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb).

For image segmentation, the RF-DETR-Seg (Preview) checkpoint is used by default.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>
    )
    ```

Different GPUs have different VRAM capacities, so adjust batch_size and grad_accum_steps to maintain a total batch size of 16. For example, on a powerful GPU like the A100, use `batch_size=16` and `grad_accum_steps=1`; on smaller GPUs like the T4, use `batch_size=4` and `grad_accum_steps=4`. This gradient accumulation strategy helps train effectively even with limited memory.

</details>

<details>
<summary>More parameters</summary>

<br>

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>dataset_dir</code></td>
      <td>Specifies the COCO-formatted dataset location with <code>train</code>, <code>valid</code>, and <code>test</code> folders, each containing <code>_annotations.coco.json</code>. Ensures the model can properly read and parse data.</td>
    </tr>
    <tr>
      <td><code>output_dir</code></td>
      <td>Directory where training artifacts (checkpoints, logs, etc.) are saved. Important for experiment tracking and resuming training.</td>
    </tr>
    <tr>
      <td><code>epochs</code></td>
      <td>Number of full passes over the dataset. Increasing this can improve performance but extends total training time.</td>
    </tr>
    <tr>
      <td><code>batch_size</code></td>
      <td>Number of samples processed per iteration. Higher values require more GPU memory but can speed up training. Must be balanced with <code>grad_accum_steps</code> to maintain the intended total batch size.</td>
    </tr>
    <tr>
      <td><code>grad_accum_steps</code></td>
      <td>Accumulates gradients over multiple mini-batches, effectively raising the total batch size without requiring as much memory at once. Helps train on smaller GPUs at the cost of slightly more time per update.</td>
    </tr>
    <tr>
      <td><code>lr</code></td>
      <td>Learning rate for most parts of the model. Influences how quickly or cautiously the model adjusts its parameters.</td>
    </tr>
    <tr>
      <td><code>lr_encoder</code></td>
      <td>Learning rate specifically for the encoder portion of the model. Useful for fine-tuning encoder layers at a different pace.</td>
    </tr>
    <tr>
      <td><code>resolution</code></td>
      <td>Sets the input image dimensions. Higher values can improve accuracy but require more memory and can slow training. Must be divisible by 56.</td>
    </tr>
    <tr>
      <td><code>weight_decay</code></td>
      <td>Coefficient for L2 regularization. Helps prevent overfitting by penalizing large weights, often improving generalization.</td>
    </tr>
    <tr>
      <td><code>device</code></td>
      <td>Specifies the hardware (e.g., <code>cpu</code> or <code>cuda</code>) to run training on. GPU significantly speeds up training.</td>
    </tr>
    <tr>
      <td><code>use_ema</code></td>
      <td>Enables Exponential Moving Average of weights, producing a smoothed checkpoint. Often improves final performance with slight overhead.</td>
    </tr>
    <tr>
      <td><code>gradient_checkpointing</code></td>
      <td>Re-computes parts of the forward pass during backpropagation to reduce memory usage. Lowers memory needs but increases training time.</td>
    </tr>
    <tr>
      <td><code>checkpoint_interval</code></td>
      <td>Frequency (in epochs) at which model checkpoints are saved. More frequent saves provide better coverage but consume more storage.</td>
    </tr>
    <tr>
      <td><code>resume</code></td>
      <td>Path to a saved checkpoint for continuing training. Restores both model weights and optimizer state.</td>
    </tr>
    <tr>
      <td><code>tensorboard</code></td>
      <td>Enables logging of training metrics to TensorBoard for monitoring progress and performance.</td>
    </tr>
    <tr>
      <td><code>wandb</code></td>
      <td>Activates logging to Weights &amp; Biases, facilitating cloud-based experiment tracking and visualization.</td>
    </tr>
    <tr>
      <td><code>project</code></td>
      <td>Project name for Weights &amp; Biases logging. Groups multiple runs under a single heading.</td>
    </tr>
    <tr>
      <td><code>run</code></td>
      <td>Run name for Weights &amp; Biases logging, helping differentiate individual training sessions within a project.</td>
    </tr>
    <tr>
      <td><code>early_stopping</code></td>
      <td>Enables an early stopping callback that monitors mAP improvements to decide if training should be stopped. Helps avoid needless epochs when mAP plateaus.</td>
    </tr>
    <tr>
      <td><code>early_stopping_patience</code></td>
      <td>Number of consecutive epochs without mAP improvement before stopping. Prevents wasting resources on minimal gains.</td>
    </tr>
    <tr>
      <td><code>early_stopping_min_delta</code></td>
      <td>Minimum change in mAP to qualify as an improvement. Ensures that trivial gains don’t reset the early stopping counter.</td>
    </tr>
    <tr>
      <td><code>early_stopping_use_ema</code></td>
      <td>Whether to track improvements using the EMA version of the model. Uses EMA metrics if available, otherwise falls back to regular mAP.</td>
    </tr>
  </tbody>
</table>

</details>

### Result checkpoints

During training, multiple model checkpoints are saved to the output directory:

- `checkpoint.pth` – the most recent checkpoint, saved at the end of the latest epoch.

- `checkpoint_<number>.pth` – periodic checkpoints saved every N epochs (default is every 10).

- `checkpoint_best_ema.pth` – best checkpoint based on validation score, using the EMA (Exponential Moving Average) weights. EMA weights are a smoothed version of the model’s parameters across training steps, often yielding better generalization.

- `checkpoint_best_regular.pth` – best checkpoint based on validation score, using the raw (non-EMA) model weights.

- `checkpoint_best_total.pth` – final checkpoint selected for inference and benchmarking. It contains only the model weights (no optimizer state or scheduler) and is chosen as the better of the EMA and non-EMA models based on validation performance.

??? note "Checkpoint file sizes"

    Checkpoint sizes vary based on what they contain:

    - **Training checkpoints** (e.g. `checkpoint.pth`, `checkpoint_<number>.pth`) include model weights, optimizer state, scheduler state, and training metadata. Use these to resume training.
    
    - **Evaluation checkpoints** (e.g. `checkpoint_best_ema.pth`, `checkpoint_best_regular.pth`) store only the model weights — either EMA or raw — and are used to track the best-performing models. These may come from different epochs depending on which version achieved the highest validation score.
    
    - **Stripped checkpoint** (e.g. `checkpoint_best_total.pth`) contains only the final model weights and is optimized for inference and deployment.

### Resume training

You can resume training from a previously saved checkpoint by passing the path to the `checkpoint.pth` file using the `resume` argument. This is useful when training is interrupted or you want to continue fine-tuning an already partially trained model. The training loop will automatically load the weights and optimizer state from the provided checkpoint file.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        resume=<CHECKPOINT_PATH>
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        resume=<CHECKPOINT_PATH>
    )
    ```


### Early stopping

Early stopping monitors validation mAP and halts training if improvements remain below a threshold for a set number of epochs. This can reduce wasted computation once the model converges. Additional parameters—such as `early_stopping_patience`, `early_stopping_min_delta`, and `early_stopping_use_ema`—let you fine-tune the stopping behavior.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        early_stopping=True
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        early_stopping=True
    )
    ```


### Multi-GPU training

You can fine-tune RF-DETR on multiple GPUs using PyTorch’s Distributed Data Parallel (DDP). Create a `main.py` script that initializes your model and calls `.train()` as usual than run it in terminal.

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py
```

Replace `8` in the `--nproc_per_node argument` with the number of GPUs you want to use. This approach creates one training process per GPU and splits the workload automatically. Note that your effective batch size is multiplied by the number of GPUs, so you may need to adjust your `batch_size` and `grad_accum_steps` to maintain the same overall batch size.

### Logging with TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful toolkit that helps you visualize and track training metrics. With TensorBoard set up, you can train your model and keep an eye on the logs to monitor performance, compare experiments, and optimize model training. To enable logging, simply pass `tensorboard=True` when training the model.

<details>
<summary>Using TensorBoard with RF-DETR</summary>

<br>

- TensorBoard logging requires additional packages. Install them with:

    ```bash
    pip install "rfdetr[metrics]"
    ```
  
- To activate logging, pass the extra parameter `tensorboard=True` to `.train()`:

    ```python
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    
    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        tensorboard=True
    )
    ```

- To use TensorBoard locally, navigate to your project directory and run:

    ```bash
    tensorboard --logdir <OUTPUT_DIR>
    ```

    Then open `http://localhost:6006/` in your browser to view your logs.

- To use TensorBoard in Google Colab run:

    ```bash
    %load_ext tensorboard
    %tensorboard --logdir <OUTPUT_DIR>
    ```
      
</details>

### Logging with Weights and Biases

[Weights and Biases (W&B)](https://www.wandb.ai) is a powerful cloud-based platform that helps you visualize and track training metrics. With W&B set up, you can monitor performance, compare experiments, and optimize model training using its rich feature set. To enable logging, simply pass `wandb=True` when training the model.

<details>
<summary>Using Weights and Biases with RF-DETR</summary>

<br>

- Weights and Biases logging requires additional packages. Install them with:

    ```bash
    pip install "rfdetr[metrics]"
    ```

- Before using W&B, make sure you are logged in:

    ```bash
    wandb login
    ```

    You can retrieve your API key at wandb.ai/authorize.

- To activate logging, pass the extra parameter `wandb=True` to `.train()`:

    ```python
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    
    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        wandb=True,
        project=<PROJECT_NAME>,
        run=<RUN_NAME>
    )
    ```

    In W&B, projects are collections of related machine learning experiments, and runs are individual sessions where training or evaluation happens. If you don't specify a name for a run, W&B will assign a random one automatically.
  
</details>

### Load and run fine-tuned model

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

    detections = model.predict(<IMAGE_PATH>)
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)

    detections = model.predict(<IMAGE_PATH>)
    ```

## ONNX export

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency.

To export your model, first install the `onnxexport` extension:

```
pip install rfdetr[onnxexport]
```

Then, run:

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

    model.export()
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)

    model.export()
    ```

This command saves the ONNX model to the `output` directory.