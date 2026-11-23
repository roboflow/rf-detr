---
description: Deploy fine-tuned RF-DETR detection and segmentation models to Roboflow for cloud inference, edge hardware, and multi-step vision workflows.
---

# Deploy a Trained RF-DETR Model

!!! tip "Key Takeaways"

    - Deploy fine-tuned RF-DETR models to Roboflow with a single `deploy_to_roboflow()` call
    - Supports both detection and segmentation model deployment
    - Roboflow serves deployed models in the cloud, on edge hardware, or on NVIDIA Jetson
    - Run the same fine-tuned checkpoint locally with `RFDETRInference`

You can deploy a fine-tuned RF-DETR model to Roboflow.

Deploying to Roboflow allows you to create multi-step computer vision applications that run both in the cloud and your own hardware.

To deploy your model to Roboflow, run:

=== "Object Detection"

    ```python
    from rfdetr import RFDETRSmall

    x = RFDETRSmall(pretrain_weights="<path/to/pretrain/weights/dir>")
    x.deploy_to_roboflow(
        workspace="<your-workspace>",
        project_id="<your-project-id>",
        api_key="<YOUR_API_KEY>",
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegSmall

    x = RFDETRSegSmall(pretrain_weights="<path/to/pretrain/weights/dir>")
    x.deploy_to_roboflow(
        workspace="<your-workspace>",
        project_id="<your-project-id>",
        api_key="<YOUR_API_KEY>",
    )
    ```

Above, set your Roboflow Workspace ID, the ID of the project to which you want to upload your model, and your Roboflow API key. By default the model is deployed to the project's latest dataset version; pass `version=<number>` to target a specific one. A project with no generated versions falls back to version `1`, for which the Roboflow SDK raises its usual error.

- [Learn how to find your Workspace and Project ID.](https://docs.roboflow.com/developer/authentication/workspace-and-project-ids)
- [Learn how to find your API key.](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key)

Roboflow then serves your model for multi-step computer vision applications in the cloud and on your own hardware, including NVIDIA Jetson.

To run inference on the deployed model in the cloud, follow the [Serverless Cloud API guide](https://docs.roboflow.com/deployment/roboflow-cloud/serverless-api), using your project's model ID and version.

To run the same fine-tuned checkpoint locally, load it with `RFDETRInference`:

```python
from rfdetr import RFDETRInference

predictor = RFDETRInference("<path/to/checkpoint.pth>")
detections = predictor.predict("https://media.roboflow.com/dog.jpeg", threshold=0.5)
```

See [Run Inference](inference.md) for compiled predictors, exported models, and runtime options.
