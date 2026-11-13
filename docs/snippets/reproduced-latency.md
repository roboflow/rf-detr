!!! info "How the latency numbers were obtained"

    - **Latency**: the end-to-end `RFDETRInference.predict` call (preprocessing, forward pass and postprocessing) at batch 1, timed once for each of 500 distinct COCO val2017 images after 20 warm-up calls, with the [latency cookbook](../../../cookbooks/measure-latency/). The median is the primary figure and p95 is the 95th percentile of the same 500 calls. For each size the table shows the configuration with the lowest median among those the cookbook tries on Apple Silicon: PyTorch with inductor in FP32 and FP16, PyTorch with TorchScript in FP16, and CoreML in FP32 and FP16.
    - **Run**: Apple M4 Max (16-core CPU, 40-core GPU), macOS 26.7.1, rfdetr 1.13.0.dev0, rfdetr_plus 1.2.0, torch 2.14.0 and coremltools 9.0, measured on 2026-11-13 on an otherwise idle machine.
    - **Ties**: repeated runs of the largest models differed by 5 to 7%, so treat a gap under about 5% to the next configuration as a tie and pick by accuracy and tooling.

!!! warning "FP16 can lower accuracy"

    Most of the fastest configurations above run in FP16. The FP16 AP in the [table above](#pre-trained-checkpoints) was scored on TensorRT FP16 engines, where it stays within 0.1 AP of FP32 for every size here. It was not scored for PyTorch or CoreML FP16 on Apple Silicon, so check the accuracy on your own data before you rely on an FP16 configuration.
