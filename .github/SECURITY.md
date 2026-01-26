# Security Policy

## Project Status

RF-DETR is a **research project** under active development. While we strive for stability, the codebase may contain undiscovered vulnerabilities typical of research-grade software.

## Supported Versions

Security fixes are generally provided for the latest stable release.

Fixes for older versions may be provided at the maintainers' discretion, depending on severity and feasibility.

| Version        | Support Status     |
| -------------- | ------------------ |
| Latest release | :white_check_mark: |
| Older versions | Case-by-case       |

## Reporting a Vulnerability

Please report security issues privately.

**Do not** create a public GitHub issue for security vulnerabilities.

Report to: **security@roboflow.com**

Include (if available):
- A clear description and impact
- Steps to reproduce / proof-of-concept
- Affected versions, environment details, and relevant logs

We will review reports and respond as soon as reasonably possible.

## Security Considerations for ML Projects

### Model Weights and Checkpoints

**Critical**: PyTorch checkpoint files (`.pt`, `.pth`) can execute arbitrary code when loaded because they are commonly pickle-based.

- **Only load models from trusted sources**
- Prefer safer formats (e.g. `safetensors`) when available
- When possible, use safer loading options (e.g. `torch.load(..., weights_only=True)` where supported)

**Note**: ONNX models (`.onnx`) are not pickle-based, but parsing/optimizer toolchains can still have security vulnerabilities. Treat untrusted files cautiously.

### Dependency Security

RF-DETR depends on the PyTorch ecosystem and other ML libraries:

- Keep PyTorch, torchvision, and transformers updated
- Monitor security advisories for dependencies
- Use virtual environments to isolate installations
- Regularly update dependencies (for users): `pip install --upgrade rfdetr`

### Data Processing

- Validate and sanitize input data
- Be cautious when processing data from untrusted sources
- Consider resource limits when processing large batches

### Training and Inference

- Untrusted training data may contain adversarial examples
- Monitor resource usage during training to detect anomalies
- Consider using resource limits in production environments

## Known Limitations

- This is research software not hardened for production use
- The package has not undergone formal security auditing
- Custom CUDA kernels may have memory safety issues
- Limited input validation in some code paths

## Best Practices

1. **Run in isolated environments**: Use containers or virtual machines for production deployments
2. **Limit resource access**: Apply appropriate resource constraints (memory, GPU, CPU)
3. **Monitor for anomalies**: Track unusual behavior during training or inference
4. **Keep updated**: Regularly update to the latest version
5. **Review dependencies**: Understand the security posture of all dependencies

## Security Updates

Security patches will be announced via:
- GitHub Security Advisories
- Release notes
- Project README

Subscribe to repository notifications to stay informed.
