# Security Policy

## Project Status

RF-DETR is an **research project** under active development. While we strive for stability, users should be aware that the codebase may contain undiscovered vulnerabilities typical of research-grade software.

## Supported Versions

Security updates are generally provided for the latest stable release. Given the nature of this project:

- Critical vulnerabilities affecting the latest release will be addressed promptly
- Fixes for older versions are evaluated on a case-by-case basis depending on severity
- Users are strongly encouraged to use the latest version

| Version        | Support Status     |
| -------------- | ------------------ |
| Latest release | :white_check_mark: |
| Older versions | Case-by-case       |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please help us by reporting it responsibly.

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report to: **security@roboflow.com**

### What to Include

- Clear description of the vulnerability
- Steps to reproduce the issue
- Affected versions (if known)
- Potential impact assessment
- Any proof-of-concept code (if applicable)

### What to Expect

- Acknowledgment within 72 hours
- Regular updates on the status of your report
- Credit in the security advisory (unless you prefer to remain anonymous)

## Security Considerations for ML Projects

### Model Weights and Checkpoints

**Critical**: Model checkpoint files (`.pt`, `.pth`, `.onnx`) can contain arbitrary Python code that executes during loading.

- **Only load models from trusted sources**
- Verify checksums when downloading pre-trained models
- Be especially cautious with community-contributed checkpoints
- Consider running untrusted models in sandboxed environments

### Dependency Security

RF-DETR depends on the PyTorch ecosystem and other ML libraries:

- Keep PyTorch, torchvision, and transformers updated
- Monitor security advisories for dependencies
- Use virtual environments to isolate installations
- Regularly update dependencies: `pip install --upgrade rfdetr`

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
