# Installation

Welcome to RF-DETR! In this section, we'll guide you through installing and setting up RF-DETR for your projects. Whether you're a developer looking to contribute to the project or an end-user ready to start using RF-DETR in your applications, we've got you covered with detailed installation instructions.

## Let's Get Started

RF-DETR supports several installation methods to suit your workflow. Choose the one that best fits your needs:

### Installing with pip

The easiest way to install RF-DETR is using pip. This method is recommended for most users.

```bash
pip install rfdetr
```

### Installing with uv

If you are using uv, you can install RF-DETR using the following command:

```bash
uv pip install rfdetr
```

For uv projects, you can also use:

```bash
uv add rfdetr
```

### Installing from Source Archive

If you want to install the latest development version of RF-DETR from source without cloning the entire repository, you can install directly from the GitHub archive URL.

```bash
pip install https://github.com/roboflow/rf-detr/archive/refs/heads/develop.zip
```

### Setting Up a Local Development Environment

To set up a local development environment for modifying or contributing to RF-DETR, follow these steps:

#### Using Virtualenv

```bash
# Clone the repository and navigate to the root directory
git clone --depth 1 -b develop https://github.com/roboflow/rf-detr.git
cd rf-detr

# Set up a Python virtual environment with a specific Python version (e.g., 3.10)
python3.10 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package in development mode
pip install -e "."
```

#### Using uv

```bash
# Clone the repository and navigate to the root directory
git clone --depth 1 -b develop https://github.com/roboflow/rf-detr.git
cd rf-detr

# Pin Python version (optional but recommended)
uv python pin 3.11

# Sync environment (creates .venv, installs pinned Python, and installs dependencies)
uv sync

# Install the package in development mode with all extras
uv pip install -e . --all-extras
```

## Additional Notes

- Ensure you have Python 3.10 or higher installed.
- For development, it is recommended to use a virtual environment to avoid conflicts with other packages.
- If you encounter any issues during installation, refer to the [Troubleshooting](#troubleshooting) section or open an issue on the [GitHub repository](https://github.com/roboflow/rf-detr).

## Troubleshooting

If you encounter any issues during installation, here are some common solutions:

- **Permission Issues**: Use `pip install --user rfdetr` to install the package for your user only.
- **Dependency Conflicts**: Use a virtual environment to isolate the installation.
- **Python Version**: Ensure you are using Python 3.10 or higher.

If the issue persists, please open an issue on the [GitHub repository](https://github.com/roboflow/rf-detr) with details about your environment and the error message.