# Installation

Welcome to RF-DETR! This guide will help you install and set up RF-DETR for your projects. Whether you're a developer looking to contribute or an end-user ready to start using RF-DETR, we've got you covered.

## Installation Methods

RF-DETR supports several installation methods. Choose the option which best fits your workflow.

!!! example "Installation"

    === "pip (recommended)"
        The easiest way to install RF-DETR is using `pip`. This method is recommended for most users.

        ```bash
        pip install rfdetr
        ```

    === "uv"
        If you are using `uv`, you can install RF-DETR using the following command:

        ```bash
        uv pip install rfdetr
        ```

        For `uv` projects, you can also use:

        ```bash
        uv add rfdetr
        ```

    === "Source Archive"
        To install the latest development version of RF-DETR from source without cloning the full repository, run the command below.

        ```bash
        pip install https://github.com/roboflow/rf-detr/archive/refs/heads/develop.zip
        ```

## Dev Environment

If you plan to contribute to RF-DETR or modify the codebase locally, set up a local development environment using the steps below.

!!! example "Development Setup"

    === "virtualenv"
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

    === "uv"
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
- If you encounter any issues during installation, refer to the [troubleshooting](#troubleshooting) section or open an issue on the [GitHub repository](https://github.com/roboflow/rf-detr).

## Troubleshooting

If you encounter any issues during installation, here are some common solutions:

- **Permission Issues**: Use `pip install --user rfdetr` to install the package for your user only.
- **Dependency Conflicts**: Use a virtual environment to isolate the installation.
- **Python Version**: Ensure you are using Python 3.10 or higher.
