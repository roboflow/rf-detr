# Contributing to RF-DETR

Thank you for helping to advance RF-DETR! Your participation is invaluable in evolving our platform—whether you’re squashing bugs, refining documentation, or rolling out new features. Every contribution pushes the project forward.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [CLA Signing](#cla-signing)
3. [Google-Style Docstrings and Mandatory Type Hints](#google-style-docstrings-and-mandatory-type-hints)
4. [Reporting Bugs](#reporting-bugs)
5. [Adding a New Model](#adding-a-new-model)
6. [License](#license)

## How to Contribute

Your contributions can be in many forms—whether it’s enhancing existing features, improving documentation, resolving bugs, or proposing new ideas. Here’s a high-level overview to get you started:

1. [Fork the Repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo): Click the “Fork” button on our GitHub page to create your own copy.
2. [Clone Locally](https://docs.github.com/en/enterprise-server@3.11/repositories/creating-and-managing-repositories/cloning-a-repository): Download your fork to your local development environment.
3. [Create a Branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop): Use a descriptive name to create a new branch (e.g., `feature/your-descriptive-name`):
   ```bash
   git checkout -b feature/your-descriptive-name
   ```
4. Develop Your Changes: Make your updates, ensuring your commit messages clearly describe your modifications.
5. [Commit and Push](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop): Run:
   ```bash
   git add .
   git commit -m "A brief description of your changes"
   git push -u origin your-descriptive-name
   ```
6. [Open a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request): Submit your pull request against the main development branch. Please detail your changes and link any related issues.

Before merging, check that all tests pass and that your changes adhere to our development and documentation standards.

## CLA Signing

In order to maintain the integrity of our project, every pull request must include a signed Contributor License Agreement (CLA). This confirms that your contributions are properly licensed under our Apache 2.0 License. After opening your pull request, simply add a comment stating:

```
I have read the CLA Document and I sign the CLA.
```

This step is essential before any merge can occur.

## Google-Style Docstrings and Mandatory Type Hints

For clarity and maintainability, any new functions or classes must include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html) and use Python type hints. Type hints are mandatory in all function definitions, ensuring explicit parameter and return type declarations. These docstrings should clearly explain parameters, return types, and provide usage examples when applicable.

For example:

```python
def sample_function(param1: int, param2: int = 10) -> bool:
    """
    Provides a brief description of function behavior.

    Args:
        param1 (int): Explanation of the first parameter.
        param2 (int): Explanation of the second parameter, defaulting to 10.

    Returns:
        bool: True if the operation succeeds, otherwise False.

    Examples:
        >>> sample_function(5, 10)
        True
    """
    return param1 == param2
```

Following this pattern helps ensure consistency throughout the codebase.

## Reporting Bugs

Bug reports are vital for continued improvement. When reporting an issue, please include a clear, minimal reproducible example that demonstrates the problem. Detailed bug reports assist us in swiftly diagnosing and addressing issues.

## License

By contributing to RF-DETR, you agree that your contributions will be licensed under the Apache 2.0 License as specified in our [LICENSE](/LICENSE) file.

Thank you for your commitment to making RF-DETR better. We look forward to your pull requests and continued collaboration. Happy coding!
