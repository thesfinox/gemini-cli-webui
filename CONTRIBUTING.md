# Contributing to Gemini Web UI

Thank you for your interest in contributing to **Gemini Web UI**!
We welcome contributions from the community to help improve this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please report it by opening an issue on [GitHub](https://github.com/your-repo/gemini-webui/issues).
Include as much detail as possible:

- Steps to reproduce the issue.
- Expected behavior vs. actual behavior.
- Screenshots or logs if applicable.
- Your operating system and environment details.

### Suggesting Enhancements

We love hearing ideas for new features!
Please open an issue to discuss your suggestion before implementing it.
This ensures that your work aligns with the project’s goals and avoids duplication of effort.

### Pull Requests

We follow the standard “Fork & Pull” workflow.

1. **Fork the Repository**: Click the “Fork” button on the top right of the repository page to create your own copy of the project.
2. **Clone your Fork**:

    ```bash
    git clone https://github.com/YOUR_USERNAME/gemini-webui.git
    cd gemini-webui
    ```

3. **Create a Branch**: Create a new branch for your feature or bug fix.

    ```bash
    git checkout -b feature/my-new-feature
    ```

4. **Make Changes**: Implement your changes. Ensure your code follows the project’s coding style and conventions.
5. **Test Your Changes**: Run the application locally to verify that your changes work as expected.
6. **Commit Your Changes**: Write clear and concise commit messages.

    ```bash
    git commit -m "feat: add new feature"
    ```

7. **Push to Your Fork**:

    ```bash
    git push origin feature/my-new-feature
    ```

8. **Open a Pull Request**: Go to the original repository and open a Pull Request (PR) from your forked branch. Provide a clear description of your changes and reference any related issues.

## Development Setup

To set up the development environment:

1. Ensure you have Python 3.13+ installed.
2. Install dependencies using `uv` (recommended) or `pip`:

    ```bash
    uv sync
    ```

    or

    ```bash
    pip install .
    ```

## Code Style

- Please ensure your code is formatted and linted.
- Add comments where necessary to explain complex logic.
- Do not forget to update documentation if your changes affect it.

Thank you for contributing!
