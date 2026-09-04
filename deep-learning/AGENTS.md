# AGENTS Guidelines for This Repository

This repository contains a collection of Deep Learning assignments based on PyTorch. When working on these assignments interactively with an AI agent, please strictly follow the guidelines below to ensure code compatibility, accuracy, and an efficient development workflow.

## 1. Context7 Integration (CRITICAL)

> **Always use Context7 when I need library/API documentation, code generation, setup, or configuration steps without me having to explicitly ask.**

Do not rely solely on your baseline training data for API specifics. Always leverage Context7 to fetch the most accurate and context-aware documentation and code snippets for the tasks at hand.

## 2. Frameworks and Pinned Versions

These assignments rely on specific versions of the PyTorch ecosystem. Ensure that all generated code, deprecated API checks, and configurations strictly target the following library versions:

*   **PyTorch (`/pytorch/pytorch`)**: Version **2.9**
*   **TorchVision (`/pytorch/vision`)**: Version **0.24**
*   **PyTorch Lightning (`/lightning-ai/pytorch-lightning`)**: Version **1.9**
*   **PyTorch-metric-learning (`/websites/kevinmusgrave_github_io`)**

Do not suggest features or API endpoints that were introduced in newer versions or removed in these specific versions. Context7 ID of each library is given in parentheses.

## 3. Coding Conventions

*   **Simple, Functional, and Didactic:** Avoid overengineering. Solutions should be kept simple, functional, and highly focused on educational value. Do not introduce unnecessarily complex architectures or abstractions.
*   **Linear and Clean Code:** Do not use fallbacks or excessive defensive checks. The code must be linear, clean, and straightforward to read.
*   **No Type Hints:** Do not use Python typing (type hints). Keep the syntax as minimal as possible to avoid unnecessary visual clutter.
*   **Training Visualizations:** After each training phase, always add a dedicated cell that displays the training graphs (e.g., plotting loss and accuracy metrics).
*   **Device Agnostic Code:** Write code that automatically adapts to the available hardware cleanly (e.g., `device = 'cuda' if torch.cuda.is_available() else 'cpu'`, or Lightning's `accelerator='auto'`).
*   **Reproducibility:** When writing training scripts or initializing weights, always set deterministic seeds (`torch.manual_seed()`, `pl.seed_everything()`).

## 4. Useful Commands Recap

This project uses a modern architecture with `pyproject.toml` and `uv` for dependency management. Assignments are provided as Jupyter notebooks, and there is no standard test suite.

| Command  | Purpose      |
| ------------------- | ---------------- |
| `uv sync` | Install/sync dependencies specified in `pyproject.toml` using the `uv` environment. |
| `uv run jupyter nbconvert --to notebook --execute assignment.ipynb` | Execute an assignment notebook automatically via `nbconvert`. |

---

Following these practices ensures that the agent-assisted development workflow stays reliable and mathematically sound. When in doubt regarding an API call, fallback to **Rule 1** and trigger Context7.
