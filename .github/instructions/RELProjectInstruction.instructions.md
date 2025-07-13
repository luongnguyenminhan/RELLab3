---
applyTo: '**'
---
# INSTRUCTION.md

This file provides comprehensive setup, development, and usage instructions for the Modular Deep Reinforcement Learning (DRL) Framework. The project is architected according to the blueprint derived from "Deep Reinforcement Learning: A Survey" by Wang et al., with a focus on modularity, testability, maintainability, and reproducibility.

## Project Overview
The Modular DRL Framework is a robust, extensible Python library implementing canonical DRL algorithms across value-based, policy-based, and maximum entropy-based categories. The design is informed by the survey's classification and analysis, ensuring a clear mapping from theory to code. The framework is intended for research, experimentation, and practical application development.

## Project Setup
1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Review the `docs/` directory for detailed documentation on algorithms, usage, and contributing guidelines.
4. Explore the `src/` directory for modular source code, organized by algorithms, networks, buffers, environments, utilities, and evaluation tools.
5. Use YAML files in `configs/` to manage hyperparameters and ensure experiment reproducibility.

## Development Guidelines
- **Modular Design:** Follow the established directory structure (`src/algorithms/`, `src/networks/`, etc.) to ensure separation of concerns and ease of extension.
- **Docstrings & Documentation:** Every Python file must begin with a Google-style module docstring, including summary, detailed description, key concepts, parameters, expected inputs/outputs, dependencies, and author/date. See the project docs for examples.
- **Configuration Management:** All experiment and algorithm parameters should be defined in YAML files under `configs/`. Avoid hardcoding hyperparameters in code.
- **Testing:** Write unit tests for all new modules in `tests/unit/` and integration tests in `tests/integration/`. Tests should cover core logic, edge cases, and reproducibility.
- **Reproducibility:** Ensure all experiments can be reproduced by specifying random seeds, saving configurations, and logging results.
- **Logging & Evaluation:** Use the utilities in `src/utils/logger.py` and `src/evaluation/` to track training progress and evaluate results. Visualizations should be generated using `matplotlib` or `seaborn`.
- **Extensibility:** When adding new algorithms or improvements (e.g., Rainbow, prioritized replay), implement them as modular plug-ins or configurable options, not as monolithic code.
- **Code Quality:** Adhere to PEP8 and project-specific style guides. Use type hints and meaningful variable names. Document all public classes and functions.
- **Dependencies:** Only use packages listed in `requirements.txt`. If new dependencies are needed, update the requirements and document the rationale.

## Usage
- **Example Scripts:** The `examples/` directory contains scripts for running agents on standard environments (e.g., CartPole, Pendulum, BipedalWalker). Use these as templates for new experiments.
- **Jupyter Notebooks:** The `notebooks/` directory provides interactive analysis and demonstration notebooks. Use these for exploratory work and sharing results.
- **Configuration:** Select and modify YAML files in `configs/` to set hyperparameters for each algorithm. Document any changes for reproducibility.
- **Running Experiments:** Use the provided scripts or notebooks to train and evaluate agents. Log all results and configurations for future reference.
- **Documentation:** Refer to `docs/algorithms.md` for algorithm details, `docs/usage.md` for practical guides, and `docs/contributing.md` for contribution standards.

## Contribution Process
- Fork the repository and create a feature branch.
- Write clear, modular code with appropriate docstrings and tests.
- Ensure all tests pass and code is linted.
- Submit a pull request with a detailed description of changes and rationale.
- Participate in code reviews and address feedback promptly.

## Engineering Standards
- Prioritize modularity, clarity, and maintainability in all code and documentation.
- Ensure all new features are testable and reproducible.
- Maintain a clear mapping between theoretical concepts (from the survey) and code modules.
- Foster a collaborative, open-source development environment.

For further details, see the full project documentation in the `docs/` directory and the original survey paper by Wang et al.
