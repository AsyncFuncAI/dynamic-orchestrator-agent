# Contributing to DOA Framework 🤝

Thank you for your interest in contributing to the Dynamic Orchestrator Agent (DOA) Framework! We welcome contributions from the community and are excited to collaborate with you.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## 📜 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful, inclusive, and constructive in all interactions.

### Our Standards

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Inclusive**: Welcome people of all backgrounds and experience levels
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Professional**: Maintain a professional tone in all communications

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of reinforcement learning concepts
- Familiarity with PyTorch (helpful but not required)

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/dynamic-orchestrator-agent.git
   cd dynamic-orchestrator-agent
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-org/dynamic-orchestrator-agent.git
   ```

## 🛠️ Development Setup

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --with dev

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
# Run tests to ensure everything is working
pytest tests/

# Run a simple example
python examples/simple_start.py
```

## 🎯 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new functionality
- 📝 **Documentation**: Improve docs, examples, and tutorials
- 🧪 **Tests**: Add or improve test coverage
- 🔧 **Code**: Implement new features or fix bugs
- 🎨 **Examples**: Create demos and use cases

### Reporting Bugs

Before creating a bug report, please:

1. **Check existing issues** to avoid duplicates
2. **Use the latest version** to ensure the bug still exists
3. **Provide detailed information**:
   - Python version
   - DOA framework version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Features

When suggesting new features:

1. **Check existing feature requests** first
2. **Explain the use case** and why it's valuable
3. **Provide examples** of how it would be used
4. **Consider implementation complexity**
5. **Be open to discussion** and alternative approaches

## 🔄 Pull Request Process

### Before You Start

1. **Create an issue** to discuss major changes
2. **Check the roadmap** to align with project direction
3. **Ensure your idea fits** the project scope

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test files
   pytest tests/test_orchestrator.py
   
   # Check code coverage
   pytest --cov=doa_framework tests/
   ```

4. **Format your code**:
   ```bash
   # Format with black
   black doa_framework/ examples/ tests/
   
   # Sort imports
   isort doa_framework/ examples/ tests/
   
   # Type checking
   mypy doa_framework/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new agent orchestration strategy"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- **Use descriptive titles** (e.g., "Add support for custom reward functions")
- **Reference related issues** (e.g., "Fixes #123")
- **Provide clear description** of changes and motivation
- **Include tests** for new functionality
- **Update documentation** if needed
- **Keep PRs focused** - one feature/fix per PR

## 📏 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for consistent import ordering
- **Type hints**: Required for public APIs
- **Docstrings**: Use Google-style docstrings

### Code Quality Tools

```bash
# Format code
black doa_framework/ examples/ tests/

# Sort imports
isort doa_framework/ examples/ tests/

# Lint code
flake8 doa_framework/ examples/ tests/

# Type checking
mypy doa_framework/

# Security check
bandit -r doa_framework/
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `PolicyNetwork`)
- **Functions/Variables**: `snake_case` (e.g., `calculate_reward`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_STEPS`)
- **Private members**: Leading underscore (e.g., `_internal_method`)

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interaction
├── examples/       # Tests for example scripts
└── fixtures/       # Test data and fixtures
```

### Writing Tests

- **Use pytest** for all tests
- **Follow AAA pattern**: Arrange, Act, Assert
- **Use descriptive test names**: `test_orchestrator_selects_correct_agent_for_task`
- **Mock external dependencies**: Use `unittest.mock` or `pytest-mock`
- **Test edge cases**: Error conditions, boundary values, etc.

### Example Test

```python
import pytest
from doa_framework import Orchestrator, PolicyNetwork
from doa_framework.agents import EchoAgent, TerminatorAgent

def test_orchestrator_runs_episode_successfully():
    # Arrange
    agents = [EchoAgent(), TerminatorAgent()]
    policy = PolicyNetwork(64, len(agents), 128)
    orchestrator = Orchestrator(agents, policy)
    
    # Act
    trajectory = orchestrator.run_episode(initial_state)
    
    # Assert
    assert len(trajectory.steps) > 0
    assert trajectory.total_undiscounted_reward is not None
```

## 📚 Documentation

### Types of Documentation

- **API Documentation**: Docstrings for all public functions/classes
- **User Guides**: How-to guides and tutorials
- **Examples**: Working code examples
- **Architecture**: High-level design documentation

### Writing Documentation

- **Use clear, concise language**
- **Provide examples** for complex concepts
- **Keep it up-to-date** with code changes
- **Include type hints** in function signatures

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## 🌟 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page
- **Special mentions** in documentation

## 💬 Community

### Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Discord**: Real-time chat with the community
- **Email**: For private inquiries

### Communication Guidelines

- **Be patient**: Maintainers are volunteers
- **Be specific**: Provide context and details
- **Be respectful**: Follow the code of conduct
- **Search first**: Check existing issues and discussions

## 🎉 Thank You!

Your contributions make the DOA Framework better for everyone. Whether you're fixing a typo, adding a feature, or helping other users, every contribution is valuable and appreciated.

Happy coding! 🚀

---

**Questions?** Feel free to reach out through any of our communication channels. We're here to help!
