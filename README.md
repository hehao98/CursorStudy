# Cursor Study

## Requirements

- Python 3.12 or higher
- Poetry for dependency management

## Installation

```bash
poetry install
poetry run pre-commit install
```

## Development

This project uses several development tools:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `pre-commit` for git hooks
- `jupyter` and `jupyterlab` for interactive analysis

### Code Quality

The project uses pre-commit hooks to ensure code quality. These hooks run automatically when you commit changes. They include:

- Basic file checks (trailing whitespace, file endings, etc.)
- Code formatting with Black
- Import sorting with isort
- Code linting with flake8

To run the hooks manually:
```bash
poetry run pre-commit run --all-files
```

### Data Management

- Place your data files in the `data/` directory
- Add data files to `.gitignore` if they are not meant to be tracked
