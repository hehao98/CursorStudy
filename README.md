# Cursor Study

## Requirements

- Python 3.12 or higher
- Poetry for dependency management

## Installation

```bash
poetry install
poetry run pre-commit install
```

## Environment Variables

This project uses a `.env` file to store sensitive configuration like API tokens. A template file `.env.example` is provided.

1. Copy the example file to create your own `.env` file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and replace the placeholder values with your actual credentials:
   ```
   GITHUB_TOKEN=your_github_token
   ```

   You can create a GitHub personal access token at https://github.com/settings/tokens

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

- The `data/` directory contains curated data files
- Add data files to `temp/` if they are not meant to be tracked
