# Contributing

## Workflow

Feature branches are used for organizing work, but the typical workflow is to
**merge locally to main and push** — not via pull requests:

1. **Create a feature branch** from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b feat/short-description
   ```

2. **Make changes, commit:**

   ```bash
   git add <files>
   git commit -m "description of change"
   ```

3. **Merge to main and push:**

   ```bash
   git checkout main
   git merge feat/short-description
   git push
   ```

Always commit and push after completing features/fixes — don't let changes accumulate.

## Branch Naming

| Prefix | Use |
| --- | --- |
| `feat/` | New features or pipeline stages |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring without behaviour change |
| `test/` | Test additions or fixes |
| `ci/` | CI/CD and infrastructure |

## Checks on a pull request

There are no local git hooks (the pre-commit configuration was removed on
2026-11-28). Every pull request runs two GitHub Actions workflows:

- **Lint** (`.github/workflows/lint.yml`): ruff lint, ruff format check, mypy,
  bandit, vulture, and detect-secrets on the files the pull request changes.
- **CI** (`.github/workflows/ci.yml`): the test suite with a 90% coverage
  threshold, and a Snakemake dry-run.

To run the same checks before pushing:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

Notebook outputs are not stripped automatically. Clear the outputs of a
notebook before committing it.

## Test Coverage

All PRs must maintain **90%+ test coverage** (hard requirement). CI runs `pytest --cov`
on every push; PRs are blocked if coverage drops below the threshold.

- Every function (public and private) needs at least one unit test
- Tests use small synthetic arrays only -- never real data files
- Use `hypothesis` for numerical functions to auto-generate adversarial inputs
- Use `pandera` to validate HDF5 schemas in tests
- Current status: **1500+ tests, 91%+ coverage**

## Citation Policy

Any analysis method or algorithm taken from a paper must be cited in **three places**:

1. **Code** -- module/function docstring with: first author, year, title, journal, DOI,
   and GitHub URL if available
2. **Docs** -- relevant markdown files under `docs/`
3. **Frontend** -- a "Methods & References" expander on any page that uses the method

Format: `Author et al. YEAR. "Title." Journal. doi:XX.XXXX/XXXXX`

## Copilot Code Review

GitHub Copilot can automatically review PRs if you have a Copilot Pro (or higher)
subscription. To enable:

1. Go to **Settings → Copilot → Code review** in the repo
2. Enable automatic reviews for open PRs
3. Copilot will leave inline comments with suggested fixes

This is optional — manual reviews work fine without it.
