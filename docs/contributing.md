# Contributing

## Branch Protection

The `main` branch is protected on GitHub:

- **Direct pushes to `main` are blocked** — all changes go through pull requests
- **PRs require 1 approving review** before merge
- **Stale reviews are dismissed** when new commits are pushed
- **Force pushes and branch deletion are blocked**

## Workflow

1. **Create a feature branch** from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b feat/short-description
   ```

2. **Make changes, commit, push:**

   ```bash
   git add <files>
   git commit -m "description of change"
   git push -u origin feat/short-description
   ```

3. **Open a PR** against `main`:

   ```bash
   gh pr create --title "Short title" --body "Summary of changes"
   ```

4. **Get a review** (from a collaborator or Copilot if available), then merge.

## Branch Naming

| Prefix | Use |
| --- | --- |
| `feat/` | New features or pipeline stages |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring without behaviour change |
| `test/` | Test additions or fixes |
| `ci/` | CI/CD and infrastructure |

## Pre-commit Hooks

Pre-commit hooks run automatically before every `git commit`:

- **ruff** — linting + formatting (replaces black + flake8 + isort)
- **mypy** — static type checking
- **nbstripout** — strips Jupyter notebook outputs to keep git history clean

Install hooks after cloning:

```bash
pre-commit install
```

If a hook fails, fix the issue and re-run `git commit`. Do not use `--no-verify`.

## Test Coverage

All PRs must maintain **90%+ test coverage** (hard requirement). CI runs `pytest --cov`
on every push; PRs are blocked if coverage drops below the threshold.

- Every function (public and private) needs at least one unit test
- Tests use small synthetic arrays only -- never real data files
- Use `hypothesis` for numerical functions to auto-generate adversarial inputs
- Use `pandera` to validate HDF5 schemas in tests
- Current status: **1119+ tests, 91%+ coverage**

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
