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

## Copilot Code Review

GitHub Copilot can automatically review PRs if you have a Copilot Pro (or higher)
subscription. To enable:

1. Go to **Settings → Copilot → Code review** in the repo
2. Enable automatic reviews for open PRs
3. Copilot will leave inline comments with suggested fixes

This is optional — manual reviews work fine without it.
