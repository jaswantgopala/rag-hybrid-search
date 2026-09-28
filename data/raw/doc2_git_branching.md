# Git Branching Strategy

## Branch Naming

Feature branches should follow the pattern `feature/short-description`, bugfix branches `fix/short-description`, and hotfixes `hotfix/short-description`. All branches are created from `main`.

## Merge Process

Once a feature branch is ready, open a pull request against `main`. As outlined in the code review process, at least two approvals are required before merging. Squash commits when merging to keep history clean.

## Release Branches

Release branches are cut from `main` when a version is ready for QA. No new features are merged into a release branch, only bug fixes. Once QA signs off, the release branch is tagged and deployed.

## Hotfix Process

Hotfixes bypass the normal release cycle. They are branched directly from the production tag, tested in staging, and merged back into both `main` and the active release branch.