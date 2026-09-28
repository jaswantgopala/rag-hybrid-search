# Deployment Process

## Environments

Code moves through three environments in order: development, staging, and production. Each environment has its own configuration managed through environment variables, never hardcoded values.

## Deployment Pipeline

Deployments are triggered automatically when a pull request is merged into `main`, which deploys to staging. Production deployments require a manual approval step in the CI/CD pipeline, gated behind a second reviewer.

## Rollback Procedure

If a production deployment causes issues, use the one-click rollback button in the deployment dashboard, which reverts to the last known good build within 2 minutes. This is the same rollback checklist referenced in the incident response runbook.

## Deployment Windows

Production deployments are avoided on Fridays after 2 PM and are blocked entirely during the last week of each fiscal quarter to reduce risk during high-traffic periods.