# Monitoring and Alerting Guidelines

## Alert Severity Levels

Alerts are classified as P1 (critical, page immediately), P2 (urgent, respond within 30 minutes), or P3 (non-urgent, review during business hours). Severity is defined in each alert's configuration in the observability dashboard.

## Dashboard Standards

Every service must have a dashboard tracking request rate, error rate, and latency (the RED metrics). Dashboards should be linked from the service's README for easy discovery during incidents.

## Alert Fatigue

Alerts that fire more than 10 times per week without leading to action should be reviewed and either tuned or removed. Excess noisy alerts are tracked in the monthly alert-health review.

## Synthetic Monitoring

Critical user flows are monitored with synthetic checks every 5 minutes from three geographic regions. A failure in two consecutive checks from the same region triggers a P2 alert.