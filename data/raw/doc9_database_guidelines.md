# Database Guidelines

## Schema Changes

All schema migrations must be backward-compatible for at least one deployment cycle to support zero-downtime deploys. Destructive changes like column drops require a two-step migration: deprecate, then remove in a later release.

## Query Performance

Any query expected to run against a table with more than 1 million rows must include an EXPLAIN ANALYZE review in the pull request description. Queries without proper indexing will be flagged in code review.

## Backups

Production databases are backed up every 6 hours with a 30-day retention window. Point-in-time recovery is available for the last 7 days in case of accidental data corruption.

## Access Control

Direct production database access requires a time-limited credential issued through the internal access-request tool, automatically expiring after 4 hours.