# Code Style Guide

## General Principles

Code should be written for readability first, performance second, unless a specific performance requirement is documented. Prefer explicit over clever, and keep functions under 40 lines where possible.

## Naming Conventions

Use `snake_case` for Python variables and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Avoid abbreviations unless they are widely understood within the team.

## Testing Requirements

Every new function must have at least one corresponding unit test. Pull requests that reduce overall test coverage will not pass the automated code review process outlined in the PR guidelines.

## Documentation

Public functions and classes require a docstring explaining their purpose, parameters, and return value. Internal helper functions are exempt unless their logic is non-obvious.