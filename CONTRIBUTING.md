# Contributing to Stock Spike Replicator

We welcome contributions to the Stock Spike Replicator project! This document provides guidelines for contributing to the project. By participating in this project, you agree to abide by its terms.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Style Guidelines](#style-guidelines)
5. [Commit Guidelines](#commit-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Development Environment Setup](#development-environment-setup)
8. [Running Tests](#running-tests)
9. [Reporting Bugs](#reporting-bugs)
10. [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [project_email@example.com].

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Set up your development environment as described in the [Development Environment Setup](#development-environment-setup) section.
4. Create a branch for your changes.

## How to Contribute

1. Ensure your changes adhere to the [Style Guidelines](#style-guidelines).
2. Make your changes in your forked repository.
3. Add or update tests as necessary.
4. Ensure the test suite passes.
5. Commit your changes following the [Commit Guidelines](#commit-guidelines).
6. Push to your fork and submit a pull request.

## Style Guidelines

### Python

- Follow PEP 8 style guide for Python code.
- Use 4 spaces for indentation.
- Maximum line length is 100 characters.
- Use docstrings for all public modules, functions, classes, and methods.

### JavaScript

- Follow the Airbnb JavaScript Style Guide.
- Use 2 spaces for indentation.
- Use semicolons at the end of each statement.
- Use single quotes for strings.

### CSS

- Follow the BEM (Block Element Modifier) methodology for naming classes.
- Use 2 spaces for indentation.

## Commit Guidelines

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally after the first line.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

## Development Environment Setup

1. Install Python 3.7+ and Node.js 14+.
2. Install PostgreSQL 12+.
3. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-spike-replicator.git
   cd stock-spike-replicator
   ```
4. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
5. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Set up the database:
   - Create a PostgreSQL database for the project
   - Update the `DATABASE_URL` in `api/core/config.py` with your database connection string
   - Run database migrations:
     ```
     alembic upgrade head
     ```
7. Set up the frontend:
   ```
   cd stock-spike-replicator-frontend
   npm install
   ```

## Running Tests

To run the Python tests:
```
python -m pytest
```

To run the JavaScript tests:
```
cd stock-spike-replicator-frontend
npm test
```

## Reporting Bugs

When reporting bugs, please include:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Suggesting Enhancements

When suggesting enhancements, please include:

- A clear and concise description of what the problem is
- Describe the solution you'd like
- Describe alternatives you've considered
- Any additional context or screenshots about the feature request

Thank you for contributing to Stock Spike Replicator!