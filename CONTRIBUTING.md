# Contributing to Insights Agent

Thank you for considering contributing to Insights Agent! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. A detailed description of the issue
3. Steps to reproduce the bug
4. Expected behavior
5. Actual behavior
6. Screenshots (if applicable)
7. Environment information (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for enhancements! Please create an issue with:

1. A clear, descriptive title
2. A detailed description of the proposed enhancement
3. Any relevant examples or mockups

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request with a clear description of the changes

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Install Playwright browser (for web scraping):
   ```
   playwright install chromium
   ```
6. Copy `.env.example` to `.env` and add your API keys and Airtable configuration

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write tests for new functionality

## Testing

Before submitting a pull request, please run all tests to ensure your changes don't break existing functionality.

## Documentation

Please update the documentation when adding or modifying features. This includes:

- README.md
- Docstrings
- Comments in the code
- Any additional documentation files
