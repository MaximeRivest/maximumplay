# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python experimentation repository called "maximumplay" that explores different AI/ML libraries and frameworks. The project uses UV for dependency management and contains several playground modules for testing different technologies.

## Dependencies and Setup

The project uses Python 3.11+ and manages dependencies through:
- `pyproject.toml` - Main project configuration with dependencies: attachments, dspy, ipykernel, pixeltable
- `uv.lock` - Dependency lock file managed by UV package manager

## Common Development Commands

Since this is an experimental repository, development typically involves:
- `uv run python src/module_name/script.py` - Run individual playground scripts
- `uv sync` - Sync dependencies from lock file
- `uv add <package>` - Add new dependencies

## Architecture and Structure

The codebase is organized as a collection of independent playground modules:

- `src/musing_on_tools/` - DSPy experiments with custom markdown adapters and tool calling
  - `dspy_md_tool_adapter.py` - Custom DSPy adapter implementing markdown-based tool calling with ReAct pattern
  - Contains research data and visualizations about MCP trends
  
- `src/pixelplay/` - Pixeltable experiments for structured data processing
  - `play.py` - Basic Pixeltable usage with computed columns and data insertion
  
- `src/attjsplay/` - Attachments.js experiments for document processing
  - `play.ts` - Examples of using attachments library for PDF/document processing with AI models

- `src/play_context_engineering/` - Context engineering experiments
- `src/vanilla_dspy/` - Pure DSPy experiments and tutorials

## Key Technical Details

### DSPy Custom Adapter
The `dspy_md_tool_adapter.py` contains a sophisticated implementation of:
- `MarkdownAdapter` - Custom DSPy adapter for markdown-based structured output
- `ReAct_md` - Modified ReAct pattern supporting code execution via `# | run` blocks
- Support for unrestricted, guided, and strict execution modes
- Tool calling through markdown code blocks with result formatting

### Pixeltable Integration
The pixelplay module demonstrates:
- Creating tables with typed columns
- Inserting structured data
- Adding computed columns for automatic calculations
- Basic querying and data retrieval

### Document Processing
The attjsplay module shows:
- Using attachments-js for PDF and document processing
- Integration with Anthropic's Claude API
- Pipeline-based document transformation and analysis

## Development Notes

This is an experimental repository where each module is self-contained. When working on code:
- Each src subdirectory is independent and can be run separately
- The project uses Jupyter-style cell markers (#%%) for interactive development
- Many files contain embedded examples and test cases
- The repository contains research data (CSV files, images) related to MCP trends analysis