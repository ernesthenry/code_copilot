# MCP Code Copilot Server - Configuration & Usage Guide

## Overview

This Python MCP server provides a comprehensive code copilot that can:
- Analyze code for syntax errors and complexity
- Read and create files
- Run tests and format code
- Provide git repository status
- Offer pre-configured prompts for common development tasks

## Installation & Setup

### 1. Dependencies

First, install the required Python packages:

```bash
pip install black pytest  # Optional: for code formatting and testing
```

### 2. Make the Server Executable

```bash
chmod +x mcp_server.py
```

### 3. VS Code Configuration

Create or update your VS Code MCP configuration file:

**Location:** `~/.vscode/mcp.json` (macOS/Linux) or `%APPDATA%\Code\User\mcp.json` (Windows)

```json
{
  "servers": {
    "code-copilot": {
      "command": "python",
      "args": ["/path/to/your/mcp_server.py"],
      "description": "Code analysis and development tools",
      "capabilities": {
        "resources": true,
        "tools": true,
        "prompts": true
      }
    }
  }
}
```

## Core MCP Concepts Explained

### 1. **Resources** 
Think of resources as "readable files" that the AI can access:
- **What**: File contents, documentation, API responses
- **How**: The server lists available files, AI can request to read them
- **Example**: Python files in your project directory

### 2. **Tools**
Tools are functions the AI can execute:
- **What**: Code analysis, file operations, running commands
- **How**: AI calls tools with parameters, gets results back
- **Example**: `analyze_code` tool checks syntax and complexity

### 3. **Prompts**
Pre-configured prompt templates for common tasks:
- **What**: Ready-made prompts for code review, debugging, optimization
- **How**: AI can use these templates with your specific context
- **Example**: Code review prompt that analyzes quality and security

## Available Tools

### `analyze_code`
Analyzes Python code for:
- Syntax errors
- Code structure (classes, functions, imports)
- Complexity estimation

**Usage:**
```json
{
  "name": "analyze_code",
  "arguments": {
    "code": "def hello():\n    print('world')",
    "file_path": "example.py"
  }
}
```

### `run_tests`
Executes Python tests using pytest:
- Runs specific test files or directories
- Provides detailed output including failures

**Usage:**
```json
{
  "name": "run_tests",
  "arguments": {
    "test_path": "tests/",
    "verbose": true
  }
}
```

### `format_code`
Formats Python code using Black:
- Consistent code style
- Configurable line length

**Usage:**
```json
{
  "name": "format_code",
  "arguments": {
    "code": "def hello( ):\nprint( 'world' )",
    "line_length": 88
  }
}
```

### `git_status`
Provides git repository information:
- Current branch
- File changes (modified, added, deleted)
- Working tree status

**Usage:**
```json
{
  "name": "git_status",
  "arguments": {
    "path": "."
  }
}
```

### `create_file`
Creates new files with specified content:
- Creates parent directories if needed
- Optional overwrite protection

**Usage:**
```json
{
  "name": "create_file",
  "arguments": {
    "file_path": "src/new_module.py",
    "content": "# New Python module\n\ndef main():\n    pass",
    "overwrite": false
  }
}
```

## Available Prompts

### `code_review`
Comprehensive code review focusing on:
- Code quality and readability
- Security issues
- Performance optimizations
- Best practices

### `debug_help`
Debugging assistance providing:
- Error explanation
- Likely causes
- Step-by-step debugging
- Fix suggestions

### `optimize_code`
Performance optimization suggestions:
- Time complexity improvements
- Memory usage optimization
- Algorithm efficiency
- Best practices

## Testing the Server

### 1. Direct Testing
You can test the server directly using JSON-RPC messages:

```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}' | python mcp_server.py
```

### 2. Integration Testing
Once configured with VS Code or another MCP client, you can:
1. Ask the AI to analyze code files in your project
2. Request code reviews
3. Run tests and get formatted output
4. Get git status information

## Protocol Details

### Message Format
The server uses JSON-RPC 2.0 over stdio:
- **Input**: JSON messages on stdin (one per line)
- **Output**: JSON responses on stdout
- **Error Handling**: Proper JSON-RPC error responses

### Capabilities
- **Resources**: ✅ File reading and listing
- **Tools**: ✅ Code analysis, testing, formatting, git operations
- **Prompts**: ✅ Pre-configured development prompts

## Extending the Server

### Adding New Tools
1. Add tool definition in `handle_list_tools()`
2. Implement tool logic in `handle_call_tool()`
3. Add the actual function (like `analyze_code()`)

### Adding New Resources
1. Modify `handle_list_resources()` to include new resource types
2. Update `handle_read_resource()` to handle new URI schemes

### Adding New Prompts
1. Add prompt definition in `handle_list_prompts()`
2. Implement prompt generation in `handle_get_prompt()`
3. Add prompt template function

## Troubleshooting

### Common Issues

1. **Server not starting**: Check Python path and permissions
2. **Tools failing**: Ensure required dependencies (black, pytest) are installed
3. **File access errors**: Verify file paths and permissions
4. **Git commands failing**: Ensure you're in a git repository

### Debugging
The server logs important events to stderr. Enable debug logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Notes
- File operations are limited to the current directory tree
- Commands have timeout limits (10-30 seconds)
- Large files may impact performance

##