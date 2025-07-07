# FastMCP Code Copilot Server Setup

## Requirements

Create a `requirements.txt` file:

```txt
fastmcp>=0.1.0
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional development tools** (for full functionality):
   ```bash
   pip install pytest black flake8
   ```

## Configuration

Create a `mcp_config.json` file to configure the server:

```json
{
  "mcpServers": {
    "code-copilot": {
      "command": "python",
      "args": ["fastmcp_server.py"],
      "env": {}
    }
  }
}
```

## Usage

### Running the Server

1. **Direct execution:**
   ```bash
   python fastmcp_server.py
   ```

2. **As MCP server** (for integration with AI assistants):
   ```bash
   # The server will communicate via stdin/stdout using JSON-RPC
   # This is typically handled by the MCP client (like Claude Desktop)
   ```

### Available Tools

The server provides these tools:

- **`analyze_code`**: Analyze Python code for syntax, complexity, and structure
- **`run_tests`**: Execute Python tests using pytest
- **`format_code`**: Format Python code using black
- **`git_status`**: Get git repository status
- **`create_file`**: Create new files with specified content
- **`lint_code`**: Lint Python code using flake8
- **`get_dependencies`**: Extract dependencies from Python files

### Available Resources

- Automatic discovery of Python files (`.py`)
- Support for other code files (`.js`, `.ts`, `.html`, `.css`, `.json`, `.md`, `.yaml`)
- File content reading with proper MIME type detection

### Available Prompts

- **`code-review`**: Generate comprehensive code review prompts
- **`debug-help`**: Create debugging assistance prompts
- **`optimize-code`**: Generate code optimization prompts
- **`test-generation`**: Create unit test generation prompts

## Key Improvements with FastMCP

### 1. **Simplified Architecture**
- Decorators for easy tool/resource/prompt registration
- Automatic JSON-RPC handling
- Built-in error handling and validation

### 2. **Enhanced Type Safety**
- Proper type hints throughout
- Automatic parameter validation
- Better error messages

### 3. **Better Resource Management**
- Automatic resource discovery
- Proper MIME type handling
- Clean resource URIs

### 4. **Extended Functionality**
- Additional tools (lint_code, get_dependencies)
- More comprehensive prompts
- Better error handling

### 5. **Improved Developer Experience**
- Cleaner, more maintainable code
- Better separation of concerns
- Easier to extend and modify

## Example Usage in AI Assistant

Once integrated with an AI assistant that supports MCP:

```
Human: Analyze this Python code for issues:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

AI: I'll analyze this code for you using the code analysis tool.
[Uses analyze_code tool]

The code analysis shows:
- Syntax is valid
- Simple recursive implementation
- Missing input validation for negative numbers
- No handling for large numbers that could cause stack overflow
- Consider adding type hints and docstring

Would you like me to help optimize this code?
```

## Integration with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "code-copilot": {
      "command": "python",
      "args": ["/path/to/fastmcp_server.py"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Module not found errors:**
   - Ensure fastmcp is installed: `pip install fastmcp`
   - Check Python path and virtual environment

2. **Permission errors:**
   - Ensure the script has execute permissions
   - Check file/directory permissions for the working directory

3. **Tool execution failures:**
   - Ensure required tools are installed (pytest, black, flake8)
   - Check that git is available if using git_status

### Debug Mode

Run with debug logging:
```bash
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from fastmcp_server import app
app.run()
"
```

## Contributing

The FastMCP implementation makes it easy to add new tools:

```python
@app.tool()
async def my_new_tool(param1: str, param2: int = 42) -> str:
    \"\"\"
    Description of what this tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
    
    Returns:
        Description of return value
    \"\"\"
    # Tool implementation
    return "result"
```

This approach with FastMCP provides a much cleaner, more maintainable, and feature-rich MCP server compared to the manual JSON-RPC implementation.