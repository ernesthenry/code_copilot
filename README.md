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
- **`index_codebase`**: Index the codebase for advanced searching and analysis
- **`search_code`**: Search for code symbols in the indexed codebase
- **`get_codebase_statistics`**: Get comprehensive statistics about the indexed codebase
- **`find_similar_functions`**: Find functions with similar names or signatures
- **`list_files`**: List files in a directory matching a pattern
- **`map_code_relationships`**: Map relationships between code components (calls, inheritance, imports)
- **`generate_code_documentation`**: Generate comprehensive documentation for a code file
- **`generate_onboarding_guide`**: Generate a personalized onboarding guide for new developers
- **`suggest_code_improvements`**: Suggest specific improvements for a code file (performance, style, security, maintainability)
- **`explore_code_interactively`**: Interactive code exploration with context-aware responses

### Available Resources

- Automatic discovery of Python files (`.py`)
- Support for other code files (`.js`, `.ts`, `.html`, `.css`, `.json`, `.md`, `.yaml`)
- File content reading with proper MIME type detection

### Available Prompts

- **`code-review`**: Generate comprehensive code review prompts
- **`debug-help`**: Create debugging assistance prompts
- **`optimize-code`**: Generate code optimization prompts
- **`test-generation`**: Create unit test generation prompts
- **`explore-codebase`**: Generate a codebase exploration prompt for various focus areas (overview, architecture, dependencies, complexity, testing, performance, security)

### Advanced Features

- **Semantic Search**: Enhanced code search using vector embeddings (mock or real).
- **Advanced Code Relationship Mapping**: Analyze function calls, class inheritance, and imports.
- **Team Onboarding Assistant**: Generates onboarding guides and checklists for new developers.
- **Interactive Code Explorer**: Context-aware exploration and explanation of code.
- **Code Improvement Suggestions**: Automated suggestions for performance, style, security, and maintainability.

### Example Advanced Usage

```
# Index the codebase for advanced search and analysis
index_codebase

# Search for all functions named 'process_data'
search_code "process_data" symbol_type="function"

# Get statistics about the codebase
get_codebase_statistics

# Find similar functions to 'load_data'
find_similar_functions "load_data"

# List all Python files in the src directory
list_files directory="src" pattern="*.py"

# Map code relationships in a file
map_code_relationships file_path="src/utils.py"

# Generate documentation for a file
generate_code_documentation file_path="src/main.py"

# Get onboarding guide for architecture
generate_onboarding_guide focus_area="architecture"

# Suggest code improvements for a file
suggest_code_improvements file_path="src/main.py" improvement_type="all"

# Explore code interactively (e.g., "How does authentication work?")
explore_code_interactively query="How does authentication work?"
```

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

##  File Structure

fastmcp_copilot/
├── __init__.py
├── main.py                     # Main server entry point
├── config.py                   # Configuration settings
├── models/
│   ├── __init__.py
│   ├── code_symbol.py          # CodeSymbol and FileIndex models
│   └── analysis_result.py      # Analysis result models
├── core/
│   ├── __init__.py
│   ├── indexer.py              # CodeIndexer class
│   ├── database.py             # Database operations
│   └── parsers/
│       ├── __init__.py
│       ├── base.py             # Base parser interface
│       ├── python_parser.py    # Python-specific parsing
│       └── generic_parser.py   # Generic file parsing
├── tools/
│   ├── __init__.py
│   ├── indexing_tools.py       # Indexing-related tools
│   ├── analysis_tools.py       # Code analysis tools
│   ├── file_tools.py           # File operations
│   ├── git_tools.py            # Git operations
│   └── development_tools.py    # Testing, formatting, linting
├── resources/
│   ├── __init__.py
│   └── file_resources.py       # File resource handlers
├── prompts/
│   ├── __init__.py
│   └── code_prompts.py         # Prompt handlers
└── utils/
    ├── __init__.py
    ├── git_utils.py            # Git utilities
    └── file_utils.py           # File utilities
```

## How Users Will Interact with This Code Copilot Server

There are several ways users can access and efficiently use the Code Copilot Server, depending on your team's workflow and preferences:

### 1. Claude Desktop (or Other MCP-Compatible AI Assistants)
- **Description:** Claude Desktop is a desktop application that supports the Model Context Protocol (MCP) and can connect directly to your Code Copilot Server.
- **How it works:** Users interact with a chat-based AI interface. When they ask questions or request code analysis, the assistant uses the tools exposed by your server (e.g., code analysis, docstring parsing, code search).
- **Pros:**
  - Natural language interface (chat).
  - Tool calling is seamless—users just ask, and the AI uses the right tool.
  - Supports multiple servers and codebases.
  - No coding required for end users.
- **Cons:**
  - Requires users to install and configure the desktop app.
  - Some advanced features may depend on the assistant’s capabilities.
- **Recommended for:** Teams who want a chat-based, AI-powered experience with minimal setup for end users.

### 2. Custom Web UI (React, Vue, etc.)
- **Description:** Build a web dashboard that connects to the Code Copilot Server via MCP or HTTP (if you add an HTTP layer).
- **How it works:** Users interact with a web interface to run code analysis, search, doc parsing, etc.
- **Pros:**
  - Fully customizable UI/UX.
  - Can integrate with other internal tools.
  - Can provide visualizations, file explorers, etc.
- **Cons:**
  - Requires web development resources.
  - More maintenance overhead.
- **Recommended for:** Organizations wanting a branded, integrated, or highly customized experience.

### 3. Command-Line Client (CLI)
- **Description:** Use or build a CLI tool (e.g., `mcp_client.py` in your repo) to interact with the server.
- **How it works:** Users run commands in the terminal to invoke tools, get results, and automate workflows.
- **Pros:**
  - Scriptable and automatable.
  - Lightweight, no GUI required.
  - Good for power users and CI/CD integration.
- **Cons:**
  - Less user-friendly for non-technical users.
  - No graphical interface.
- **Recommended for:** Developers, DevOps, and advanced users who prefer the terminal.

### 4. IDE Integration (VSCode, PyCharm, etc.)
- **Description:** Build or use an extension/plugin for popular IDEs that connects to the Code Copilot Server.
- **How it works:** Users access Copilot features directly from their code editor.
- **Pros:**
  - Seamless workflow—no context switching.
  - Can provide code actions, inline suggestions, etc.
- **Cons:**
  - Requires plugin development.
  - May be limited by IDE APIs.
- **Recommended for:** Teams who want the most integrated developer experience.

### Best Out-of-the-Box Option

**Claude Desktop** (or any MCP-compatible AI assistant) is the best out-of-the-box client for most teams:
- No code required to get started.
- Natural language interface.
- Supports all the tools you expose from the server.
- Easy for both technical and non-technical users.

**How to use:**
1. Run your Code Copilot Server.
2. Configure Claude Desktop (or similar) to connect to your server (see the README for config).
3. Users interact via chat and get tool-powered answers.

### Summary Table

| Client Type         | User-Friendliness | Setup Effort | Customization | Best For                |
|---------------------|------------------|--------------|---------------|-------------------------|
| Claude Desktop/AI   | ⭐⭐⭐⭐⭐           | ⭐⭐           | ⭐⭐            | Most users, quick start |
| Web UI              | ⭐⭐⭐⭐            | ⭐⭐⭐⭐         | ⭐⭐⭐⭐⭐         | Custom org needs        |
| CLI                 | ⭐⭐⭐             | ⭐            | ⭐⭐⭐           | Devs, automation        |
| IDE Plugin          | ⭐⭐⭐⭐            | ⭐⭐⭐⭐         | ⭐⭐⭐⭐          | Devs, deep integration  |

**Recommendation:**
Start with Claude Desktop or another MCP-compatible AI assistant for the fastest, most user-friendly experience. Expand to web or IDE integrations as your needs grow.