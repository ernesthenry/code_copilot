#!/usr/bin/env python3
"""
FastMCP Code Copilot Server
A Model Context Protocol server built with FastMCP that provides code analysis, 
file operations, and development tools for AI assistants.
"""

import asyncio
import json
import os
import subprocess
import ast
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# FastMCP imports
from fastmcp import FastMCP
from fastmcp.resources import Resource
from fastmcp.tools import Tool
from fastmcp.prompts import Prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
app = FastMCP("Code Copilot Server", version="1.0.0")

# Resource handlers
@app.resource("file://{path:path}")
async def read_file(path: str) -> Resource:
    """Read a file resource."""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        content = file_path.read_text(encoding='utf-8')
        return Resource(
            uri=f"file://{file_path.absolute()}",
            name=str(file_path.name),
            description=f"File: {file_path.name}",
            text=content,
            mimeType="text/plain"
        )
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available file resources in the current directory."""
    resources = []
    
    try:
        current_dir = Path.cwd()
        
        # Python files
        for file_path in current_dir.rglob("*.py"):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                resources.append(Resource(
                    uri=f"file://{file_path.absolute()}",
                    name=str(file_path.relative_to(current_dir)),
                    description=f"Python file: {file_path.name}",
                    mimeType="text/x-python"
                ))
        
        # Other code files
        for ext in [".js", ".ts", ".html", ".css", ".json", ".md", ".yaml", ".yml"]:
            for file_path in current_dir.rglob(f"*{ext}"):
                if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                    resources.append(Resource(
                        uri=f"file://{file_path.absolute()}",
                        name=str(file_path.relative_to(current_dir)),
                        description=f"{ext.upper()} file: {file_path.name}",
                        mimeType="text/plain"
                    ))
                    
    except Exception as e:
        logger.warning(f"Error listing resources: {e}")
    
    return resources

# Tool handlers
@app.tool()
async def analyze_code(code: str, file_path: str = "<string>") -> str:
    """
    Analyze Python code for syntax errors, complexity, and structure.
    
    Args:
        code: Python code to analyze
        file_path: Optional file path for context
    
    Returns:
        JSON string with analysis results
    """
    analysis = {
        "file_path": file_path,
        "syntax_valid": False,
        "syntax_errors": [],
        "complexity_info": {},
        "structure_info": {}
    }
    
    try:
        # Parse the code to check syntax
        tree = ast.parse(code, filename=file_path)
        analysis["syntax_valid"] = True
        
        # Analyze structure
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        
        analysis["structure_info"] = {
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "lines_of_code": len(code.splitlines())
        }
        
        # Simple complexity analysis
        analysis["complexity_info"] = {
            "num_classes": len(classes),
            "num_functions": len(functions),
            "num_imports": len(imports),
            "estimated_complexity": "Low" if len(functions) + len(classes) < 10 else "Medium"
        }
        
    except SyntaxError as e:
        analysis["syntax_errors"].append({
            "line": e.lineno,
            "message": e.msg,
            "text": e.text
        })
    
    return json.dumps(analysis, indent=2)

@app.tool()
async def run_tests(test_path: str, verbose: bool = False) -> str:
    """
    Run Python tests using pytest.
    
    Args:
        test_path: Path to test file or directory
        verbose: Enable verbose output
    
    Returns:
        Test execution results
    """
    if not Path(test_path).exists():
        return f"Error: Test path '{test_path}' does not exist"
    
    try:
        cmd = ["python", "-m", "pytest", test_path]
        if verbose:
            cmd.append("-v")
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        
        output = f"Exit code: {result.returncode}\n\n"
        output += f"STDOUT:\n{result.stdout}\n\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}"
        
        return output
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out (30s limit)"
    except Exception as e:
        return f"Error running tests: {str(e)}"

@app.tool()
async def format_code(code: str, line_length: int = 88) -> str:
    """
    Format Python code using black formatter.
    
    Args:
        code: Python code to format
        line_length: Maximum line length (default: 88)
    
    Returns:
        Formatted code or error message
    """
    try:
        # Run black formatter
        result = subprocess.run([
            "python", "-m", "black", 
            "--line-length", str(line_length),
            "--code", code
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return result.stdout if result.stdout else code
        else:
            return f"Formatting error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Error: Code formatting timed out"
    except Exception as e:
        return f"Error formatting code: {str(e)}"

@app.tool()
async def git_status(path: str = ".") -> str:
    """
    Get git repository status.
    
    Args:
        path: Repository path (default: current directory)
    
    Returns:
        Git status information
    """
    try:
        # Check if it's a git repository
        result = subprocess.run([
            "git", "-C", path, "status", "--porcelain"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return f"Error: Not a git repository or git command failed\n{result.stderr}"
        
        status_output = result.stdout.strip()
        
        # Get additional git info
        branch_result = subprocess.run([
            "git", "-C", path, "branch", "--show-current"
        ], capture_output=True, text=True, timeout=5)
        
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
        
        output = f"Git Status for: {Path(path).absolute()}\n"
        output += f"Current branch: {current_branch}\n\n"
        
        if status_output:
            output += "Changes:\n"
            for line in status_output.split('\n'):
                if line.strip():
                    status_code = line[:2]
                    file_path = line[3:]
                    status_desc = get_git_status_description(status_code)
                    output += f"  {status_desc}: {file_path}\n"
        else:
            output += "Working tree clean - no changes detected\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return "Error: Git command timed out"
    except Exception as e:
        return f"Error getting git status: {str(e)}"

def get_git_status_description(status_code: str) -> str:
    """Convert git status codes to human-readable descriptions."""
    status_map = {
        "??": "Untracked",
        "A ": "Added",
        "M ": "Modified",
        " M": "Modified",
        "MM": "Modified (staged & unstaged)",
        "D ": "Deleted",
        " D": "Deleted",
        "R ": "Renamed",
        "C ": "Copied"
    }
    return status_map.get(status_code, f"Unknown ({status_code})")

@app.tool()
async def create_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Create a new file with specified content.
    
    Args:
        file_path: Path for the new file
        content: File content
        overwrite: Whether to overwrite existing file
    
    Returns:
        Success message or error description
    """
    path = Path(file_path)
    
    try:
        if path.exists() and not overwrite:
            return f"Error: File '{file_path}' already exists. Use overwrite=true to replace it."
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        path.write_text(content, encoding='utf-8')
        
        return f"Successfully created file: {path.absolute()}\nSize: {len(content)} characters"
        
    except Exception as e:
        return f"Error creating file: {str(e)}"

@app.tool()
async def lint_code(code: str, file_path: str = "<string>") -> str:
    """
    Lint Python code using flake8.
    
    Args:
        code: Python code to lint
        file_path: Optional file path for context
    
    Returns:
        Linting results
    """
    try:
        # Create a temporary file for linting
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        # Run flake8
        result = subprocess.run([
            "python", "-m", "flake8", tmp_path, "--max-line-length=88"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return "No linting issues found!"
        else:
            # Replace temp file path with original file path in output
            output = result.stdout.replace(tmp_path, file_path)
            return f"Linting issues found:\n{output}"
            
    except subprocess.TimeoutExpired:
        return "Error: Linting timed out"
    except Exception as e:
        return f"Error linting code: {str(e)}"
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.tool()
async def get_dependencies(file_path: str) -> str:
    """
    Extract dependencies from a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        List of dependencies
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        content = path.read_text(encoding='utf-8')
        tree = ast.parse(content, filename=file_path)
        
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])
        
        # Filter out standard library modules (basic check)
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'subprocess', 'tempfile', 'pathlib',
            'typing', 'logging', 'asyncio', 'collections', 'itertools', 're'
        }
        
        third_party_deps = dependencies - stdlib_modules
        
        result = {
            "file_path": file_path,
            "all_imports": sorted(list(dependencies)),
            "third_party_dependencies": sorted(list(third_party_deps)),
            "standard_library": sorted(list(dependencies & stdlib_modules))
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error analyzing dependencies: {str(e)}"

# Prompt handlers
@app.prompt("code-review")
async def code_review_prompt(file_path: str) -> List[Dict[str, Any]]:
    """Generate a code review prompt."""
    prompt_text = f"""Please perform a comprehensive code review of {file_path}. Focus on:

1. **Code Quality & Readability**
   - Variable and function naming
   - Code structure and organization
   - Comments and documentation

2. **Potential Issues**
   - Bugs or logical errors
   - Security vulnerabilities
   - Performance bottlenecks

3. **Best Practices**
   - Python idioms and conventions
   - Error handling
   - Testing considerations

4. **Suggestions**
   - Refactoring opportunities
   - Design pattern improvements
   - Optimization possibilities

Provide specific, actionable feedback with examples where possible."""
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

@app.prompt("debug-help")
async def debug_help_prompt(error_message: str, code_context: str = "") -> List[Dict[str, Any]]:
    """Generate a debugging help prompt."""
    prompt_text = f"""I'm encountering this error: {error_message}

Please help me debug this issue by:
1. **Explaining the Error**: What does this error mean?
2. **Root Cause Analysis**: What are the most likely causes?
3. **Debugging Steps**: Provide step-by-step debugging suggestions
4. **Solution**: Suggest fixes with code examples if applicable
5. **Prevention**: How to avoid this error in the future"""
    
    if code_context:
        prompt_text += f"\n\nHere's the relevant code context:\n```python\n{code_context}\n```"
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

@app.prompt("optimize-code")
async def optimize_code_prompt(code: str) -> List[Dict[str, Any]]:
    """Generate a code optimization prompt."""
    prompt_text = f"""Please analyze this code for performance optimization opportunities:

```python
{code}
```

Focus on:
1. **Time Complexity**: Algorithm efficiency improvements
2. **Memory Usage**: Optimization opportunities
3. **Code Structure**: Better organization and patterns
4. **Python-Specific**: Leveraging Python idioms and built-ins
5. **Scalability**: How the code performs with larger inputs

Provide specific suggestions with improved code examples and explain the performance benefits."""
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

@app.prompt("test-generation")
async def test_generation_prompt(code: str, file_path: str = "") -> List[Dict[str, Any]]:
    """Generate a test generation prompt."""
    prompt_text = f"""Please generate comprehensive unit tests for this code:

```python
{code}
```

Create tests that cover:
1. **Happy Path**: Normal operation scenarios
2. **Edge Cases**: Boundary conditions and unusual inputs
3. **Error Handling**: Exception cases and error conditions
4. **Integration**: How components work together

Use pytest framework and include:
- Clear test function names
- Appropriate fixtures if needed
- Parameterized tests for multiple scenarios
- Mocking for external dependencies
- Clear assertions with helpful error messages"""
    
    if file_path:
        prompt_text += f"\n\nFile path: {file_path}"
    
    return [{"role": "user", "content": {"type": "text", "text": prompt_text}}]

# Main execution
if __name__ == "__main__":
    # Run the FastMCP server
    app.run()