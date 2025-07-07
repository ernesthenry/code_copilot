"""
MCP Code Copilot Server
A Model Context Protocol server that provides code analysis, file operations,
and development tools for AI assistants.

This server demonstrates:
- File system operations (reading, writing, listing)
- Code analysis (syntax checking, complexity analysis)
- Development tools (running tests, formatting code)
- Git operations (status, diff, commit history)
"""

import asyncio
import json
import sys
import os
import subprocess
import ast
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServer:
    """
    Main MCP Server class that handles all protocol communication.
    
    The Model Context Protocol works through JSON-RPC 2.0 messages over stdio.
    This server provides tools and resources for code development assistance.
    """
    
    def __init__(self):
        self.name = "code-copilot-server"
        self.version = "1.0.0"
        self.capabilities = {
            "resources": True,  # Can provide file-like resources
            "tools": True,      # Can execute tools/functions
            "prompts": True     # Can provide pre-configured prompts
        }
        
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main message handler for MCP protocol messages.
        
        MCP uses JSON-RPC 2.0 format:
        - method: The operation to perform
        - params: Parameters for the operation
        - id: Request ID for response matching
        """
        method = message.get("method")
        params = message.get("params", {})
        request_id = message.get("id")
        
        try:
            if method == "initialize":
                return await self.handle_initialize(params, request_id)
            elif method == "resources/list":
                return await self.handle_list_resources(params, request_id)
            elif method == "resources/read":
                return await self.handle_read_resource(params, request_id)
            elif method == "tools/list":
                return await self.handle_list_tools(params, request_id)
            elif method == "tools/call":
                return await self.handle_call_tool(params, request_id)
            elif method == "prompts/list":
                return await self.handle_list_prompts(params, request_id)
            elif method == "prompts/get":
                return await self.handle_get_prompt(params, request_id)
            else:
                return self.create_error_response(
                    request_id, -32601, f"Method not found: {method}"
                )
        except Exception as e:
            logger.error(f"Error handling {method}: {e}")
            return self.create_error_response(
                request_id, -32603, f"Internal error: {str(e)}"
            )
    
    async def handle_initialize(self, params: Dict, request_id: Any) -> Dict[str, Any]:
        """
        Handle MCP initialization request.
        This establishes the server's capabilities and protocol version.
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities,
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        }
    
    async def handle_list_resources(self, params: Dict, request_id: Any) -> Dict[str, Any]:
        """
        List available resources (files, documentation, etc.).
        Resources in MCP are file-like data that can be read by the AI.
        """
        resources = []
        
        # Add current directory files as resources
        try:
            current_dir = Path.cwd()
            for file_path in current_dir.rglob("*.py"):
                if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                    resources.append({
                        "uri": f"file://{file_path.absolute()}",
                        "name": str(file_path.relative_to(current_dir)),
                        "description": f"Python file: {file_path.name}",
                        "mimeType": "text/x-python"
                    })
            
            # Add other common code files
            for ext in [".js", ".ts", ".html", ".css", ".json", ".md"]:
                for file_path in current_dir.rglob(f"*{ext}"):
                    if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                        resources.append({
                            "uri": f"file://{file_path.absolute()}",
                            "name": str(file_path.relative_to(current_dir)),
                            "description": f"{ext.upper()} file: {file_path.name}",
                            "mimeType": "text/plain"
                        })
        except Exception as e:
            logger.warning(f"Error listing resources: {e}")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resources": resources}
        }
    
    async def handle_read_resource(self, params: Dict, request_id: Any) -> Dict[str, Any]:
        """
        Read the contents of a specific resource.
        This allows the AI to access file contents for analysis.
        """
        uri = params.get("uri")
        if not uri or not uri.startswith("file://"):
            return self.create_error_response(
                request_id, -32602, "Invalid or missing file URI"
            )
        
        file_path = Path(uri[7:])  # Remove "file://" prefix
        
        try:
            if not file_path.exists():
                return self.create_error_response(
                    request_id, -32602, f"File not found: {file_path}"
                )
            
            content = file_path.read_text(encoding='utf-8')
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": content
                    }]
                }
            }
        except Exception as e:
            return self.create_error_response(
                request_id, -32603, f"Error reading file: {str(e)}"
            )
    
    async def handle_list_tools(self, params: Dict, request_id: Any) -> Dict[str, Any]:
        """
        List available tools that the AI can execute.
        Tools are functions that perform actions or computations.
        """
        tools = [
            {
                "name": "analyze_code",
                "description": "Analyze Python code for syntax errors, complexity, and structure",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to analyze"},
                        "file_path": {"type": "string", "description": "Optional file path for context"}
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "run_tests",
                "description": "Run Python tests using pytest",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "test_path": {"type": "string", "description": "Path to test file or directory"},
                        "verbose": {"type": "boolean", "description": "Verbose output", "default": False}
                    },
                    "required": ["test_path"]
                }
            },
            {
                "name": "format_code",
                "description": "Format Python code using black formatter",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to format"},
                        "line_length": {"type": "integer", "description": "Maximum line length", "default": 88}
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "git_status",
                "description": "Get git repository status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repository path", "default": "."}
                    }
                }
            },
            {
                "name": "create_file",
                "description": "Create a new file with specified content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path for the new file"},
                        "content": {"type": "string", "description": "File content"},
                        "overwrite": {"type": "boolean", "description": "Overwrite if exists", "default": False}
                    },
                    "required": ["file_path", "content"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools}
        }
    
    async def handle_call_tool(self, params: Dict, request_id: Any) -> Dict[str, Any]:
        """
        Execute a specific tool with given parameters.
        This is where the actual work gets done.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "analyze_code":
                result = await self.analyze_code(arguments)
            elif tool_name == "run_tests":
                result = await self.run_tests(arguments)
            elif tool_name == "format_code":
                result = await self.format_code(arguments)
            elif tool_name == "git_status":
                result = await self.git_status(arguments)
            elif tool_name == "create_file":
                result = await self.create_file(arguments)
            else:
                return self.create_error_response(
                    request_id, -32602, f"Unknown tool: {tool_name}"
                )
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": result}]}
            }
        except Exception as e:
            return self.create_error_response(
                request_id, -32603, f"Tool execution error: {str(e)}"
            )
    
    async def analyze_code(self, args: Dict) -> str:
        """Analyze Python code for syntax, complexity, and structure."""
        code = args.get("code", "")
        file_path = args.get("file_path", "<string>")
        
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
    
    async def run_tests(self, args: Dict) -> str:
        """Run Python tests using pytest."""
        test_path = args.get("test_path", ".")
        verbose = args.get("verbose", False)
        
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
    
    async def format_code(self, args: Dict) -> str:
        """Format Python code using black formatter."""
        code = args.get("code", "")
        line_length = args.get("line_length", 88)
        
        try:
            # Create a temporary file for formatting
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
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
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    async def git_status(self, args: Dict) -> str:
        """Get git repository status."""
        path = args.get("path", ".")
        
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
                        status_desc = self.get_git_status_description(status_code)
                        output += f"  {status_desc}: {file_path}\n"
            else:
                output += "Working tree clean - no changes detected\n"
            
            return output
            
        except subprocess.TimeoutExpired:
            return "Error: Git command timed out"
        except Exception as e:
            return f"Error getting git status: {str(e)}"
    
    def get_git_status_description(self, status_code: str) -> str:
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
    
    async def create_file(self, args: Dict) -> str:
        """Create a new file with specified content."""
        file_path = Path(args.get("file_path", ""))
        content = args.get("content", "")
        overwrite = args.get("overwrite", False)
        
        if not file_path:
            return "Error: file_path is required"
        
        try:
            if file_path.exists() and not overwrite:
                return f"Error: File '{file_path}' already exists. Use overwrite=true to replace it."
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            file_path.write_text(content, encoding='utf-8')
            
            return f"Successfully created file: {file_path.absolute()}\nSize: {len(content)} characters"
            
        except Exception as e:
            return f"Error creating file: {str(e)}"
    
    async def handle_list_prompts(self, params: Dict, request_id: Any) -> Dict[str, Any]:
        """List available prompts for common development tasks."""
        prompts = [
            {
                "name": "code_review",
                "description": "Perform a comprehensive code review",
                "arguments": [
                    {
                        "name": "file_path",
                        "description": "Path to the code file to review",
                        "required": True
                    }
                ]
            },
            {
                "name": "debug_help",
                "description": "Help debug code issues",
                "arguments": [
                    {
                        "name": "error_message",
                        "description": "The error message or issue description",
                        "required": True
                    },
                    {
                        "name": "code_context",
                        "description": "Relevant code context",
                        "required": False
                    }
                ]
            },
            {
                "name": "optimize_code",
                "description": "Suggest optimizations for code performance",
                "arguments": [
                    {
                        "name": "code",
                        "description": "Code to optimize",
                        "required": True
                    }
                ]
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"prompts": prompts}
        }
    
    async def handle_get_prompt(self, params: Dict, request_id: Any) -> Dict[str, Any]:
        """Get a specific prompt with arguments filled in."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        prompts = {
            "code_review": self.get_code_review_prompt(arguments),
            "debug_help": self.get_debug_help_prompt(arguments),
            "optimize_code": self.get_optimize_code_prompt(arguments)
        }
        
        if prompt_name not in prompts:
            return self.create_error_response(
                request_id, -32602, f"Unknown prompt: {prompt_name}"
            )
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "description": f"Generated prompt for {prompt_name}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": prompts[prompt_name]
                        }
                    }
                ]
            }
        }
    
    def get_code_review_prompt(self, args: Dict) -> str:
        file_path = args.get("file_path", "the provided code")
        return f"""Please perform a comprehensive code review of {file_path}. Focus on:

1. Code quality and readability
2. Potential bugs or security issues
3. Performance optimizations
4. Best practices adherence
5. Documentation and comments
6. Error handling
7. Testing considerations

Provide specific, actionable feedback with examples where possible."""
    
    def get_debug_help_prompt(self, args: Dict) -> str:
        error_message = args.get("error_message", "an unspecified error")
        code_context = args.get("code_context", "")
        
        prompt = f"""I'm encountering this error: {error_message}

Please help me debug this issue by:
1. Explaining what the error means
2. Identifying the most likely causes  
3. Providing step-by-step debugging suggestions
4. Suggesting fixes with code examples if applicable"""
        
        if code_context:
            prompt += f"\n\nHere's the relevant code context:\n```\n{code_context}\n```"
        
        return prompt
    
    def get_optimize_code_prompt(self, args: Dict) -> str:
        code = args.get("code", "")
        return f"""Please analyze this code for performance optimization opportunities:

```
{code}
```

Focus on:
1. Time complexity improvements
2. Memory usage optimization
3. Algorithm efficiency
4. Code structure improvements
5. Library/framework best practices

Provide specific suggestions with improved code examples."""
    
    def create_error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

async def main():
    """
    Main server loop that handles MCP communication over stdio.
    
    The MCP protocol uses JSON-RPC 2.0 over standard input/output.
    Each message is a JSON object on a single line.
    """
    server = MCPServer()
    
    logger.info(f"Starting {server.name} v{server.version}")
    
    try:
        while True:
            # Read a line from stdin
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON message
                message = json.loads(line)
                
                # Handle the message
                response = await server.handle_message(message)
                
                if response:
                    # Send response to stdout
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_response = server.create_error_response(
                    None, -32700, "Parse error"
                )
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                error_response = server.create_error_response(
                    None, -32603, f"Internal error: {str(e)}"
                )
                print(json.dumps(error_response), flush=True)
                
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    
    logger.info("Server shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())