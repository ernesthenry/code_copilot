"""
Simple MCP Client for testing your server
"""

import json
import subprocess
import sys

class SimpleMCPClient:
    def __init__(self, server_command):
        self.server_command = server_command
    
    def send_message(self, message):
        """Send a JSON-RPC message to the MCP server"""
        try:
            # Start the server process
            process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send the message
            stdout, stderr = process.communicate(
                input=json.dumps(message) + '\n',
                timeout=10
            )
            
            # Parse the response
            if stdout.strip():
                return json.loads(stdout.strip())
            else:
                return {"error": "No response from server", "stderr": stderr}
                
        except Exception as e:
            return {"error": str(e)}
    
    def initialize(self):
        """Initialize the MCP connection"""
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"}
        }
        return self.send_message(message)
    
    def list_tools(self):
        """List available tools"""
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        return self.send_message(message)
    
    def analyze_code(self, code):
        """Analyze code using the server"""
        message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "analyze_code",
                "arguments": {"code": code}
            }
        }
        return self.send_message(message)

def main():
    # Path to your MCP server
    server_path = input("Enter path to your mcp_server.py: ").strip()
    if not server_path:
        server_path = "./mcp_server.py"
    
    client = SimpleMCPClient(["python", server_path])
    
    print("=== MCP Client Test ===")
    
    # Test initialization
    print("\n1. Initializing...")
    response = client.initialize()
    print(json.dumps(response, indent=2))
    
    # Test listing tools
    print("\n2. Listing tools...")
    response = client.list_tools()
    if "result" in response:
        tools = response["result"].get("tools", [])
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
    
    # Test code analysis
    print("\n3. Testing code analysis...")
    test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
    response = client.analyze_code(test_code)
    if "result" in response:
        content = response["result"]["content"][0]["text"]
        analysis = json.loads(content)
        print(f"Analysis result:")
        print(f"  - Syntax valid: {analysis['syntax_valid']}")
        print(f"  - Functions: {analysis['structure_info']['functions']}")
        print(f"  - Lines of code: {analysis['structure_info']['lines_of_code']}")

if __name__ == "__main__":
    main()