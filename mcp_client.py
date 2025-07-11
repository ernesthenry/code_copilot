from dotenv import load_dotenv
import requests
import json
import asyncio
import logging
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any
from contextlib import AsyncExitStack
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class ResourceDefinition(TypedDict):
    uri: str
    name: str
    description: str
    mimeType: str

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:latest"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api/chat"
    
    def create_message(self, messages: List[Dict], tools: List[ToolDefinition] = None, stream: bool = False):
        """Create a message using Ollama API with tool calling support. If stream=True, yield chunks as they arrive."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        # Add tools if provided - convert to Ollama format
        if tools:
            payload["tools"] = self._convert_tools_to_ollama_format(tools)
        
        if stream:
            with requests.post(self.api_url, json=payload, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        yield line.decode()
        else:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
    
    def _convert_tools_to_ollama_format(self, tools: List[ToolDefinition]) -> List[Dict]:
        """Convert MCP tools to Ollama tool format."""
        ollama_tools = []
        for tool in tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            ollama_tools.append(ollama_tool)
        return ollama_tools

class MCP_ChatBot:

    def __init__(self, model: str = "qwen2.5:latest"):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.ollama = OllamaClient(model=model)
        self.available_tools: List[ToolDefinition] = []
        self.available_resources: List[ResourceDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.resource_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)
            
            # List available tools for this session
            try:
                response = await session.list_tools()
                tools = response.tools
                logger.info(f"Connected to {server_name} with tools: {[t.name for t in tools]}")
                
                for tool in tools:
                    self.tool_to_session[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
            except Exception as e:
                logger.warning(f"Could not list tools for {server_name}: {e}")
            
            # List available resources for this session
            try:
                resources_response = await session.list_resources()
                resources = resources_response.resources
                logger.info(f"Available resources from {server_name}: {[r.name for r in resources]}")
                
                for resource in resources:
                    self.resource_to_session[resource.uri] = session
                    self.available_resources.append({
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType
                    })
            except Exception as e:
                logger.warning(f"Could not list resources for {server_name}: {e}")
                
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            raise

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
                
            logger.info(f"Total tools available: {len(self.available_tools)}")
            logger.info(f"Total resources available: {len(self.available_resources)}")
            
        except Exception as e:
            logger.error(f"Error loading server configuration: {e}")
            raise

    async def get_resource_content(self, resource_uri: str) -> str:
        """Get content from a resource."""
        try:
            session = self.resource_to_session.get(resource_uri)
            if not session:
                return f"Resource not found: {resource_uri}"
            
            result = await session.read_resource(resource_uri)
            if result.contents:
                # Handle different content types
                content = result.contents[0]
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'blob'):
                    return f"Binary content ({len(content.blob)} bytes)"
                else:
                    return str(content)
            return "No content available"
            
        except Exception as e:
            logger.error(f"Error reading resource {resource_uri}: {e}")
            return f"Error reading resource: {str(e)}"

    def _parse_tool_calls_from_response(self, response_text: str) -> List[Dict]:
        """Parse tool calls from Ollama response text."""
        tool_calls = []
        
        # Look for tool call patterns in the response
        # This is a simplified parser - you might need to adjust based on your model's output format
        lines = response_text.split('\n')
        current_tool = None
        
        for line in lines:
            line = line.strip()
            
            # Look for tool call indicators
            if line.startswith('🔧') or 'tool:' in line.lower() or 'calling' in line.lower():
                # Try to extract tool name
                for tool in self.available_tools:
                    if tool['name'].lower() in line.lower():
                        current_tool = tool['name']
                        break
            
            # Look for JSON-like arguments
            if current_tool and (line.startswith('{') or 'args:' in line.lower()):
                try:
                    # Try to parse JSON arguments
                    if line.startswith('{'):
                        args = json.loads(line)
                    else:
                        # Extract JSON from the line
                        json_start = line.find('{')
                        if json_start != -1:
                            args = json.loads(line[json_start:])
                        else:
                            args = {}
                    
                    tool_calls.append({
                        'name': current_tool,
                        'arguments': args,
                        'id': f"tool_{len(tool_calls)}"
                    })
                    current_tool = None
                except json.JSONDecodeError:
                    continue
        
        return tool_calls

    async def process_query(self, query: str):
        """Process a user query with tool calling support."""
        messages = [{'role': 'user', 'content': query}]
        
        # Add system message to help with tool calling
        system_message = {
            'role': 'system', 
            'content': f'''You are an AI assistant with access to various tools. When you need to use a tool, clearly state:
1. The tool name
2. The arguments as JSON

Available tools:
{json.dumps([{"name": t["name"], "description": t["description"]} for t in self.available_tools], indent=2)}

When calling a tool, format your response like:
🔧 Calling tool: TOOL_NAME
📋 Arguments: {{"arg1": "value1", "arg2": "value2"}}

Be helpful and use tools when appropriate to answer the user's question.'''
        }
        messages.insert(0, system_message)
        
        try:
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                print("🟡 Waiting for LLM response...", flush=True)
                # Stream response from Ollama
                response_chunks = []
                response_text = ""
                for chunk in self.ollama.create_message(messages, self.available_tools, stream=True):
                    print(chunk, end='', flush=True)
                    response_chunks.append(chunk)
                print()  # Newline after streaming
                response_text = ''.join(response_chunks)
                
                # Check if the response contains tool calls
                tool_calls = self._parse_tool_calls_from_response(response_text)
                
                if not tool_calls:
                    # No tool calls, this is the final response
                    break
                
                # Add assistant message
                messages.append({'role': 'assistant', 'content': response_text})
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['arguments']
                    tool_id = tool_call['id']
                    
                    print(f"🔧 Running tool: {tool_name}...", flush=True)
                    print(f"📋 Arguments: {json.dumps(tool_args, indent=2)}")
                    
                    try:
                        session = self.tool_to_session.get(tool_name)
                        if not session:
                            raise Exception(f"Tool '{tool_name}' not found")
                        
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        # Handle the result
                        if result.content:
                            if isinstance(result.content, list):
                                tool_result_content = []
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        tool_result_content.append(item.text)
                                    else:
                                        tool_result_content.append(str(item))
                                result_text = '\n'.join(tool_result_content)
                            else:
                                if hasattr(result.content, 'text'):
                                    result_text = result.content.text
                                else:
                                    result_text = str(result.content)
                        else:
                            result_text = "Tool executed successfully (no output)"
                        
                        print(f"✅ Tool result: {result_text[:200]}{'...' if len(result_text) > 200 else ''}")
                        tool_results.append(f"Tool {tool_name} result: {result_text}")
                        
                    except Exception as e:
                        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                        logger.error(error_msg)
                        print(f"❌ {error_msg}")
                        tool_results.append(f"Tool {tool_name} error: {error_msg}")
                
                # Add tool results to messages
                if tool_results:
                    tool_result_message = "Tool execution results:\n" + "\n".join(tool_results)
                    messages.append({'role': 'user', 'content': tool_result_message})
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error processing query: {str(e)}")

    async def list_available_tools(self):
        """List all available tools from connected servers."""
        print("\n📋 Available Tools:")
        print("=" * 50)
        
        for tool in self.available_tools:
            print(f"\n🔧 {tool['name']}")
            print(f"   {tool['description']}")
            
            # Show input schema if available
            if tool['input_schema'].get('properties'):
                print("   Parameters:")
                for param, details in tool['input_schema']['properties'].items():
                    param_type = details.get('type', 'unknown')
                    description = details.get('description', 'No description')
                    print(f"     • {param} ({param_type}): {description}")

    async def list_available_resources(self):
        """List all available resources from connected servers."""
        print("\n📁 Available Resources:")
        print("=" * 50)
        
        for resource in self.available_resources:
            print(f"\n📄 {resource['name']}")
            print(f"   URI: {resource['uri']}")
            print(f"   Type: {resource['mimeType']}")
            print(f"   Description: {resource['description']}")

    async def chat_loop(self):
        """Run an interactive chat loop with enhanced commands."""
        print(f"\n🤖 MCP Chatbot with Ollama ({self.ollama.model}) Started!")
        print("=" * 50)
        print("Commands:")
        print("  • Type your queries normally")
        print("  • 'tools' - List available tools")
        print("  • 'resources' - List available resources")
        print("  • 'quit' - Exit the chatbot")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n💬 Query: ").strip()
        
                if query.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'tools':
                    await self.list_available_tools()
                elif query.lower() == 'resources':
                    await self.list_available_resources()
                elif query:
                    await self.process_query(query)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"❌ Error: {str(e)}")

    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        logger.info("Cleaning up resources...")
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot(model="qwen2.5:latest")
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
