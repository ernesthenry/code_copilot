from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any
from contextlib import AsyncExitStack
import json
import asyncio
import logging

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

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
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

    async def process_query(self, query: str):
        """Process a user query with tool calling support."""
        messages = [{'role': 'user', 'content': query}]
        
        try:
            response = self.anthropic.messages.create(
                max_tokens=4096,
                model='claude-3-5-sonnet-20241022',  # Updated model name
                tools=self.available_tools,
                messages=messages
            )
            
            process_query = True
            while process_query:
                assistant_content = []
                
                for content in response.content:
                    if content.type == 'text':
                        print(content.text)
                        assistant_content.append(content)
                        if len(response.content) == 1:
                            process_query = False
                            
                    elif content.type == 'tool_use':
                        assistant_content.append(content)
                        
                        # Add assistant message with tool use
                        messages.append({'role': 'assistant', 'content': assistant_content})
                        
                        tool_id = content.id
                        tool_args = content.input
                        tool_name = content.name
                        
                        print(f"🔧 Calling tool: {tool_name}")
                        print(f"📋 Arguments: {json.dumps(tool_args, indent=2)}")
                        
                        # Execute the tool
                        try:
                            session = self.tool_to_session.get(tool_name)
                            if not session:
                                raise Exception(f"Tool '{tool_name}' not found")
                            
                            result = await session.call_tool(tool_name, arguments=tool_args)
                            
                            # Handle the result based on its type
                            if result.content:
                                if isinstance(result.content, list):
                                    # Handle multiple content items
                                    tool_result_content = []
                                    for item in result.content:
                                        if hasattr(item, 'text'):
                                            tool_result_content.append(item.text)
                                        else:
                                            tool_result_content.append(str(item))
                                    result_text = '\n'.join(tool_result_content)
                                else:
                                    # Handle single content item
                                    if hasattr(result.content, 'text'):
                                        result_text = result.content.text
                                    else:
                                        result_text = str(result.content)
                            else:
                                result_text = "Tool executed successfully (no output)"
                            
                            print(f"✅ Tool result: {result_text[:200]}{'...' if len(result_text) > 200 else ''}")
                            
                        except Exception as e:
                            result_text = f"Error executing tool '{tool_name}': {str(e)}"
                            logger.error(result_text)
                            print(f"❌ {result_text}")
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "user", 
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result_text
                                }
                            ]
                        })
                        
                        # Get next response from Claude
                        try:
                            response = self.anthropic.messages.create(
                                max_tokens=4096,
                                model='claude-3-5-sonnet-20241022',
                                tools=self.available_tools,
                                messages=messages
                            )
                            
                            # Check if this is the final response
                            if (len(response.content) == 1 and 
                                response.content[0].type == "text"):
                                print(response.content[0].text)
                                process_query = False
                                
                        except Exception as e:
                            logger.error(f"Error getting response from Claude: {e}")
                            print(f"❌ Error getting response: {str(e)}")
                            process_query = False
                            
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
        print("\n🤖 MCP Chatbot Started!")
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
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())