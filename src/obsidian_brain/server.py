import asyncio
from datetime import datetime
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

from .enhanced_mind import EnhancedMind
import os

# Configure logging with a format that includes timestamps and log levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ObsidianBrainServer:
    def __init__(self, mind_path: str):
        self.server = Server("obsidian-brain")
        self.mind = EnhancedMind(mind_path=mind_path)
        
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        self._register_handlers()
        
        logger.info(f"Initialized ObsidianBrainServer with mind path: {mind_path}")

    def _format_conversation(
        self,
        conversation: List[Dict[str, Any]],
        additional_notes: Optional[str] = None
    ) -> str:
        formatted = "## Conversation Summary\n\n"
        

        formatted += "### Metadata\n"
        formatted += f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        formatted += f"- Messages: {len(conversation)}\n\n"
        
        formatted += "### Discussion\n\n"
        for msg in conversation:
            timestamp = msg.get('timestamp', 'Unknown time')
            role = msg.get('role', 'Unknown')
            content = msg.get('content', '')
            
            formatted += f"**{timestamp} - {role}**\n\n"
            formatted += f"{content}\n\n"
            formatted += "---\n\n"
        
        if additional_notes:
            formatted += "### Additional Notes\n\n"
            formatted += f"{additional_notes}\n\n"
        
        return formatted

    def _register_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="fetch-memories",
                    description="Search Obsidian vault using hybrid search (combines keyword and semantic search)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "default": 10},
                            "vector_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.5,
                                "description": "Weight given to vector search results vs keyword search - vector results suck right now"
                            }
                        },
                        "required": ["query"],
                    },
                ),
                types.Tool(
                    name="add-note",
                    description="Add a new note to Obsidian vault",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": []
                            },
                            "folder": {
                                "type": "string",
                                "description": "Optional subfolder path within vault",
                            }
                        },
                        "required": ["title", "content"],
                    },
                ),
                types.Tool(
                    name="save-conversation",
                    description="Save the current conversation as a note in Obsidian",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conversation_id": {"type": "string"},
                            "title": {"type": "string"},
                            "additional_notes": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": ["conversation", "claude"]
                            },
                            "folder": {
                                "type": "string",
                                "default": "conversations",
                                "description": "Subfolder to store conversation in"
                            }
                        },
                        "required": ["conversation_id", "title"],
                    },
                ),
                types.Tool(
                    name="track-conversation",
                    description="Track a message in the current conversation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conversation_id": {"type": "string"},
                            "role": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["conversation_id", "role", "content"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, 
            arguments: dict | None
        ) -> list[types.TextContent]:
            try:
                match name:
                    case "fetch-memories":
                        return await self.fetch_memories_handler(arguments)
                    case "add-note":
                        return await self.add_note_handler(arguments)
                    case "track-conversation":
                        return await self.track_conversation_handler(arguments)
                    case "save-conversation":
                        return await self.save_conversation_handler(arguments)
                    case _:
                        raise ValueError(f"Unknown tool: {name}")                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [types.TextContent(
                    type="text",
                    text=f"Error executing tool {name}: {str(e)}"
                )]

    async def fetch_memories_handler(self, arguments: dict | None) -> list[types.TextContent]:
        results = await self.mind.hybrid_search(
            query=arguments.get("query"),
            k=arguments.get("max_results", 10),
            vector_weight=arguments.get("vector_weight", 0.5)
        )
        formatted_results = []
        for result in results:
            formatted_results.append(
                f"Source: {result.source.upper()} (Score: {result.score:.2f})\n"
                f"Title: {result.metadata.get('title', 'Untitled')}\n"
                f"Path: {result.metadata.get('path', 'Unknown')}\n"
                f"Last Modified: {result.metadata.get('last_modified', 'Unknown')}\n"
                f"Content:\n{result.content}\n"
                f"{'=' * 50}\n"
            )

        return [types.TextContent(
            type="text",
            text="\n".join(formatted_results)
        )]

    async def add_note_handler(self, arguments: dict | None) -> list[types.TextContent]:
        filepath = await self.mind.add_note(
            title=arguments.get("title"),
            content=arguments.get("content"),
            tags=arguments.get("tags", []),
            folder=arguments.get("folder")
        )
        
        return [types.TextContent(
            type="text",
            text=f"Note successfully added at: {filepath}"
        )]
    
    async def track_conversation_handler(self, arguments: dict | None) -> list[types.TextContent]:
        conversation_id = arguments.get("conversation_id")
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # add message to conversation history with timestamp
        self.conversations[conversation_id].append({
            "timestamp": datetime.now().isoformat(),
            "role": arguments.get("role"),
            "content": arguments.get("content")
        })
        
        return [types.TextContent(
            type="text",
            text=f"Message tracked in conversation: {conversation_id}"
        )]
    
    async def save_conversation_handler(self, arguments: dict | None) -> list[types.TextContent]:
        conversation_id = arguments.get("conversation_id")
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        content = self._format_conversation(
            self.conversations[conversation_id],
            arguments.get("additional_notes")
        )
        
        filepath = await self.mind.add_note(
            title=arguments.get("title"),
            content=content,
            tags=arguments.get("tags", ["conversation", "claude"]),
            folder=arguments.get("folder", "conversations")
        )
        
        return [types.TextContent(
            type="text",
            text=f"Conversation saved as note: {filepath}"
        )]

    async def run(self):
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            try:
                init_task = asyncio.create_task(self.mind.process_files())
                watcher_task = asyncio.create_task(self.mind.init_watcher())
                
                server_task = asyncio.create_task(self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="obsidian-brain",
                        server_version="0.2.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                ))
                
                await asyncio.gather(init_task, watcher_task, server_task)
                
            except Exception as e:
                logger.error(f"Server error: {e}")
                raise
            finally:
                logger.info("Server shutting down")

async def main():
    mind_path = os.getenv("OBSIDIAN_MIND_PATH")
    if not mind_path:
        raise ValueError("OBSIDIAN_MIND_PATH environment variable not set")
    try:
        server = ObsidianBrainServer(mind_path)
        await server.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
