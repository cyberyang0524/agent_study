from abc import ABC, abstractmethod
from typing import Dict, Any, List

from config import ReActConfig
from utils.react import run_react
from utils.tools import empty_tools

class BaseAgent(ABC):
    def __init__(self, name: str, description: str, memory_limit: int = 40, model: str = "glm-4-flash"):
        self.name = name
        self.description = description
        self.memory: List[Dict[str, str]] = []
        self.memory_limit = memory_limit # Limit number of messages (e.g., 20 turns = 40 messages)
        self.model = model
        self.use_react = False

    def update_memory(self, role: str, content: str):
        """Updates the agent's memory with a new message, keeping only the recent history."""
        self.memory.append({"role": role, "content": content})
        
        # FIFO Truncation: Keep only the last N messages
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]

    def get_memory(self) -> List[Dict[str, str]]:
        """Returns the current memory (conversation history)."""
        return self.memory

    def clear_memory(self):
        """Clears the agent's memory."""
        self.memory = []

    # Tool hook (override in subclasses)
    def get_tools(self):
        return empty_tools()

    # Optional ReAct handler for subclasses to call
    def _maybe_run_react(self, user_input: str, stream: bool = False, images: List[str] = None):
        if not self.use_react or not ReActConfig.ENABLE:
            return None
        current_message = {"role": "user", "content": user_input} if not images else [
            {"type": "text", "text": user_input},
            *[{"type": "image_url", "image_url": {"url": u}} for u in images]
        ]
        if images:
            current_message = {"role": "user", "content": current_message}
        messages = self.get_memory() + [current_message]
        tools = self.get_tools()
        return run_react(None if hasattr(self, "llm_client") is False else self.llm_client, self.system_prompt, messages, tools, self.model, stream)

    @abstractmethod
    def process(self, user_input: str, context: Dict[str, Any] = None, stream: bool = False, images: List[str] = None) -> Any:
        """
        Process the user input and return a response.
        If stream is True, returns a generator/iterator.
        If images is provided, it contains a list of image paths/URLs.
        """
        pass
