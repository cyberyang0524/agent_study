from typing import Any
from .base_agent import BaseAgent
from utils.llm_client import LLMClient
from utils.tools import inquiry_tools

from config import ModelConfig, ReActConfig

class InquiryAgent(BaseAgent):
    def __init__(self):
        super().__init__("Inquiry Agent", "Handles general hospital inquiries.", model=ModelConfig.INQUIRY_AGENT_MODEL)
        self.llm_client = LLMClient()
        self.system_prompt = """
        You are an Inquiry Agent for a hospital.
        Your goal is to answer general questions about the hospital, such as:
        - Location and address
        - Visiting hours
        - Contact information
        - Parking info
        
        Provide accurate and helpful information.
        """
        self.use_react = ReActConfig.INQUIRY_USE_REACT

    def get_tools(self):
        return inquiry_tools()

    def process(self, user_input: str, context: dict = None, stream: bool = False, images: list = None) -> Any:
        react_result = self._maybe_run_react(user_input, stream=stream, images=images)
        if react_result is not None:
            return react_result
        messages = self.get_memory() + [{"role": "user", "content": user_input}]
        return self.llm_client.chat_completion(messages, system_prompt=self.system_prompt, stream=stream, model=self.model)
