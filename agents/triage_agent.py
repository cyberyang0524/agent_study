from typing import Any
from .base_agent import BaseAgent
from utils.llm_client import LLMClient
from utils.tools import triage_tools

from config import ModelConfig, ReActConfig

class TriageAgent(BaseAgent):
    def __init__(self):
        super().__init__("Triage Agent", "Analyzes symptoms and suggests departments.", model=ModelConfig.TRIAGE_AGENT_MODEL)
        self.llm_client = LLMClient()
        self.system_prompt = """
        You are a Triage Agent for a hospital.
        Your goal is to listen to the user's symptoms and suggest the appropriate medical department.
        You can also help users find information about specific doctors.
        Be professional, empathetic, and concise.
        If the symptoms seem severe (e.g., chest pain, difficulty breathing), advise them to go to the Emergency Room immediately.
        """
        self.use_react = ReActConfig.TRIAGE_USE_REACT

    def get_tools(self):
        return triage_tools()

    def process(self, user_input: str, context: dict = None, stream: bool = False, images: list = None) -> Any:
        react_result = self._maybe_run_react(user_input, stream=stream, images=images)
        if react_result is not None:
            return react_result
        messages = self.get_memory() + [{"role": "user", "content": user_input}]
        return self.llm_client.chat_completion(messages, system_prompt=self.system_prompt, stream=stream, model=self.model)
