from typing import Any
from .base_agent import BaseAgent
from utils.llm_client import LLMClient
from utils.tools import appointment_tools

from config import ModelConfig, ReActConfig

class AppointmentAgent(BaseAgent):
    def __init__(self):
        super().__init__("Appointment Agent", "Handles appointment scheduling.", model=ModelConfig.APPOINTMENT_AGENT_MODEL)
        self.llm_client = LLMClient()
        self.system_prompt = """
        You are an Appointment Agent for a hospital.
        Your goal is to help users schedule appointments with doctors.
        Ask for necessary details like:
        1. Department
        2. Doctor's name (optional)
        3. Preferred date and time
        
        Once you have enough information, confirm the appointment details with the user.
        """
        self.use_react = ReActConfig.APPOINTMENT_USE_REACT

    def get_tools(self):
        return appointment_tools()

    def process(self, user_input: str, context: dict = None, stream: bool = False, images: list = None) -> Any:
        react_result = self._maybe_run_react(user_input, stream=stream, images=images)
        if react_result is not None:
            return react_result
        messages = self.get_memory() + [{"role": "user", "content": user_input}]
        return self.llm_client.chat_completion(messages, system_prompt=self.system_prompt, stream=stream, model=self.model)
