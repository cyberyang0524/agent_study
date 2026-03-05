import json
from typing import Any
from .base_agent import BaseAgent
from utils.llm_client import LLMClient

from config import ModelConfig

class RouterAgent(BaseAgent):
    def __init__(self):
        super().__init__("Router Agent", "Analyzes user intent and routes to specific agents.", model=ModelConfig.ROUTER_AGENT_MODEL)
        self.llm_client = LLMClient()
        self.system_prompt = """
        You are a Router Agent for a hospital customer service system.
        Your job is to analyze the user's input and determine which specialized agent should handle the request.
        
        Available Agents:
        - triage: Handles symptoms analysis, suggests departments, and FINDS DOCTORS. Keywords: pain, symptoms, fever, checkup, not feeling well, body parts, who is this doctor, what is this doctor good at.
        - appointment: Handles scheduling appointments with doctors. Keywords: book, schedule, time, doctor, appointment, register.
        - inquiry: Handles general hospital information. Keywords: address, phone, hours, location, parking, contact.
        - report: Handles explanation of medical test results and reports. Keywords: report, test result, blood test, xray, ct, mri, values, high, low.
        
        Strictly Output JSON ONLY. No markdown, no conversational text.
        
        Return a JSON object with the following format:
        {
            "next_agent": "agent_name",
            "reason": "reason for routing"
        }
        
        If the user input is ambiguous or conversational (e.g., "Hello", "Thank you"), route to 'triage' so they can guide the user.
        """

    def process(self, user_input: str, context: dict = None, stream: bool = False, images: list = None) -> Any:
        # Construct messages for LLM
        messages = [{"role": "user", "content": user_input}]
        
        # Router always uses non-streaming to get full JSON
        response = self.llm_client.chat_completion(messages, system_prompt=self.system_prompt, stream=False, model=self.model, temperature=0.1)
        
        # Helper function to clean markdown code blocks
        def clean_json_string(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                first_newline = s.find("\n")
                if first_newline != -1:
                    last_backticks = s.rfind("```")
                    if last_backticks != -1:
                        s = s[first_newline+1:last_backticks].strip()
            return s

        try:
            # Check if response is already a dict (from mock)
            if isinstance(response, dict):
                 return response
                 
            # If string, clean and parse
            cleaned_response = clean_json_string(response)
            return json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            # First attempt failed, try ONE retry with stronger instruction
            print(f"JSON Parse Error: {e}. Retrying...")
            retry_messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": "You returned invalid JSON. Please return ONLY the JSON object. No other text."}
            ]
            retry_response = self.llm_client.chat_completion(retry_messages, system_prompt=self.system_prompt, stream=False, temperature=0.1)
            
            try:
                if isinstance(retry_response, dict):
                    return retry_response
                cleaned_retry = clean_json_string(retry_response)
                return json.loads(cleaned_retry)
            except json.JSONDecodeError as e2:
                 # Fallback if retry also fails
                print(f"Retry JSON Parse Error: {e2}")
                print(f"Original Response: {response}")
                return {"next_agent": "triage", "reason": f"Failed to parse router decision after retry. Defaulting to triage."}
