from typing import Any
from .base_agent import BaseAgent
from utils.llm_client import LLMClient
from utils.tools import report_tools

from config import ModelConfig, ReActConfig

class ReportAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Report Agent", "Analyzes medical test reports and explains results.", model=ModelConfig.REPORT_AGENT_MODEL) # Default to configured model, will upgrade to vision model if images present
        self.llm_client = LLMClient()
        self.system_prompt = """
        You are a Report Analysis Agent for a hospital.
        Your goal is to explain medical test reports to patients in simple, easy-to-understand language.
        
        Guidelines:
        1. Explain what the test measures.
        2. Interpret the results (high, low, normal) if provided.
        3. Explain potential causes for abnormal results, but emphasize that this is not a diagnosis.
        4. Advise the patient to consult their doctor for a definitive diagnosis and treatment plan.
        5. Be reassuring and avoid causing unnecessary panic.
        
        Example:
        User: "My WBC count is 12.5."
        Agent: "WBC stands for White Blood Cells, which fight infection. A count of 12.5 is slightly high (normal is usually 4-11). This often indicates the body is fighting an infection or inflammation. It's usually not serious, but please show this to your doctor."
        """
        self.use_react = ReActConfig.REPORT_USE_REACT

    def get_tools(self):
        return report_tools()

    def process(self, user_input: str, context: dict = None, stream: bool = False, images: list = None) -> Any:
        react_result = self._maybe_run_react(user_input, stream=stream, images=images)
        if react_result is not None:
            return react_result

        current_model = self.model
        if images:
            # Multi-modal input (GLM-4V style)
            current_model = "glm-4v-plus" # Force upgrade to vision model
            # Text content
            content = [{"type": "text", "text": user_input}]
            # Image content
            for img_url in images:
                 content.append({
                     "type": "image_url",
                     "image_url": {
                         "url": img_url
                     }
                 })
            current_message = {"role": "user", "content": content}
        else:
            # Text-only input
            current_message = {"role": "user", "content": user_input}
            
        messages = self.get_memory() + [current_message]
        return self.llm_client.chat_completion(messages, system_prompt=self.system_prompt, stream=stream, model=current_model)
