import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import ModelConfig

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://open.bigmodel.cn/api/paas/v4/"):
        # Prioritize ZHIPUAI_API_KEY, then passed api_key, then OPENAI_API_KEY
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None # Default to mock if no key provided

    def chat_completion(self, messages: List[Dict[str, Any]], system_prompt: str = "", model: str = ModelConfig.DEFAULT_MODEL, stream: bool = False, temperature: float = 0.7) -> Any:
        """
        Simulates a chat completion.
        If an API key is present, it calls the actual LLM (Zhipu AI via OpenAI SDK).
        Otherwise, it uses simple keyword matching for demonstration.
        """
        if self.client:
            try:
                # Prepare messages with system prompt
                full_messages = [{"role": "system", "content": system_prompt}] + messages
                
                # Check for images in the last user message and switch model if needed
                if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), list):
                     model = ModelConfig.VISION_MODEL # Use vision model if content is a list (multimodal)

                response = self.client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    stream=stream
                )
                
                if stream:
                    return response
                else:
                    return response.choices[0].message.content
            except Exception as e:
                print(f"Error calling LLM: {e}")
                # Fallback to mock if API fails
                print("Falling back to mock response...")
                if stream:
                    # Mock stream generator
                    def mock_stream():
                        mock_text = self._mock_response(messages[-1]['content'] if messages else "", system_prompt)
                        yield type('obj', (object,), {'choices': [type('obj', (object,), {'delta': type('obj', (object,), {'content': mock_text})})]})
                    return mock_stream()
                return self._mock_response(messages[-1]['content'] if messages else "", system_prompt)

        # Mock implementation for demonstration without API key
        user_input = messages[-1]['content'] if messages else ""
        if stream:
             # Mock stream generator
             def mock_stream():
                mock_text = self._mock_response(user_input, system_prompt)
                yield type('obj', (object,), {'choices': [type('obj', (object,), {'delta': type('obj', (object,), {'content': mock_text})})]})
             return mock_stream()
        return self._mock_response(user_input, system_prompt)

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text string.
        """
        if self.client:
            try:
                response = self.client.embeddings.create(
                    model="embedding-2", # ZhipuAI embedding model
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error getting embedding: {e}")
                return [0.0] * 1024 # Mock embedding
        return [0.0] * 1024 # Mock embedding

    def _mock_response(self, user_input: str, system_prompt: str) -> str:
        """
        Simple rule-based mock response generator.
        """
        user_input_lower = user_input.lower()
        
        # Router Logic Mock
        if "Router Agent" in system_prompt:
            if any(k in user_input for k in ["头痛", "发烧", "肚子疼", "骨折", "挂号", "科室"]):
                return json.dumps({"next_agent": "triage", "reason": "User describes symptoms or asks about departments."})
            elif any(k in user_input for k in ["预约", "时间", "专家", "号"]):
                return json.dumps({"next_agent": "appointment", "reason": "User wants to make an appointment."})
            elif any(k in user_input for k in ["地址", "电话", "几点", "在哪里", "咨询"]):
                return json.dumps({"next_agent": "inquiry", "reason": "User asks general questions."})
            elif any(k in user_input for k in ["报告", "结果", "化验", "血", "高", "低"]):
                return json.dumps({"next_agent": "report", "reason": "User wants to understand test results."})
            else:
                return json.dumps({"next_agent": "triage", "reason": "Default to triage for assessment."})

        # Triage Logic Mock
        if "Triage Agent" in system_prompt:
            if "头痛" in user_input:
                return "建议您挂神经内科。如果您伴有发热，请先去发热门诊。"
            elif "骨折" in user_input or "摔" in user_input:
                return "建议您挂骨科。请尽量固定受伤部位，避免二次伤害。"
            elif "肚子" in user_input:
                return "建议您挂消化内科。如果是剧烈疼痛，请立即前往急诊。"
            else:
                return "根据您的描述，建议先挂全科门诊进行初步检查。"

        # Appointment Logic Mock
        if "Appointment Agent" in system_prompt:
            if "王医生" in user_input:
                return "好的，为您查询王医生的出诊时间。王医生周三上午有号，需要帮您预约吗？"
            elif "明天" in user_input:
                return "明天的号源充足，请问您需要上午还是下午？"
            elif "预约" in user_input:
                return "请提供您想要预约的科室、医生姓名或日期。"
            else:
                return "好的，已为您记录预约需求。"

        # Inquiry Logic Mock
        if "Inquiry Agent" in system_prompt:
            if "地址" in user_input or "在哪" in user_input:
                return "本院位于北京市朝阳区演示路123号。"
            elif "时间" in user_input:
                return "门诊时间为周一至周日 8:00 - 17:00，急诊24小时开放。"
            else:
                return "您好，我是医院咨询助手，请问有什么可以帮您？"

        # Report Logic Mock
        if "Report Agent" in system_prompt:
            return f"我是报告解读助手。您提到的数值（{user_input}）可能表明身体有炎症或其他情况。建议您携带报告单咨询主治医生以获得准确诊断。"

        return "I am a mock agent and I don't understand."
