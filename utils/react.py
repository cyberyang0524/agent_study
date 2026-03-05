import json
import re
from typing import List, Dict, Any, Iterable, Tuple, Callable

from config import ReActConfig
from utils.tools import Tool


def build_react_system_prompt(base_system_prompt: str, tools: List[Tool]) -> str:
    tool_markdown = "\n".join([f"- {name}: {desc}" for name, desc, _ in tools])
    guidance = f"""
You have access to the following tools:
{tool_markdown}

To use a tool, you MUST use the following format:

Thought: <your reasoning about what to do next>
Action: <tool_name> <json_arguments>

example:
Thought: The user has a headache, I should check which department treats headaches.
Action: lookup_department_by_symptom {{"text": "头痛"}}

After you receive the 'Observation', you can continue reasoning or output the final answer.

If you have enough information to answer the user, or if no tool is needed, use:
Final Answer: <your response to the user>

Rules:
1. You must start with 'Thought:'.
2. 'Action:' must be one of the available tool names.
3. 'Action:' arguments must be valid JSON, on the same line or next line.
4. Do not output 'Observation:' yourself. The system will provide it.
"""
    return base_system_prompt + "\n\n" + guidance


ACTION_PATTERN = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*(\{.*?\})", re.DOTALL)


def parse_action(text: str) -> Tuple[str, Dict[str, Any]]:
    # 1. Try standard pattern
    m = ACTION_PATTERN.search(text)
    if m:
        tool_name = m.group(1)
        try:
            args = json.loads(m.group(2).replace("'", '"')) # simple fix for single quotes
            return tool_name, args
        except:
            pass
    
    # 2. Fallback: look for tool_name {json} at the end of text if no Action: prefix
    # This helps when the model forgets "Action:"
    lines = text.strip().splitlines()
    if lines:
        last_line = lines[-1].strip()
        # Regex for "tool_name {args}"
        fallback = re.match(r"^([a-zA-Z0-9_]+)\s+(\{.*\})$", last_line)
        if fallback:
            try:
                return fallback.group(1), json.loads(fallback.group(2).replace("'", '"'))
            except:
                pass

    return "", {}


def to_stream_generator(text: str) -> Iterable[Any]:
    # Create a tiny generator that yields a single "delta" with the full text
    class Delta:
        def __init__(self, c): self.content = c
    class Choice:
        def __init__(self, d): self.delta = d
    class Chunk:
        def __init__(self, t): self.choices = [Choice(Delta(t))]
    yield Chunk(text)


def run_react(llm_client, base_system_prompt: str, seed_messages: List[Dict[str, Any]], tools: List[Tool], model: str, stream: bool) -> Any:
    system_prompt = build_react_system_prompt(base_system_prompt, tools)
    tool_map: Dict[str, Callable[[Dict[str, Any]], str]] = {name: fn for name, _, fn in tools}

    conversation: List[Dict[str, Any]] = seed_messages[:]

    for _ in range(ReActConfig.MAX_STEPS):
        step_resp = llm_client.chat_completion(conversation, system_prompt=system_prompt, model=model, stream=False, temperature=0.3)
        conversation.append({"role": "assistant", "content": step_resp})

        tool_name, args = parse_action(step_resp)
        if tool_name and tool_name in tool_map:
            try:
                observation = tool_map[tool_name](args)
            except Exception as e:
                observation = f"工具执行失败: {e}"
            conversation.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        # If we see Final Answer
        if "Final Answer:" in step_resp:
            final = step_resp.split("Final Answer:", 1)[1].strip()
            return to_stream_generator(final) if stream else final

    # Max steps reached; ask model to produce final answer
    final_prompt = conversation + [{"role": "user", "content": "请基于以上思考和观察，给出 Final Answer:"}]
    final_resp = llm_client.chat_completion(final_prompt, system_prompt=system_prompt, model=model, stream=False, temperature=0.3)
    if "Final Answer:" in final_resp:
        text = final_resp.split("Final Answer:", 1)[1].strip()
    else:
        text = final_resp
    return to_stream_generator(text) if stream else text
