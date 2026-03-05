from typing import Callable, Dict, Any, List, Tuple
import datetime
import os
import re

# Tool registry type
Tool = Tuple[str, str, Callable[[Dict[str, Any]], str]]


# ---------- Shared data (mock) ----------
DOC_FILE = r"d:\agent\doc.txt"
DEPT_FILE = r"d:\agent\dept.txt"

HOSPITAL_INFO = {
    "name": "示例医院",
    "address": "北京市朝阳区演示路123号",
    "phone": "010-12345678",
    "parking": "地上+地下停车场，首小时免费",
}

OPENING_HOURS = "门诊时间：周一至周日 8:00 - 17:00；急诊24小时"

SCHEDULE_MOCK = {
    "王医生": {
        "2026-03-06": "上午有号（8:30-11:30）",
        "2026-03-07": "下午有号（13:30-16:30）",
    }
}



# ---------- Tool implementations ----------
_DOCTORS_CACHE: List[Dict[str, str]] = None
_DEPARTMENTS_CACHE: Dict[str, Dict[str, Any]] = None


def _load_doctors() -> List[Dict[str, str]]:
    global _DOCTORS_CACHE
    if _DOCTORS_CACHE is not None:
        return _DOCTORS_CACHE
    data: List[Dict[str, str]] = []
    if os.path.exists(DOC_FILE):
        with open(DOC_FILE, "r", encoding="utf-8") as f:
            current_doc = None
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Heuristic: A line with at least 2 pipes (Name|Dept|Expertise) is a new record
                if line.count("|") >= 2:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        current_doc = {
                            "name": parts[0],
                            "dept": parts[1],
                            "expertise": "|".join(parts[2:])
                        }
                        data.append(current_doc)
                elif current_doc is not None:
                    # Continuation of the previous doctor's expertise
                    current_doc["expertise"] += " " + line
    _DOCTORS_CACHE = data
    return data


def _load_departments() -> Dict[str, Dict[str, Any]]:
    global _DEPARTMENTS_CACHE
    if _DEPARTMENTS_CACHE is not None:
        return _DEPARTMENTS_CACHE
    depts: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(DEPT_FILE):
        current: str = ""
        with open(DEPT_FILE, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                
                # Check for address line
                if "地址：" in line or "地址:" in line:
                    # Extract address part
                    # Handle " - 地址：" or "地址："
                    if "地址：" in line:
                        addr = line.split("地址：", 1)[1].strip()
                    else:
                        addr = line.split("地址:", 1)[1].strip()
                    
                    if current:
                        if depts[current]["address"]:
                            depts[current]["address"] += "；" + addr
                        else:
                            depts[current]["address"] = addr
                    continue

                # Department header: ends with colon (Chinese or English)
                if line.endswith("：") or line.endswith(":"):
                    current = line[:-1].strip()
                    depts[current] = {"desc": [], "address": None}
                    continue
                
                # Bullet description or continuation
                if line.startswith("-") or line.startswith("•"):
                    clean_line = line.lstrip("-• ").strip()
                    if current:
                        depts[current]["desc"].append(clean_line)
                else:
                    # Continuation of description or implicit header?
                    # Treat as description for now
                    if current:
                        depts[current]["desc"].append(line)

    _DEPARTMENTS_CACHE = depts
    return depts


def tool_get_hospital_info(_: Dict[str, Any]) -> str:
    try:
        kb = get_knowledge_base()
        chunks = kb.search("医院地址电话停车")
        if not chunks:
            return "暂无医院基础信息"
        return "检索结果：\n" + "\n".join(chunks)
    except Exception as e:
        return f"查询失败: {e}"

def tool_get_opening_hours(_: Dict[str, Any]) -> str:
    try:
        kb = get_knowledge_base()
        chunks = kb.search("门诊时间")
        if not chunks:
            return "暂无门诊时间信息"
        return "检索结果：\n" + "\n".join(chunks)
    except Exception as e:
        return f"查询失败: {e}"


from utils.llm_client import LLMClient

def tool_lookup_department_by_symptom(args: Dict[str, Any]) -> str:
    text = (args.get("text") or "").strip()
    if not text:
        return "请提供症状描述"

    depts = _load_departments()
    dept_names = list(depts.keys())
    
    # Use LLM to decide which department is best
    # This avoids hardcoded mapping and allows "reasoning" based on department descriptions
    
    # 1. Prepare context: list of departments and their brief descriptions (if any)
    # To save tokens, we might just list names or name + first sentence of desc
    dept_context = []
    for name, info in depts.items():
        desc = info.get("desc", [])
        short_desc = desc[0] if desc else "暂无简介"
        dept_context.append(f"- {name}: {short_desc}")
    
    context_str = "\n".join(dept_context)
    
    system_prompt = f"""
    You are a medical triage assistant.
    The user will provide a symptom description.
    You must select the MOST suitable department from the following list:
    
    {context_str}
    
    Rules:
    1. Only return the name of the department.
    2. If multiple departments fit, choose the most specific one.
    3. If no department fits well, return "全科医学科".
    4. Do not output any other text, just the department name.
    """
    
    try:
        client = LLMClient()
        response = client.chat_completion(
            messages=[{"role": "user", "content": f"Symptom: {text}"}],
            system_prompt=system_prompt,
            temperature=0.1
        )
        
        suggested_dept = str(response).strip()
        
        # Verify if the returned department exists
        # Allow fuzzy matching if exact match fails
        if suggested_dept not in depts:
            # Try to find partial match
            for d in depts:
                if d in suggested_dept or suggested_dept in d:
                    suggested_dept = d
                    break
        
        if suggested_dept in depts:
            info = depts[suggested_dept]
            addr = info.get("address") or "地址信息未收录"
            return f"建议科室：{suggested_dept}；地址：{addr}"
        else:
             # Fallback if LLM hallucinates a department not in list
             return f"建议科室：全科医学科（模型建议：{suggested_dept}，但未在目录中找到）"
             
    except Exception as e:
        print(f"LLM triage failed: {e}")
        return "建议科室：全科医学科（系统繁忙，请前往分诊台咨询）"


def tool_check_emergency_signs(args: Dict[str, Any]) -> str:
    text = (args.get("text") or "").lower()
    if any(k in text for k in ["剧烈", "呼吸困难", "意识不清", "突然加重", "胸痛"]):
        return "存在急症征象，建议立即前往急诊。"
    return "未发现明显急症征象。"


def tool_query_schedule(args: Dict[str, Any]) -> str:
    doctor = args.get("doctor_name") or "王医生"
    date = args.get("date")
    if not date or date in ["今天", "明天"]:
        base = datetime.date.today()
        if date == "明天":
            base = base + datetime.timedelta(days=1)
        date = base.strftime("%Y-%m-%d")
    info = SCHEDULE_MOCK.get(doctor, {}).get(date)
    return info or f"{doctor} {date} 暂无号源信息"


def tool_get_lab_reference_range(args: Dict[str, Any]) -> str:
    test = (args.get("test_name") or "").upper()
    try:
        kb = get_knowledge_base()
        chunks = kb.search(f"{test} 参考范围")
        if not chunks:
            return f"{test} 的参考区间暂无数据"
        return "检索结果：\n" + "\n".join(chunks)
    except Exception as e:
        return f"查询失败: {e}"

def tool_explain_indicator(args: Dict[str, Any]) -> str:
    test = (args.get("test_name") or "").upper()
    value = args.get("value")
    
    # Use RAG to get reference info first
    kb = get_knowledge_base()
    chunks = kb.search(f"{test} 参考范围")
    ref_info = "\n".join(chunks) if chunks else "暂无参考范围信息"

    system_prompt = f"""
    You are a medical lab assistant.
    Based on the following reference information:
    {ref_info}
    
    Analyze the test result: {test} = {value}
    
    Output format:
    {test} 当前值：{value}，判断：[偏高/偏低/正常]。该解读仅供参考，请结合临床。
    """
    
    try:
        client = LLMClient()
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Analyze result"}],
            system_prompt=system_prompt,
            temperature=0.1
        )
        return str(response)
    except Exception as e:
        return f"分析失败: {e}"


# ---------- Tool set providers ----------
def tool_find_doctor(args: Dict[str, Any]) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "缺少 query 参数"
    
    docs = _load_doctors()
    
    # 1. Prepare doctor list context
    # Format: Name | Dept | Expertise
    doc_context = []
    for d in docs:
        doc_context.append(f"{d['name']} | {d['dept']} | {d['expertise']}")
    
    context_str = "\n".join(doc_context)
    
    # 2. Use LLM to match the best doctor
    system_prompt = f"""
    You are a medical receptionist.
    The user is looking for a doctor based on a query (symptom, disease, or doctor name).
    You have the following list of doctors:
    
    {context_str}
    
    Rules:
    1. Select up to 3 doctors who are MOST suitable for the user's query.
    2. Rank them by relevance.
    3. Output ONLY a JSON list of doctor names. Example: ["张三", "李四"]
    4. If no doctor is suitable, output [].
    """
    
    try:
        client = LLMClient()
        response = client.chat_completion(
            messages=[{"role": "user", "content": f"Query: {query}"}],
            system_prompt=system_prompt,
            temperature=0.1
        )
        
        # Parse the JSON response
        try:
            # Clean up potential markdown code blocks
            resp_str = str(response).strip()
            if resp_str.startswith("```"):
                resp_str = resp_str.split("\n", 1)[1].rsplit("\n", 1)[0]
            elif resp_str.startswith("`"):
                resp_str = resp_str.strip("`")
            
            suggested_names = json.loads(resp_str)
        except Exception:
            # Fallback: if LLM didn't return valid JSON, try to extract names that appear in our list
            suggested_names = []
            for d in docs:
                if d["name"] in resp_str:
                    suggested_names.append(d["name"])
        
        if not suggested_names:
             return "未找到匹配医生"

        # 3. Retrieve details for suggested doctors
        hits = []
        for name in suggested_names:
            for d in docs:
                if d["name"] == name:
                    hits.append(f"{d['name']}｜{d['dept']}｜擅长：{d['expertise']}")
                    break
        
        if not hits:
            return "未找到匹配医生"
            
        return "\n".join(hits)

    except Exception as e:
        print(f"LLM find doctor failed: {e}")
        # Fallback to simple keyword search
        hits = []
        for d in docs:
            blob = f"{d['name']}|{d['dept']}|{d['expertise']}"
            if query in blob:
                hits.append(d)
        if not hits:
            return "未找到匹配医生"
        out = []
        for d in hits[:5]:
            out.append(f"{d['name']}｜{d['dept']}｜擅长：{d['expertise']}")
        return "\n".join(out)


def tool_find_doctors_by_department(args: Dict[str, Any]) -> str:
    dept = (args.get("department") or "").strip()
    if not dept:
        return "缺少 department 参数"
    docs = _load_doctors()
    hits = [d for d in docs if dept in d["dept"]]
    if not hits:
        return f"{dept} 暂无匹配医生"
    out = []
    for d in hits[:12]:
        out.append(f"{d['name']}｜擅长：{d['expertise']}")
    return "\n".join(out)


def tool_get_department_info(args: Dict[str, Any]) -> str:
    dept = (args.get("department") or "").strip()
    if not dept:
        return "缺少 department 参数"
    depts = _load_departments()
    info = depts.get(dept)
    if not info:
        return f"{dept} 暂无信息"
    desc = "；".join(info.get("desc") or []) or "暂无简介"
    addr = info.get("address") or "地址未收录"
    return f"{dept}｜简介：{desc}｜地址：{addr}"


from utils.rag import get_knowledge_base

def tool_search_knowledge_base(args: Dict[str, Any]) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "请提供查询问题"
        
    try:
        kb = get_knowledge_base()
        chunks = kb.search(query)
        if not chunks:
            return "未在知识库中找到相关信息，请咨询人工客服。"
            
        return "知识库检索结果：\n\n" + "\n---\n".join(chunks)
    except Exception as e:
        return f"知识库检索失败: {e}"

def triage_tools() -> List[Tool]:
    return [
        ("lookup_department_by_symptom", "根据症状推荐科室，参数：{\"text\": \"症状描述\"}", tool_lookup_department_by_symptom),
        ("get_department_info", "获取科室信息，参数：{\"department\": \"科室名\"}", tool_get_department_info),
        ("check_emergency_signs", "识别急症征象，参数：{\"text\": \"症状描述\"}", tool_check_emergency_signs),
        ("find_doctor", "按关键词或姓名查医生，参数：{\"query\": \"姓名或关键词\"}", tool_find_doctor),
    ]


def appointment_tools() -> List[Tool]:
    return [
        ("query_schedule", "查询医生排班，参数：{\"doctor_name\": \"X医生\", \"date\": \"YYYY-MM-DD|今天|明天\"}", tool_query_schedule),
        ("find_doctors_by_department", "按科室查医生，参数：{\"department\": \"科室名\"}", tool_find_doctors_by_department),
        ("find_doctor", "按关键词查医生，参数：{\"query\": \"姓名或关键词\"}", tool_find_doctor),
    ]


def inquiry_tools() -> List[Tool]:
    return [
        ("search_knowledge_base", "检索医院常见问题知识库，参数：{\"query\": \"问题描述\"}", tool_search_knowledge_base),
        ("get_hospital_info", "获取医院基础信息，无参数", tool_get_hospital_info),
        ("get_opening_hours", "获取医院门诊/急诊时间，无参数", tool_get_opening_hours),
        ("get_department_info", "获取科室信息，参数：{\"department\": \"科室名\"}", tool_get_department_info),
        ("find_doctor", "按关键词查医生，参数：{\"query\": \"姓名或关键词\"}", tool_find_doctor),
    ]


def report_tools() -> List[Tool]:
    return [
        ("get_lab_reference_range", "获取化验项目参考区间，参数：{\"test_name\": \"WBC|RBC|HGB|GLU\"}", tool_get_lab_reference_range),
        ("explain_indicator", "结合数值给出客观判断，参数：{\"test_name\": \"WBC\", \"value\": 12.5}", tool_explain_indicator),
        ("find_doctor", "按关键词查医生，参数：{\"query\": \"姓名或关键词\"}", tool_find_doctor),
        ("get_department_info", "获取科室信息，参数：{\"department\": \"科室名\"}", tool_get_department_info),
    ]


def empty_tools() -> List[Tool]:
    return []
