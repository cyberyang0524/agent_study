# 医院客服多智能体系统 (Hospital Customer Service Multi-Agent System)

这是一个基于 Python 的多智能体系统演示学习项目，旨在模拟现代化的医院智能客服场景。系统采用了 **ReAct (Reasoning + Acting)** 架构，结合 **RAG (检索增强生成)** 技术和 **多模态** 能力，能够通过协调多个专门的智能体来服务患者。(个人学习用)

## 核心特性

*   **多智能体协作 (Multi-Agent)**: 包含路由、分诊、挂号、咨询、报告解读等多个专业智能体。
*   **ReAct 架构**: 智能体具备"思考-行动-观察"的循环能力，能够自主调用工具获取信息，而非仅靠模型训练数据。
*   **RAG 知识库**: 内置检索增强生成引擎，基于本地 `faq.txt` 提供精准的医院常见问题解答。
*   **智能匹配**:
    *   **智能分诊**: 基于 LLM 对症状进行深度语义分析，匹配最合适的科室（支持模糊描述）。
    *   **医生推荐**: 根据用户描述（如"手抖"、"失眠"）智能推荐对应擅长领域的专家。
*   **多模态支持**: 报告解读智能体支持读取和分析医疗影像/报告单图片。
*   **流式输出**: 支持打字机效果的实时流式回复。
*   **本地知识库**: demo通过简单的文本文件 (`doc.txt`, `dept.txt`, `faq.txt`) 定制医院数据。

## 系统架构

系统由以下核心智能体组成：

1.  **Router Agent (路由智能体)**: 系统的"大脑"，负责接收用户输入，分析意图，并将任务分发给最合适的子智能体。
2.  **Triage Agent (分诊智能体)**: 负责根据用户描述的症状，利用医学知识和科室数据库建议挂号科室。
3.  **Appointment Agent (挂号智能体)**: 负责查询医生排班、推荐医生并处理预约流程。
4.  **Inquiry Agent (咨询智能体)**: 负责回答关于医院的基础信息（利用 RAG 技术检索 FAQ）。
5.  **Report Agent (报告解读智能体计划学习中)**: 负责解读检验检查报告（支持上传图片）。

## 目录结构

```text
d:\agent\
├── main.py                 # 系统入口，处理用户交互循环
├── config.py               # 配置文件 (模型配置, ReAct 开关)
├── agents/                 # 智能体实现
│   ├── base_agent.py       # 基础智能体类 (定义 ReAct 循环入口)
│   ├── router_agent.py     # 路由智能体
│   ├── triage_agent.py     # 分诊智能体
│   ├── appointment_agent.py # 挂号智能体
│   ├── inquiry_agent.py    # 咨询智能体
│   └── report_agent.py     # 报告解读智能体
├── utils/                  # 工具类
│   ├── llm_client.py       # LLM 客户端 (支持智谱AI/OpenAI, 包含 Embedding)
│   ├── react.py            # ReAct 执行引擎 (Thought-Action-Observation 循环)
│   ├── rag.py              # RAG 检索引擎 (向量检索)
│   └── tools.py            # 工具函数库 (定义所有 Agent 可用的 Tool)
├── doc.txt                 # 医生信息库 (姓名|科室|擅长)
├── dept.txt                # 科室介绍库 (科室名|介绍|地址)
└── faq.txt                 # 常见问题知识库 (Q&A)
```

## 快速开始

### 1. 环境准备
确保 Python 环境已安装，并安装必要依赖：
```bash
pip install openai
```
*(注：本项目主要依赖 `openai` SDK 来调用兼容接口)*

### 2. 配置 API Key
在环境变量中设置 API Key (推荐使用智谱 AI):
```bash
set ZHIPUAI_API_KEY=your_api_key_here
# 或者
set OPENAI_API_KEY=your_api_key_here
```
或者直接在 `utils/llm_client.py` 中配置。

### 3. 定制数据 (可选)
您可以直接编辑根目录下的 `.txt` 文件来更新医院数据：
*   `doc.txt`: 添加或修改医生信息。
*   `dept.txt`: 修改科室介绍和地址。
*   `faq.txt`: 添加客服常见问答。

### 4. 运行系统
```bash
python main.py
```

## 交互示例

*   **分诊**: "我头痛得厉害，有时候还恶心，该挂什么科？"
*   **找医生**: "我想找个擅长看帕金森的专家。"
*   **咨询**: "你们医院几点下班？医保能用吗？"
*   **挂号**: "帮我查一下王医生明天的号。"
*   **报告解读**: (上传图片或输入数值) "帮我看看这个血常规报告，白细胞有点高。"
