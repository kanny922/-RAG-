# -*- coding: utf-8 -*-
"""
prompt_template.py —— 提示词（Prompt）模板模块
==============================================

Prompt 是与大模型沟通的"说明书"，直接决定回答质量。本模块集中管理三类提示词：

1. SYSTEM_PROMPT  —— 系统角色设定：告诉模型"你是谁、能做什么、不能做什么"；
2. build_prompt() —— 问答提示词组装：把检索到的参考资料 + 历史对话 + 用户问题
                     拼成符合 OpenAI messages 格式的完整输入；
3. REWRITE_PROMPT —— 问题改写提示词：多轮对话中，用户常会用"它""这个"等指代，
                     直接拿去检索效果很差，需要先改写成独立完整的问题。

教学要点：
- "仅依据给定上下文回答"是抑制大模型幻觉（编造答案）的关键约束；
- 要求模型用 [1][2] 标注引用，前端才能把答案和知识库原文对应起来（溯源）；
- 工业场景涉及人身安全，必须要求模型提醒遵守安全规程。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

# ---------------------------------------------------------------------- #
# 1. 系统提示词：角色与行为准则
# ---------------------------------------------------------------------- #
SYSTEM_PROMPT = """你是一名严谨专业的工业知识库问答助手，服务于工厂一线工程师与管理人员。

请严格遵守以下行为准则：
1. 【仅依据资料作答】你只能依据用户消息中给出的"参考资料"回答问题，严禁编造资料中不存在的信息；
2. 【不知道就明说】如果参考资料不足以回答问题，必须如实回答"根据现有资料无法回答"，并可以建议用户补充相关文档；
3. 【标注引用来源】回答中引用资料内容时，必须在对应位置用 [1][2] 等编号标注来源；
4. 【专业严谨】使用规范、准确的工业术语，语气专业严谨，不闲聊、不猜测；
5. 【安全第一】凡涉及设备操作、检修、工艺参数调整等可能影响人身与设备安全的内容，
   必须在回答末尾明确提示：现场操作须严格遵守相关安全规程与作业指导书。"""

# ---------------------------------------------------------------------- #
# 2. 问题改写提示词：多轮对话 → 独立检索问题
# ---------------------------------------------------------------------- #
REWRITE_PROMPT = """你是一个查询改写助手。请根据下面的对话历史，把用户的最新追问改写成一个独立、完整、可直接用于知识库检索的问题。

改写要求：
1. 补全追问中省略的指代与主语（如"它""这个""上面说的"）；
2. 不要回答问题本身，只输出改写后的问题；
3. 如果原问题已经足够独立完整，请原样输出；
4. 输出中不要包含任何解释或引号。

【对话历史】
{history}

【最新追问】
{question}

【改写后的独立问题】"""


def build_prompt(
    question: str,
    contexts: List[Dict],
    history: Optional[Union[object, List[Dict]]] = None,
) -> List[Dict]:
    """组装完整的问答 messages（OpenAI Chat 格式）。

    Args:
        question: 用户问题（建议传入经过改写的独立问题）。
        contexts: Retriever 返回的检索结果列表，
                  每项形如 {"content": ..., "metadata": {"source": ..., "page": ...}}。
        history:  多轮对话历史，支持两种形式：
                  - ChatHistory 对象（有 to_messages() 方法）；
                  - 或直接的 messages 列表。
                  传 None 表示单轮问答。

    Returns:
        messages 列表，可直接传给 LLM 的 chat() 方法：
        [system 角色设定] + [历史对话若干轮] + [user：参考资料 + 问题]
    """
    # ---------- 1. 把检索结果格式化为带编号的参考资料 ----------
    context_blocks: List[str] = []
    for index, ctx in enumerate(contexts, start=1):
        metadata = ctx.get("metadata", {}) or {}
        source = metadata.get("source", "未知文档")
        page = metadata.get("page", "")
        page_info = f"第 {page} 页" if page not in ("", None) else "页码未知"
        content = (ctx.get("content") or "").strip()
        context_blocks.append(f"[{index}] 来源：《{source}》（{page_info}）\n{content}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "（本次未检索到任何参考资料）"

    # ---------- 2. 拼装用户消息：参考资料在前，问题在后 ----------
    user_content = (
        f"【参考资料】\n{context_text}\n\n"
        f"【用户问题】\n{question}\n\n"
        "请仅依据上述参考资料作答，并在引用处用 [编号] 标注来源；"
        "资料不足以回答时，请直接回答“根据现有资料无法回答”。"
    )

    # ---------- 3. 按 OpenAI messages 格式组装完整输入 ----------
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history is not None:
        # 兼容 ChatHistory 对象与普通 messages 列表两种传法
        if hasattr(history, "to_messages"):
            messages.extend(history.to_messages())
        else:
            messages.extend(history)

    messages.append({"role": "user", "content": user_content})
    return messages


if __name__ == "__main__":
    # 简单自测：观察组装后的 prompt 长什么样
    demo_contexts = [
        {
            "content": "离心泵轴承温度不得超过 75 摄氏度，振动速度有效值不得超过 4.5 毫米每秒。",
            "metadata": {"source": "设备巡检规范.pdf", "page": 3},
        },
        {
            "content": "发现设备异常应立即上报班长，并按《设备异常处理流程》执行。",
            "metadata": {"source": "设备巡检规范.pdf", "page": 4},
        },
    ]
    msgs = build_prompt("离心泵轴承温度上限是多少？", demo_contexts)
    for m in msgs:
        print(f"===== {m['role']} =====")
        print(m["content"][:300])
        print()
