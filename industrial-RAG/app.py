# -*- coding: utf-8 -*-
"""
app.py —— Gradio 前端与 RAG 全流程串联
======================================

本文件是整个项目的"总装车间"，把前面各模块串成完整链路：

    document_loader → text_splitter → vector_store（入库）
    用户提问 → 多轮改写（REWRITE_PROMPT）→ retriever（召回+重排）
             → prompt_template（组装）→ llm_client（流式生成）→ 界面展示

启动方式：
    python app.py
    然后浏览器访问 http://localhost:7860

界面布局（gr.Blocks）：
- 左侧栏：文档上传组件、"重建索引"按钮、知识库统计信息；
- 右侧栏：对话窗口 + 问题输入框 + "溯源来源"折叠面板
          （展示每条引用来源的 文档名 / 页码 / 得分 / 原文片段，
           答案中实际引用到的原文用 【】 高亮标记）。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from document_loader import DocumentLoader
from llm_client import BaseLLM, ChatHistory, OpenAILLM
from prompt_template import REWRITE_PROMPT, build_prompt
from retriever import Retriever
from text_splitter import TextSplitter
from vector_store import EmbeddingModel, VectorStore

# ---------------------------------------------------------------------- #
# 全局常量与组件
# ---------------------------------------------------------------------- #
DOCS_DIR = Path("docs")          # 知识库文档目录
INDEX_DIR = "./chroma_db"        # 向量索引持久化目录

embedding_model: Optional[EmbeddingModel] = None
vector_store: Optional[VectorStore] = None
retriever: Optional[Retriever] = None
llm: Optional[BaseLLM] = None
init_error: str = ""             # 记录初始化失败原因，展示给用户


# ---------------------------------------------------------------------- #
# 一、初始化与索引构建
# ---------------------------------------------------------------------- #
def init_knowledge_base() -> None:
    """程序启动时初始化：准备 docs 目录、加载模型、构建或复用向量索引。

    关键逻辑——index_exists 判断：
    向量索引已持久化到磁盘（chroma_db/），如果库中已有数据，说明之前
    已经入过库，本次直接复用，避免每次启动都重复解析 + 向量化（很耗时）。
    """
    global embedding_model, vector_store, retriever, llm, init_error

    # 1. 确保知识库目录存在，不存在则创建并提示用户放入文档
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"[提示] 已自动创建知识库目录：{DOCS_DIR.resolve()}")
        print("       请将 pdf / docx / txt 文档放入该目录，然后在界面上点击「重建索引」。")

    # 2. 加载 Embedding 模型（失败则记录原因，界面给出友好提示）
    try:
        embedding_model = EmbeddingModel()
    except Exception as exc:
        init_error = f"Embedding 模型加载失败：{exc}"
        print(f"[错误] {init_error}")
        return

    # 3. 打开（或创建）持久化向量库
    try:
        vector_store = VectorStore(persist_directory=INDEX_DIR)
    except Exception as exc:
        init_error = f"向量数据库初始化失败：{exc}"
        print(f"[错误] {init_error}")
        return

    # 4. index_exists 判断：索引为空 且 docs 中有文档 → 首次自动构建
    if vector_store.count() == 0:
        has_files = any(
            p.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS
            for p in DOCS_DIR.rglob("*") if p.is_file()
        )
        if has_files:
            print("[信息] 检测到空索引，正在首次构建知识库索引……")
            try:
                build_index()
            except Exception as exc:
                print(f"[错误] 首次构建索引失败：{exc}（可在界面点击「重建索引」重试）")
        else:
            print("[提示] 知识库为空：docs/ 目录中暂无文档，等待用户上传。")
    else:
        print(f"[信息] 检测到已有持久化索引（{vector_store.count()} 个文本块），直接复用，无需重复入库。")

    # 5. 创建检索器与 LLM 客户端
    retriever = Retriever(embedding_model, vector_store, use_reranker=False)
    try:
        llm = OpenAILLM()  # 默认对接 DeepSeek；如需本地离线模型，可替换为 OllamaLLM()
    except Exception as exc:
        print(f"[警告] LLM 初始化失败：{exc}")
        llm = None


def build_index() -> Tuple[int, int]:
    """完整重建索引：清空旧索引 → 解析文档 → 分块 → 向量化 → 入库。

    Returns:
        (文档片段数, 向量块数) 二元组，供界面统计展示。
    """
    loader = DocumentLoader()
    splitter = TextSplitter(chunk_size=400, chunk_overlap=80)

    documents = loader.load_directory(DOCS_DIR)   # ① 解析全部文档
    if not documents:
        return 0, 0
    chunks = splitter.split_documents(documents)  # ② 递归分块（附 chunk_id）
    print(f"[信息] 文档分块完成：{len(documents)} 个片段 → {len(chunks)} 个文本块。")
    embeddings = embedding_model.encode([c.page_content for c in chunks])  # ③ 向量化
    vector_store.clear()                          # ④ 清空旧索引后再入库（保证幂等）
    vector_store.add_documents(chunks, embeddings)
    return len(documents), len(chunks)


# ---------------------------------------------------------------------- #
# 二、界面事件处理：上传 / 重建索引 / 统计
# ---------------------------------------------------------------------- #
def handle_upload(files) -> str:
    """把用户上传的文件保存到 docs/ 目录（需再点「重建索引」才会生效）。"""
    if not files:
        return "⚠️ 未选择任何文件。"
    DOCS_DIR.mkdir(exist_ok=True)
    saved: List[str] = []
    for file in files:
        # 兼容 gradio 不同版本：上传对象可能是路径字符串，也可能是带 .name 的对象
        src = Path(file if isinstance(file, str) else file.name)
        dst = DOCS_DIR / src.name
        shutil.copy(src, dst)
        saved.append(src.name)
    return f"✅ 已上传 {len(saved)} 个文件：{'、'.join(saved)}\n\n👉 请点击「重建索引」使新文档生效。"


def get_kb_stats() -> str:
    """生成知识库统计信息（Markdown 格式）。"""
    if vector_store is None:
        return f"### 📚 知识库统计\n\n❌ 知识库未初始化。\n\n{init_error}"
    files = [
        p.name for p in DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS
    ] if DOCS_DIR.exists() else []
    return (
        "### 📚 知识库统计\n\n"
        f"- 文档数量：**{len(files)}** 个\n"
        f"- 向量文本块：**{vector_store.count()}** 条\n"
        f"- 索引目录：`{INDEX_DIR}`（已持久化，重启自动复用）"
    )


def handle_rebuild() -> str:
    """「重建索引」按钮的处理函数：全量重建并返回最新统计。"""
    if embedding_model is None or vector_store is None:
        return f"❌ 无法重建索引：{init_error or '系统未初始化'}"
    try:
        doc_count, chunk_count = build_index()
    except Exception as exc:
        return f"❌ 重建索引失败：{exc}\n\n" + get_kb_stats()
    if chunk_count == 0:
        return "⚠️ docs/ 目录中没有可用文档，请先上传或手动放入文档。\n\n" + get_kb_stats()
    return f"✅ 索引重建完成！共解析 **{doc_count}** 个文档片段，生成 **{chunk_count}** 个向量块。\n\n" + get_kb_stats()


# ---------------------------------------------------------------------- #
# 三、溯源来源面板：引用高亮
# ---------------------------------------------------------------------- #
def highlight_cited(content: str, answer: str, min_len: int = 6) -> str:
    """把答案中实际引用到的原文用【】高亮（教学向的简单实现）。

    判断"被引用"的两条规则（满足其一即高亮）：
    1. 原文中的某个句子原样出现在答案里（模型复述了原文）；
    2. 兜底：取答案与原文的"最长公共子串"，长度达标则高亮。
    """
    # 规则 1：按中文句末标点切句，出现在答案中的句子整体高亮
    sentences = re.split(r"(?<=[。；！？\n])", content)
    marked: List[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if len(stripped) >= min_len and stripped in answer:
            marked.append(sentence.replace(stripped, f"【{stripped}】"))
        else:
            marked.append(sentence)
    highlighted = "".join(marked)
    if "【" in highlighted:
        return highlighted

    # 规则 2（兜底）：最长公共子串高亮
    lcs = _longest_common_substring(answer, content)
    if len(lcs) >= min_len:
        return highlighted.replace(lcs, f"【{lcs}】", 1)
    return highlighted


def _longest_common_substring(a: str, b: str) -> str:
    """求两个字符串的最长公共子串（经典动态规划，滚动数组省内存）。"""
    if not a or not b:
        return ""
    if len(a) > len(b):  # 保证 a 是较短串，作为 DP 的行
        a, b = b, a
    prev = [0] * (len(a) + 1)
    best_len, best_end = 0, 0
    for j in range(1, len(b) + 1):
        curr = [0] * (len(a) + 1)
        for i in range(1, len(a) + 1):
            if a[i - 1] == b[j - 1]:
                curr[i] = prev[i - 1] + 1
                if curr[i] > best_len:
                    best_len, best_end = curr[i], i
        prev = curr
    return a[best_end - best_len:best_end]


def format_sources(contexts: List[Dict], answer: str) -> str:
    """把检索结果格式化为溯源来源面板内容（Markdown）。

    每条来源展示：编号、文档名、页码、相关度得分、原文片段（引用处【】高亮）。
    """
    if not contexts:
        return "（本次回答未检索到可参考的知识库内容）"
    blocks: List[str] = []
    for index, ctx in enumerate(contexts, start=1):
        metadata = ctx.get("metadata", {}) or {}
        source = metadata.get("source", "未知文档")
        page = metadata.get("page", "")
        page_text = f"第 {page} 页" if page not in ("", None) else "页码未知"
        score = ctx.get("score", 0.0)

        snippet = (ctx.get("content") or "").strip()
        if len(snippet) > 200:  # 片段过长时截断，保证面板整洁
            snippet = snippet[:200] + "……"
        snippet = highlight_cited(snippet, answer)

        blocks.append(
            f"**[{index}] 《{source}》 {page_text} · 相关度 {score:.4f}**\n\n> {snippet}"
        )
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------- #
# 四、回答主流程（生成器：支持流式输出）
# ---------------------------------------------------------------------- #
def respond(user_message: str, chatbot: List[Dict], history_state: ChatHistory):
    """回答主流程：历史改写 → 检索重排 → 组装 prompt → LLM 流式生成。

    本函数是**生成器**：每次 yield 都会把最新界面状态推送给 Gradio，
    从而实现"打字机"式的流式回答效果。

    Args:
        user_message:  用户输入的问题。
        chatbot:       Gradio Chatbot 当前消息列表（messages 格式）。
        history_state: gr.State 保存的 ChatHistory 对象（多轮对话窗口）。

    Yields:
        (chatbot, history_state, 溯源面板内容) 三元组。
    """
    history = history_state or ChatHistory(max_turns=5)
    question = (user_message or "").strip()

    # ---------- 0. 前置检查：给出友好中文提示而非报错 ----------
    if not question:
        yield chatbot, history, "⚠️ 请输入问题。"
        return
    if retriever is None or llm is None:
        error_text = (
            "⚠️ 系统尚未就绪，无法回答问题。\n\n"
            f"原因：{init_error or 'LLM 或检索器初始化失败'}\n\n"
            "请检查：① 网络能否访问 DeepSeek；② DEEPSEEK_API_KEY 等环境变量是否配置。"
        )
        chatbot = chatbot + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_text},
        ]
        yield chatbot, history, format_sources([], "")
        return
    if vector_store.count() == 0:
        chatbot = chatbot + [
            {"role": "user", "content": question},
            {"role": "assistant", "content":
                "知识库目前是空的。请先在左侧上传文档并点击「重建索引」，然后再提问。"},
        ]
        yield chatbot, history, format_sources([], "")
        return

    # 先把用户问题和一个空的助手气泡放上界面，后续流式填充
    chatbot = chatbot + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    yield chatbot, history, "🔎 正在检索知识库……"

    # ---------- 1. 多轮改写：把追问改写成独立问题再检索 ----------
    standalone_question = question
    if len(history) > 0:
        try:
            rewrite_messages = [{
                "role": "user",
                "content": REWRITE_PROMPT.format(
                    history=history.to_text(), question=question
                ),
            }]
            rewritten = llm.chat(rewrite_messages, temperature=0.1, stream=False)
            if rewritten and rewritten.strip():
                standalone_question = rewritten.strip()
                print(f"[信息] 问题改写：「{question}」→「{standalone_question}」")
        except Exception as exc:
            print(f"[提示] 问题改写失败，改用原始问题检索：{exc}")

    # ---------- 2. 检索 + 重排 ----------
    try:
        contexts = retriever.retrieve(standalone_question)
    except Exception as exc:
        chatbot[-1]["content"] = f"❌ 检索知识库时出错：{exc}"
        yield chatbot, history, format_sources([], "")
        return

    # ---------- 3. 组装 prompt，调用 LLM 流式生成 ----------
    messages = build_prompt(standalone_question, contexts, history)
    answer = ""
    try:
        stream = llm.chat(messages, temperature=0.3, stream=True)
        for delta in stream:  # 逐片段累积，增量刷新界面
            answer += delta
            chatbot[-1]["content"] = answer
            yield chatbot, history, format_sources(contexts, answer)
    except Exception as exc:
        answer = (
            f"❌ 调用大模型失败：{exc}\n\n"
            "（请检查 DEEPSEEK_API_KEY / OPENAI_BASE_URL / LLM_MODEL 环境变量配置）"
        )
        chatbot[-1]["content"] = answer
        yield chatbot, history, format_sources(contexts, answer)

    # ---------- 4. 回答完成后写入历史（供下一轮改写与上下文使用） ----------
    history.add_turn(question, answer)
    yield chatbot, history, format_sources(contexts, answer)


def clear_chat():
    """「清空对话」按钮：重置聊天窗口、历史记录与溯源面板。"""
    return [], ChatHistory(max_turns=5), "（提问后此处显示引用来源）"


# ---------------------------------------------------------------------- #
# 五、Gradio 界面搭建
# ---------------------------------------------------------------------- #
def create_demo() -> gr.Blocks:
    """搭建 Gradio 界面：左侧知识库管理，右侧问答对话。"""
    with gr.Blocks(title="工业业务智能文档问答助手") as demo:
        gr.Markdown(
            "# 🏭 工业业务智能文档问答助手\n"
            "基于 **RAG（检索增强生成）**：先检索企业知识库，再由大模型依据资料作答，"
            "回答可溯源、有依据。"
        )

        # 多轮对话历史：用 gr.State 在每个会话中独立保存
        history_state = gr.State(ChatHistory(max_turns=5))

        with gr.Row():
            # ---------------- 左侧：知识库管理 ----------------
            with gr.Column(scale=1):
                gr.Markdown("### 📁 知识库管理")
                file_upload = gr.File(
                    label="上传文档（支持 pdf / docx / txt）",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"],
                )
                upload_status = gr.Markdown()
                rebuild_btn = gr.Button("🔄 重建索引", variant="primary")
                kb_stats = gr.Markdown(get_kb_stats())

            # ---------------- 右侧：问答对话 ----------------
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="对话窗口",
                    height=420,
                )
                question_box = gr.Textbox(
                    label="请输入你的问题",
                    placeholder="例如：离心泵轴承温度不能超过多少度？",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空对话")
                with gr.Accordion("🔍 溯源来源（查看答案引用依据）", open=False):
                    source_panel = gr.Markdown("（提问后此处显示引用来源）")

        # ---------------- 事件绑定 ----------------
        # 上传文档：仅保存到 docs/，提示用户再点重建索引
        file_upload.upload(
            fn=handle_upload,
            inputs=file_upload,
            outputs=upload_status,
        )
        # 重建索引：全量重建并刷新统计
        rebuild_btn.click(
            fn=handle_rebuild,
            outputs=kb_stats,
        )
        # 发送问题（按钮 / 回车两种方式），回答后清空输入框
        for trigger in (submit_btn.click, question_box.submit):
            trigger(
                fn=respond,
                inputs=[question_box, chatbot, history_state],
                outputs=[chatbot, history_state, source_panel],
            ).then(lambda: "", outputs=question_box)
        # 清空对话
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, history_state, source_panel],
        )

    return demo


# ---------------------------------------------------------------------- #
# 程序入口
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    init_knowledge_base()           # 启动时初始化知识库（构建或复用索引）
    demo = create_demo()
    # queue() 开启请求队列，保证流式生成在多用户下稳定；监听 0.0.0.0 便于局域网访问
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
