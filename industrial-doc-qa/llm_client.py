# -*- coding: utf-8 -*-
"""
llm_client.py —— 大模型客户端封装与多轮对话历史模块
====================================================

本模块包含三个类：

1. :class:`BaseLLM`     —— 抽象基类：定义所有大模型客户端的统一接口 chat()；
2. :class:`OpenAILLM`   —— 基于 openai SDK 的实现（默认使用 DeepSeek），
                          通过环境变量配置，兼容 OpenAI 官方及各种
                          OpenAI 兼容接口（如 DeepSeek、通义千问、vLLM 等）；
3. :class:`OllamaLLM`   —— 本地 Ollama 备选实现：无需 API Key，离线可跑，
                          只依赖标准库 urllib，不引入额外包；
4. :class:`ChatHistory` —— 多轮对话历史：固定窗口保存最近 max_turns 轮，
                          防止对话越拖越长导致超出模型上下文。

环境变量说明（OpenAILLM，默认 DeepSeek）：
- DEEPSEEK_API_KEY / OPENAI_API_KEY：接口密钥（必填，优先读取 DEEPSEEK_API_KEY）；
- OPENAI_BASE_URL                  ：接口地址（可选，默认 DeepSeek 官方地址）；
- LLM_MODEL                        ：模型名（可选，默认 deepseek-chat）。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union


# ---------------------------------------------------------------------- #
# 本地配置（仅用于快速测试，生产环境请使用环境变量）
# ---------------------------------------------------------------------- #
# 安全警告：请勿将真实 API Key 写入本文件后提交到 Git！
# 建议做法：将 DEEPSEEK_API_KEY 设置为环境变量，或放在 .env 文件中。
LOCAL_DEEPSEEK_API_KEY: Optional[str] = "sk-3a2ecca31b0546909c994e115954431f"  # ???"sk-xxxx"
LOCAL_OPENAI_BASE_URL: Optional[str] = "https://api.deepseek.com"
LOCAL_LLM_MODEL: Optional[str] = "deepseek-chat"


# ---------------------------------------------------------------------- #
# 抽象基类
# ---------------------------------------------------------------------- #
class BaseLLM(ABC):
    """大模型客户端抽象基类：所有实现都必须提供 chat() 方法。"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """与大模型对话。

        Args:
            messages:    OpenAI Chat 格式的消息列表
                         [{"role": "system"/"user"/"assistant", "content": "..."}]。
            temperature: 采样温度，越低回答越保守稳定（知识库问答建议 0.1~0.3）。
            stream:      是否流式输出：
                         - False → 返回完整回答字符串；
                         - True  → 返回生成器，逐个产出文本片段（打字机效果）。

        Returns:
            完整回答字符串，或文本片段生成器。
        """
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# OpenAI 兼容接口实现（默认）
# ---------------------------------------------------------------------- #
class OpenAILLM(BaseLLM):
    """基于 openai SDK 的大模型客户端（默认对接 DeepSeek）。

    Args:
        api_key:  接口密钥；None 时优先读 DEEPSEEK_API_KEY，再读 OPENAI_API_KEY。
        base_url: 接口地址；None 时读 OPENAI_BASE_URL，默认 DeepSeek 官方地址。
        model:    模型名；None 时读 LLM_MODEL，默认 deepseek-chat。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("缺少 openai 依赖，请先执行：pip install openai") from exc

        self.api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or LOCAL_DEEPSEEK_API_KEY
            or ""
        )
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or LOCAL_OPENAI_BASE_URL
            or "https://api.deepseek.com"
        )
        self.model = (
            model
            or os.getenv("LLM_MODEL")
            or LOCAL_LLM_MODEL
            or "deepseek-chat"
        )

        if LOCAL_DEEPSEEK_API_KEY:
            print("[提示] 正在使用 llm_client.py 中 LOCAL_DEEPSEEK_API_KEY 配置的 API Key。")
            print("       请勿将含真实密钥的 llm_client.py 提交到 Git！")

        if not self.api_key:
            # 不直接抛异常：让程序先跑起来，调用时才报友好错误（见 chat 方法）
            print("[警告] 未检测到 DEEPSEEK_API_KEY / OPENAI_API_KEY 环境变量，大模型调用将会失败。")
            print("       DeepSeek 配置示例：export DEEPSEEK_API_KEY=sk-xxxx")

        # openai SDK 要求 api_key 非空，未配置时填占位串，真正的校验交给服务端
        self.client = OpenAI(api_key=self.api_key or "EMPTY", base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """调用 OpenAI 兼容的 Chat Completions 接口，支持流式输出。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY / OPENAI_API_KEY，也无法从 llm_client.py 中的 "
                "LOCAL_DEEPSEEK_API_KEY 读取到有效密钥，无法调用大模型。"
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
        except Exception as exc:
            raise RuntimeError(
                f"调用大模型接口失败：{exc}\n"
                "（请检查 DEEPSEEK_API_KEY / OPENAI_BASE_URL / LLM_MODEL 配置是否正确）"
            ) from exc

        if stream:
            return self._stream_generator(response)
        return response.choices[0].message.content or ""

    @staticmethod
    def _stream_generator(response) -> Iterator[str]:
        """把 SDK 的流式响应逐个转成纯文本片段。"""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# ---------------------------------------------------------------------- #
# 本地 Ollama 备选实现（可选）
# ---------------------------------------------------------------------- #
class OllamaLLM(BaseLLM):
    """本地 Ollama 大模型客户端（备选方案）。

    适用场景：没有 OpenAI Key、或要求数据不出内网 / 完全离线。
    前置条件：安装 Ollama 并执行 `ollama pull qwen2.5:7b`、`ollama serve`。
    实现上只用标准库 urllib 请求 Ollama 的 /api/chat 接口，不增加依赖。

    Args:
        model:    Ollama 模型名，默认 qwen2.5:7b。
        base_url: Ollama 服务地址，默认 http://localhost:11434。
    """

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """调用本地 Ollama 的 /api/chat 接口，支持流式输出。"""
        import json
        import urllib.request

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except Exception as exc:
            raise RuntimeError(
                f"调用本地 Ollama 服务失败：{exc}\n"
                f"（请确认已执行 ollama serve，并已拉取模型：ollama pull {self.model}）"
            ) from exc

        if stream:
            return self._stream_generator(response)
        # 非流式：一次性读完整个 JSON 响应
        data = json.loads(response.read().decode("utf-8"))
        return data.get("message", {}).get("content", "")

    @staticmethod
    def _stream_generator(response) -> Iterator[str]:
        """Ollama 流式响应是逐行的 JSON（NDJSON），逐行解析出文本片段。"""
        import json

        for line in response:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue  # 跳过个别解析失败的行，不中断生成
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done"):
                break


# ---------------------------------------------------------------------- #
# 多轮对话历史
# ---------------------------------------------------------------------- #
class ChatHistory:
    """多轮对话历史：固定窗口保存最近 max_turns 轮问答。

    为什么要限制窗口？——每轮对话都会塞进 prompt，轮次太多会：
    1. 超出模型上下文长度限制；2. 增加接口调用费用；3. 早期话题干扰当前回答。

    Args:
        max_turns: 最多保留的对话轮数（一轮 = 一次提问 + 一次回答）。
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        # 每轮记录为 {"question": 用户原话, "answer": 助手回答}
        self.turns: List[Dict[str, str]] = []

    def add_turn(self, question: str, answer: str) -> None:
        """追加一轮对话；超过 max_turns 时自动丢弃最早的轮次。"""
        self.turns.append({"question": question, "answer": answer})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def to_messages(self) -> List[Dict]:
        """转换成 OpenAI messages 格式（拼进 prompt 时用）。"""
        messages: List[Dict] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        return messages

    def to_text(self) -> str:
        """转换成纯文本格式（给问题改写 prompt 用）。"""
        if not self.turns:
            return "（暂无历史对话）"
        lines: List[str] = []
        for i, turn in enumerate(self.turns, start=1):
            lines.append(f"第{i}轮 用户：{turn['question']}")
            lines.append(f"第{i}轮 助手：{turn['answer']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """清空全部历史。"""
        self.turns = []

    def __len__(self) -> int:
        return len(self.turns)


if __name__ == "__main__":
    # 简单自测：仅演示 ChatHistory 的窗口截断逻辑（不真正调用大模型）
    history = ChatHistory(max_turns=2)
    history.add_turn("问题1", "回答1")
    history.add_turn("问题2", "回答2")
    history.add_turn("问题3", "回答3")  # 超出窗口，问题1 会被丢弃
    print(f"当前保留 {len(history)} 轮对话：")
    print(history.to_text())
