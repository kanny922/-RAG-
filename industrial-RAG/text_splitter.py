# -*- coding: utf-8 -*-
"""
text_splitter.py —— 中文优先的递归文本分块模块
===============================================

为什么需要分块？
- Embedding 模型有输入长度限制，且文本越短语义越聚焦，检索越精准；
- 但块又不能切得太碎，否则上下文不完整，大模型"看不懂"。

本模块实现经典的**递归字符切分**（与 LangChain RecursiveCharacterTextSplitter 同思路），
并针对中文语料调整了分隔符优先级：

    段落("\\n\\n") > 换行("\\n") > 句号("。") > 分号("；") > 逗号("，") > 空格(" ")

切分思路（两步走）：
1. 递归切分：优先用高优先级分隔符把长文拆成不超过 chunk_size 的小片段；
2. 合并成块：把相邻小片段尽量拼满 chunk_size，块与块之间保留 chunk_overlap
   个字符的重叠，避免一个完整语义被从中间劈开后两边都"缺一半"。
"""

from __future__ import annotations

from typing import List, Optional

from document_loader import Document


class TextSplitter:
    """中文优先的递归文本分块器。

    Args:
        chunk_size:    每个文本块的最大字符数（按中文字符计）。
        chunk_overlap: 相邻块之间的重叠字符数，用于保留上下文衔接。
        separators:    分隔符优先级列表，一般使用默认值即可。
    """

    #: 默认分隔符优先级：从"语义最完整"到"最细碎"排列
    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " "]

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap（{chunk_overlap}）必须小于 chunk_size（{chunk_size}）"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or list(self.DEFAULT_SEPARATORS)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def split_text(self, text: str) -> List[str]:
        """把一段长文本切成若干不超过 chunk_size 的块（含重叠）。

        Args:
            text: 原始长文本。

        Returns:
            文本块列表；输入为空时返回空列表。
        """
        text = (text or "").strip()
        if not text:
            return []
        # 第一步：递归切成小片段
        splits = self._recursive_split(text, self.separators)
        # 第二步：合并小片段成最终的块（带重叠）
        return self._merge_splits(splits)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """批量切分 Document 列表，保留原 metadata 并附加 chunk_id。

        Args:
            documents: DocumentLoader 解析出的 Document 列表。

        Returns:
            切分后的 Document 列表，每个块的 metadata 中新增：
            - chunk_id : 全局唯一的块编号（向量入库时用作主键）
        """
        chunks: List[Document] = []
        chunk_index = 0
        for doc in documents:
            for piece in self.split_text(doc.page_content):
                piece = piece.strip()
                if not piece:
                    continue
                # 拷贝一份 metadata 再修改，避免污染原始 Document
                metadata = dict(doc.metadata)
                metadata["chunk_id"] = chunk_index
                chunks.append(Document(page_content=piece, metadata=metadata))
                chunk_index += 1
        return chunks

    # ------------------------------------------------------------------ #
    # 第一步：递归切分
    # ------------------------------------------------------------------ #
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """按分隔符优先级递归地把文本拆成不超过 chunk_size 的小片段。

        递归逻辑：
        - 找到当前文本中真实存在的、优先级最高的分隔符；
        - 用它把文本切开，分隔符拼回上一段末尾（"。"属于句子的一部分）；
        - 仍超过 chunk_size 的片段，用次一级分隔符继续递归切；
        - 所有分隔符都用尽了还超长 → 按 chunk_size 硬切（兜底）。
        """
        if len(text) <= self.chunk_size:
            return [text]

        # 1. 选出当前层级要用的分隔符：第一个真的出现在文本中的
        separator: Optional[str] = None
        next_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep in text:
                separator = sep
                next_separators = separators[i + 1:]
                break

        # 2. 兜底：没有任何分隔符可用，直接按固定长度硬切
        if separator is None:
            return [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        # 3. 用选中的分隔符切开，并把分隔符拼回上一段末尾
        raw_pieces = text.split(separator)
        pieces = [p + separator for p in raw_pieces[:-1]]
        if raw_pieces[-1]:  # 最后一段后面本来就没有分隔符
            pieces.append(raw_pieces[-1])

        # 4. 超长片段用次一级分隔符递归处理
        splits: List[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                splits.append(piece)
            else:
                splits.extend(self._recursive_split(piece, next_separators))
        return splits

    # ------------------------------------------------------------------ #
    # 第二步：合并小片段（带重叠）
    # ------------------------------------------------------------------ #
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """把相邻小片段尽量拼满 chunk_size，块间保留 chunk_overlap 重叠。

        重叠的实现方式：每当一个块"装满"输出后，不从零开始下一个块，
        而是保留队尾总长度不超过 chunk_overlap 的若干片段，作为下一块的开头。
        """
        merged: List[str] = []
        current: List[str] = []   # 当前块包含的小片段
        current_len = 0           # 当前块的字符总数

        for split in splits:
            split_len = len(split)
            if current and current_len + split_len > self.chunk_size:
                # 当前块装满了 → 输出，并保留尾部片段作为重叠
                merged.append("".join(current))
                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed)
            current.append(split)
            current_len += split_len

        if current:  # 别忘了最后一块
            merged.append("".join(current))
        return merged


if __name__ == "__main__":
    # 简单自测：直观感受中文优先的切分效果
    sample = (
        "第一章 设备巡检规范。\n\n"
        "离心泵应每班巡检一次，重点检查轴承温度、振动值与密封泄漏情况。"
        "轴承温度不得超过 75 摄氏度；振动速度有效值不得超过 4.5 毫米每秒。"
        "发现异常应立即上报班长，并按《设备异常处理流程》执行。\n\n"
        "第二章 安全注意事项。\n"
        "进入装置区必须佩戴安全帽与防护眼镜，严禁携带火种。"
    )
    splitter = TextSplitter(chunk_size=80, chunk_overlap=20)
    for i, chunk in enumerate(splitter.split_text(sample)):
        print(f"--- 块 {i}（{len(chunk)} 字）---")
        print(chunk)
