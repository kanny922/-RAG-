# -*- coding: utf-8 -*-
"""
retriever.py —— 两段式检索模块（召回 + 重排）
==============================================

为什么不只做一次向量检索？——这是工业界 RAG 的常见优化套路：

- **召回（Recall）**：向量检索速度快，但"语义相近"不等于"能回答问题"，
  所以先宽松地取回 recall_top_k（默认 20）条候选，宁可多不可少；
- **重排（Rerank）**：再用更精细的 CrossEncoder 模型把"问题 + 候选文本"
  拼在一起逐条打分，挑出真正相关的 rerank_top_n（默认 5）条交给大模型。

两段式设计兼顾了**速度**（向量索引毫秒级）与**精度**（CrossEncoder 打分更准）。

降级策略（教学要点）：
- 重排模型加载失败（没网 / 内存不足）时，自动降级为"直接取召回结果的前 N 条"，
  并打印中文提示，保证整个系统在任何环境下都能跑通主流程。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from vector_store import EmbeddingModel, VectorStore


class Retriever:
    """两段式检索器：向量召回 → CrossEncoder 重排。

    Args:
        embedding_model: EmbeddingModel 实例，负责把问题编码成向量。
        vector_store:    VectorStore 实例，负责向量相似度召回。
        recall_top_k:    第一阶段召回的候选条数。
        rerank_top_n:    第二阶段重排后保留的条数。
        reranker_name:   重排模型名，默认 BAAI/bge-reranker-base。
        use_reranker:    是否启用重排；设为 False 可跳过加载重排模型。
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        recall_top_k: int = 20,
        rerank_top_n: int = 5,
        reranker_name: str = "BAAI/bge-reranker-base",
        use_reranker: bool = True,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.recall_top_k = recall_top_k
        self.rerank_top_n = rerank_top_n
        self.reranker_name = reranker_name
        self.use_reranker = use_reranker

        # 重排模型懒加载：第一次检索时才初始化，加快程序启动速度
        self._reranker: Optional[object] = None
        self._reranker_checked: bool = False

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str) -> List[Dict]:
        """对一个问题执行完整的"召回 + 重排"检索。

        Args:
            question: 用户问题（建议先经过多轮改写，变成独立问题）。

        Returns:
            结构化结果列表，按最终得分从高到低排序，最多 rerank_top_n 条：
            [{"content": 文本块内容, "metadata": 元数据, "score": 得分}]
            - 启用重排时，score 为 CrossEncoder 打分（可正可负，越大越相关）；
            - 降级模式下，score 为向量相似度（0~1 之间）。
        """
        # ---------- 第一阶段：向量召回 ----------
        query_embedding = self.embedding_model.encode(question)[0]  # 取第 0 条
        candidates = self.vector_store.query(query_embedding, top_k=self.recall_top_k)
        if not candidates:
            print("[提示] 向量库中没有检索到任何候选内容。")
            return []

        # ---------- 第二阶段：重排（带降级） ----------
        reranker = self._get_reranker()
        if reranker is None:
            # 降级：直接按向量相似度取前 N 条
            return candidates[: self.rerank_top_n]

        # CrossEncoder 的输入是 [问题, 候选文本] 组成的句对
        pairs = [[question, c["content"]] for c in candidates]
        try:
            scores = reranker.predict(pairs)
        except Exception as exc:
            print(f"[提示] 重排打分失败（{exc}），回退为按向量相似度取前 {self.rerank_top_n} 条。")
            return candidates[: self.rerank_top_n]

        # 用重排得分覆盖召回得分，并按新得分从高到低排序
        for candidate, score in zip(candidates, scores):
            candidate["score"] = float(score)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: self.rerank_top_n]

    # ------------------------------------------------------------------ #
    # 内部：重排模型懒加载与降级
    # ------------------------------------------------------------------ #
    def _get_reranker(self):
        """加载 CrossEncoder 重排模型；不可用时打印提示并返回 None。"""
        if self._reranker_checked:
            return self._reranker
        self._reranker_checked = True

        if not self.use_reranker:
            print(f"[提示] 重排功能已被关闭，直接取召回结果的前 {self.rerank_top_n} 条。")
            return None

        try:
            from sentence_transformers import CrossEncoder
            print(f"[信息] 正在加载重排模型：{self.reranker_name}（首次运行需联网下载）……")
            self._reranker = CrossEncoder(self.reranker_name, max_length=512)
            print("[信息] 重排模型加载完成。")
        except Exception as exc:
            # 网络不通、显存不足等都会走到这里 —— 降级而非崩溃
            print(f"[提示] 重排模型不可用（{exc}），将直接使用召回结果的前 {self.rerank_top_n} 条。")
            self._reranker = None
        return self._reranker


if __name__ == "__main__":
    # 使用示例（需要先构建好向量索引）：
    #     embedding = EmbeddingModel()
    #     store = VectorStore()
    #     retriever = Retriever(embedding, store)
    #     for item in retriever.retrieve("离心泵轴承温度上限是多少？"):
    #         print(item["score"], item["metadata"], item["content"][:50])
    print("retriever.py 是库模块，请通过 app.py 使用完整流程。")
