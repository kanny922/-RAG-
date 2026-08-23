# -*- coding: utf-8 -*-
"""
vector_store.py —— 文本向量化（Embedding）与向量数据库模块
==========================================================

本模块包含两个类，对应 RAG 中"把知识变成可检索的向量"的两个环节：

1. :class:`EmbeddingModel` —— 文本向量化
   使用 sentence-transformers 加载中文向量模型 BAAI/bge-small-zh-v1.5，
   把任意文本编码成一个 512 维的浮点向量。
   编码时开启 normalize_embeddings=True（向量单位化），
   这样向量内积 == 余弦相似度，检索打分更直观。
   模型文件已内置到项目 models/bge-small-zh-v1.5/ 目录，默认从本地加载。

2. :class:`VectorStore` —— 向量持久化与相似度检索
   基于 ChromaDB 的 PersistentClient（数据落盘，重启后不丢失），
   提供 入库 / 检索 / 统计 / 清空 四个核心方法。

教学要点：
- 模型文件已随项目携带，默认不联网下载；
- ChromaDB 默认使用余弦距离（cosine distance），
  本模块在查询时把它换算成"相似度得分"：score = 1 - distance。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

from document_loader import Document

# 本地 Embedding 模型路径（已内置到项目 models/ 目录，无需联网下载）
LOCAL_EMBEDDING_MODEL_PATH = Path(__file__).parent / "models" / "bge-small-zh-v1.5"

# 单个向量或向量列表的类型别名（方便阅读）
VectorLike = Union[Sequence[float], np.ndarray]


class EmbeddingModel:
    """中文文本向量化模型（BGE 中文系列）。

    Args:
        model_name: HuggingFace 模型名，默认 BAAI/bge-small-zh-v1.5。
                    若本地已缓存同名模型，会自动使用本地路径，无需联网。
        device:     运行设备，如 "cpu" / "cuda"；None 表示自动选择。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "缺少 sentence-transformers 依赖，请先执行：pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        # 如果存在本地缓存且使用默认模型，优先加载本地版本，避免网络下载
        effective_model = model_name
        if model_name == "BAAI/bge-small-zh-v1.5" and LOCAL_EMBEDDING_MODEL_PATH.exists():
            effective_model = str(LOCAL_EMBEDDING_MODEL_PATH)

        print(f"[信息] 正在加载 Embedding 模型：{effective_model}……")
        try:
            self.model = SentenceTransformer(effective_model, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding 模型加载失败：{exc}\n"
                "（请检查 models/bge-small-zh-v1.5/ 目录下模型文件是否完整）"
            ) from exc
        print("[信息] Embedding 模型加载完成。")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """把一段或多段文本编码成向量。

        Args:
            texts: 单个字符串，或字符串列表。

        Returns:
            numpy 数组：单条输入形状为 (1, 维度)，多条输入为 (N, 维度)。
            向量已做 L2 归一化，可直接用内积表示余弦相似度。
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, 0))
        return self.model.encode(
            texts,
            normalize_embeddings=True,  # 单位化：内积即余弦相似度
            show_progress_bar=False,
        )


class VectorStore:
    """基于 ChromaDB 的持久化向量库。

    Args:
        persist_directory: 数据落盘目录（重启后索引仍在，避免重复入库）。
        collection_name:   集合名，类似数据库中的"表"。
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "industrial_docs",
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("缺少 chromadb 依赖，请先执行：pip install chromadb") from exc

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # PersistentClient：数据写入本地磁盘，程序重启后可直接复用
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """获取或创建集合，并指定使用余弦距离做相似度计算。"""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # 向量索引使用余弦距离
        )

    # ------------------------------------------------------------------ #
    # 入库
    # ------------------------------------------------------------------ #
    def add_documents(self, chunks: List[Document], embeddings: np.ndarray) -> None:
        """把文本块连同其向量、metadata 一起写入向量库。

        Args:
            chunks:     TextSplitter 切分出的 Document 块列表。
            embeddings: 与 chunks 一一对应的向量（形状 (N, 维度)）。
        """
        if not chunks:
            print("[提示] 没有需要入库的文本块。")
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"文本块数量（{len(chunks)}）与向量数量（{len(embeddings)}）不一致！"
            )

        # 用 chunk_id 生成全局唯一主键，保证重复入库时幂等（同 id 会报错，先清再加）
        ids = [f"chunk_{c.metadata.get('chunk_id', i)}" for i, c in enumerate(chunks)]
        documents = [c.page_content for c in chunks]
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]
        vectors = [np.asarray(e, dtype=float).tolist() for e in embeddings]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )
        print(f"[信息] 已向向量库写入 {len(chunks)} 个文本块。")

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def query(self, embedding: VectorLike, top_k: int = 20) -> List[Dict]:
        """按向量相似度检索 Top-K 最相关的文本块。

        Args:
            embedding: 查询向量（由 EmbeddingModel.encode 产生）。
            top_k:     返回的最相关结果条数。

        Returns:
            结构化结果列表，按相关度从高到低排序：
            [{"content": 文本块内容, "metadata": 元数据, "score": 相似度得分}]
            score 由余弦距离换算而来：score = 1 - distance，越大越相关。
        """
        if self.count() == 0:
            return []

        vector = np.asarray(embedding, dtype=float).tolist()
        # top_k 不能超过库中总量，否则 ChromaDB 会报错
        n_results = max(1, min(top_k, self.count()))
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB 的返回结构是"按查询分组"的二维列表，这里只有一个查询，取第 0 组
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: List[Dict] = []
        for content, metadata, distance in zip(documents, metadatas, distances):
            output.append({
                "content": content,
                "metadata": metadata or {},
                "score": 1.0 - float(distance),  # 余弦距离 → 相似度得分
            })
        return output

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        """返回向量库中当前的文本块总数。"""
        return self.collection.count()

    def clear(self) -> None:
        """清空整个集合（重建索引前调用）。"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # 集合不存在时忽略
        self.collection = self._get_or_create_collection()
        print("[信息] 向量库已清空。")

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_metadata(metadata: Dict) -> Dict:
        """清洗 metadata：ChromaDB 只接受 str / int / float / bool 类型的值。"""
        cleaned = {}
        for key, value in (metadata or {}).items():
            if value is None:
                cleaned[key] = ""  # None 不允许，用空字符串代替
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)  # 其他类型统一转成字符串
        return cleaned


if __name__ == "__main__":
    # 简单自测：编码两段文本并计算余弦相似度（需要先能加载模型）
    model = EmbeddingModel()
    vecs = model.encode(["离心泵轴承温度不得超过75摄氏度", "今天天气怎么样"])
    similarity = float(np.dot(vecs[0], vecs[1]))  # 单位化后内积 == 余弦相似度
    print(f"两句话的余弦相似度：{similarity:.4f}")
