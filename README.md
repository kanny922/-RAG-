# 工业知识库RAG问答系统
针对工业文档检索慢、知识难以沉淀的痛点，基于RAG搭建**可溯源工业知识库问答系统**，支持多格式文档解析，支持离线打包交付使用。

## ✨项目实现要点
1. **文档解析模块**
搭建RAG工业知识库问答系统，支持 PDF / DOCX / TXT 多格式文档解析，将不同格式文档统一抽象为`Document`文档对象。

2. **文本切分与向量化存储**
实现**中文标点优先的递归文本分块**，分块大小400字，重叠80字；
使用`bge‑small‑zh‑v1.5`作为Embedding模型，搭配ChromaDB完成文本向量化与向量库持久化索引。

3. **两段式检索策略**
设计「向量召回 Top‑20 + BGE‑reranker 重排序精排 Top‑5」两段式检索链路，平衡检索速度与答案相关性，优化检索效果。

4. **防幻觉 & 答案溯源**
编写带防幻觉约束、支持`[1][2]`文献引用标注的Prompt模板，实现回答溯源、原文片段高亮展示，降低大模型幻觉问题。

5. **大模型统一封装与多轮对话**
封装`BaseLLM`统一调用接口，支持 DeepSeek API、Ollama本地大模型一键切换；
实现多轮对话历史窗口管理，支持上下文追问、改写问题。

6. **多端交付方案**
- Web端：基于Gradio快速搭建交互式问答Web界面
- 桌面端：使用PySide6开发GUI桌面客户端
- 离线交付：通过PyInstaller打包为exe程序，无Python环境的用户可直接运行

## 🛠 技术栈
`RAG` `Embedding` `ChromaDB` `Rerank` `Prompt工程` `多轮对话` `Gradio` `PySide6` `PyInstaller`

## 📝 学习收获
- 完整掌握RAG全链路：文档解析→文本分块→向量化存储→检索重排→Prompt约束→多轮对话→产品化打包交付
- 理解中文场景下分块策略对检索效果的影响，中文优先标点分割，合理设置块大小与重叠度，避免语义割裂
- 掌握两段式检索设计思路：粗召回扩大候选集，重排序做精度筛选，在速度与精度之间做权衡
- 通过Prompt约束+引用溯源，缓解大模型幻觉，适配工业场景对答案可靠性的要求
- 实现LLM调用层抽象封装，解耦不同大模型后端，做到API模型与本地Ollama模型无缝切换
- 学会Python项目桌面化打包，实现面向普通用户的离线交付方案

## ⚠️ 踩坑记录
> 记录项目中遇到的实际问题，面试官重点看这部分
1. 直接使用通用递归分块对工业文档效果差，长句被粗暴切断；优化为**优先中文标点符号做分割**，保留语义完整性。
2. 只做向量召回容易出现相关性差的结果，引入BGE‑reranker重排序，大幅提升候选片段质量。
3. 大模型容易编造不存在的工业资料，Prompt增加强约束，强制模型只依据检索到的文档作答，输出引用标记。
4. Ollama本地模型与API模型入参格式不一致，封装BaseLLM层统一输入输出，屏蔽底层差异。
5. PyInstaller打包exe体积巨大、容易缺依赖；需要处理模型文件路径、隐藏导入依赖，精简打包产物。

## 📂 仓库目录结构
industrial‑rag‑qa‑system/
├── docs/ # 测试工业样例文档 pdf/docx/txt
├── src/
│ ├── document_loader.py # 多格式文档解析
│ ├── chunker.py # 中文优先递归分块
│ ├── vector_store.py # ChromaDB 向量库封装
│ ├── retriever.py # 两段式召回 + rerank 重排
│ ├── llm_base.py # BaseLLM 统一接口，DeepSeek / Ollama
│ ├── prompt_template.py # 防幻觉、引用溯源 Prompt
│ ├── gradio_app.py # gradio web 问答界面
│ └── pyside6_gui.py # 桌面端 GUI 代码
├── logs/ # 实验记录日志
├── requirements.txt
└── README.md

## 🚀 运行说明
```bash
# 安装依赖
pip install -r requirements.txt

# 启动Gradio网页端
python src/gradio_app.py

# 启动PySide6桌面端
python src/pyside6_gui.py

# 打包exe
pyinstaller xxx.spec
