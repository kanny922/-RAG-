# 核心模块说明
## 分层设计的优势：职责单一、可独立调试、可替换（换模型只改 llm_client.py，换向量库只改 vector_store.py）。
## 1.1 document_loader.py：文档统一解析
### （1）类：Document
frozen dataclass，统一的文档抽象：
### （2）类：DocumentLoader
load(path)：按扩展名自动路由——.pdf 走 PyMuPDF 逐页提取（无文本页自动尝试 OCR）、.docx 走 python-docx 段落提取、.txt 直接读取；
load_directory(dir_path)：批量扫描目录下全部支持格式的文档；
OCR 分支：pdf2image 转图 + RapidOCR 识别；OCR 依赖缺失或失败时打印中文警告并跳过，不中断整体流程。
## 1.2 text_splitter.py：中文优先递归分块
类：TextSplitter
初始化参数 chunk_size=400、chunk_overlap=80；
split_documents(documents)：按分隔符优先级递归切分，保留并传递 metadata，附加 chunk_id；
合并相邻碎块时保证块间重叠，避免关键句子被拦腰切断。
## 1.3 vector_store.py：Embedding + 向量库
### （1）类：EmbeddingModel
基于 sentence-transformers 加载 BGE 模型，model_name 可参数化；
encode(texts) 批量编码，normalize_embeddings=True 输出归一化向量。
### （2）类：VectorStore
基于 chromadb PersistentClient 的持久化向量库：
## 1.4 retriever.py：两段式检索
### 类：Retriever（recall_top_k=20，rerank_top_n=5）
执行”先召回、后精排”两段式检索：
召回：将问题编码为向量，在向量库中取余弦相似度 Top-20——速度快，宁可多不可少；
重排：用 CrossEncoder（BAAI/bge-reranker-base）把”问题+候选块”逐条精打分，保留 Top-5——慢但准；
重排模型不可用（离线/未下载）时自动降级为直接取召回前 5 条并打印提示；
返回结构化结果 [{content, metadata, score}]，供 Prompt 组装与溯源展示。
## 1.5 prompt_template.py：工业领域 Prompt 模板
SYSTEM_PROMPT：工业知识库问答助手角色设定，四条硬约束——仅依据给定上下文回答；无法回答时明确声明”根据现有资料无法回答”；答案中用 [1][2] 标注引用来源；涉及设备操作必须提示遵守安全规程；
build_prompt(question, contexts, history)：把检索上下文按 [1][2]… 编号（附来源文件与页码），连同对话历史与用户问题组装为完整消息序列；
REWRITE_PROMPT：多轮场景下，把含指代的追问改写为独立完整的检索问题。
## 1.6 llm_client.py：LLM 封装与多轮对话
### （1）类：BaseLLM / OpenAILLM / OllamaLLM（默认使用 DeepSeek）
BaseLLM 定义统一抽象接口 chat(messages, temperature, stream)；
OpenAILLM 从环境变量或 llm_client.py 中的 LOCAL_DEEPSEEK_API_KEY 读取 api_key / base_url / model，支持流式输出；
OllamaLLM 为本地模型备选实现（默认 qwen2.5:7b，数据不出厂）；
建议问答场景 temperature 取 0.1~0.3，保证回答稳定不发散。
### （2）类：ChatHistory
多轮对话历史管理：固定窗口保存最近 max_turns=5 轮问答（一轮 = 一次提问 + 一次回答），超出自动丢弃最早轮次；to_messages() 输出标准消息列表，clear() 清空。
## 1.7 app.py：Gradio 前端与答案溯源
启动时自动初始化知识库：检测到持久化索引则直接复用，否则扫描 docs/ 建库；
界面左栏：文档上传、“重建索引”按钮、知识库统计（文档数 / 块数）；
界面右栏：Chatbot 聊天窗口 + “引用来源”折叠面板；
问答流程：历史改写 → 两段式检索 → Prompt 组装 → LLM 流式生成；
溯源高亮：每条来源展示文档名、页码、相似度得分与原文片段，被答案引用的句子用【】标记（highlight_cited + format_sources）。
# 2. 运行与使用
## 2.1 启动系统
```bash
conda activate doc-qa
python app.py
```
浏览器访问 http://localhost:7860 。如需局域网共享，app.py 已配置 server_name="0.0.0.0"，同事可通过 http://<你的IP>:7860 访问。
## 2.2 首次建库
将文档放入 docs/（或在界面上传）；
首次启动自动建库，或点击”重建索引”手动触发；
建库完成后，索引持久化在 chroma_db/，后续启动秒开。
## 2.3 日常问答
在输入框提问，如”该设备的润滑周期是多少？“；
系统流式输出回答，关键结论后附 [1][2] 引用编号；
展开”引用来源”面板核对出处：文档名、页码、得分、高亮原文一目了然；
支持追问，如”那它的更换步骤呢？“——系统自动结合历史改写为独立问题再检索；
点击”清空对话”重置多轮历史。
## 2.4 使用本地大模型（可选）
数据敏感场景可完全离线运行：
## 安装并启动 Ollama，拉取模型
ollama pull qwen2.5:7b
## 在 app.py 中将 OpenAILLM 替换为 OllamaLLM（或修改 LOCAL_DEEPSEEK_API_KEY） 即可（接口一致，一行切换）
## 3. 输出与评估
## 3.1 回答质量判断
系统回答质量可从以下三个维度判断，这些维度由 prompt_template.py 中的 SYSTEM_PROMPT 约束：
•	可溯源：每个关键结论都应带有 [1][2] 等引用编号，能在“溯源来源”面板找到对应原文。
•	不离谱：当参考资料不足以回答问题时，模型必须明确回答“根据现有资料无法回答”，而不是编造信息。
•	安全提示：涉及设备操作、检修、工艺参数调整等可能影响人身安全的内容，回答末尾会提示遵守安全规程。
## 3.2 检索质量调优
检索质量由 text_splitter.py、retriever.py 中的参数共同决定，可根据文档特点与问答效果调整：
•	文本分块：chunk_size=400 / chunk_overlap=80。块太大语义分散，块太小上下文缺失；工业文档可适当增大 chunk_size 以保留完整段落。
•	召回与重排：recall_top_k=20 / rerank_top_n=5。先用向量检索快速召回 20 条候选，再用 CrossEncoder 精选为 5 条交给大模型。
•	重排条数控制：reranker 只对 Top-20 精排，兼顾精度与速度；重排模型不可用或禁用时，自动降级为直接取召回前 5 条。
## 3.3 响应延迟优化
系统通过以下机制降低启动与问答延迟：
•	索引持久化：向量库写入 chroma_db/ 目录，重启后秒开，避免每次启动都重复解析和 Embedding。
•	批量向量化：vector_store.py 在入库时批量 encode 文本块，避免逐条调用模型。
•	流式生成：llm_client.py 支持 stream=True，边生成边显示，降低首字延迟感知。
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
```
industrial-doc-qa/
├── document_loader.py     # 文档解析：PDF / Word / 扫描件 OCR → Document 列表
├── text_splitter.py       # 文本分块：中文优先递归分块 → Chunk 列表
├── vector_store.py        # 向量库：BGE Embedding + Chroma 持久化存储
├── retriever.py           # 检索器：向量召回 Top-20 + 重排序精选 Top-5
├── prompt_template.py     # Prompt 模板：角色设定 + 上下文组装 + 引用约束
├── llm_client.py          # LLM 封装：统一接口 + 多轮对话历史管理
├── app.py                 # Gradio 前端：聊天界面 + 溯源高亮 + 知识库管理
├── requirements.txt       # Python 依赖清单
├── docs/                  # 企业文档存放目录（运行后自动创建）
└── chroma_db/             # 向量库持久化目录（首次建库后自动生成）
```

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
