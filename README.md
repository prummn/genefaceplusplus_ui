# 实时语音克隆与数字人对话系统 (Real-time Voice Cloning & Digital Human Dialogue System)

## 📖 项目简介

本项目旨在构建一个从底层模型训练到上层实时交互的全栈数字人系统。项目逻辑层层递进，分为三个核心阶段：

1.  **模型训练 (Model Training)**: 构建专属的数字人形象基座。
2.  **视频生成 (Video Generation)**: 基于音频驱动静态形象生成动态说话视频。
3.  **人机对话 (Human-Computer Interaction)**: 集成语音识别、大语言模型与语音克隆，实现与数字人的实时语音交流。


---

## 🚀 核心功能模块

### 第一阶段：模型训练 (Model Training)

此模块负责为数字人提供“躯体”和“形象”，完全基于 **GeneFace++** 实现。
* **自定义形象**: 支持用户上传目标人物的视频素材。
* **任务调度**: 后端 (`backend/model_trainer.py`) 集成了训练任务的参数配置（如 max_steps、GPU 选择）与状态监控逻辑。
* **核心引擎**: 使用 **GeneFace++** Docker 镜像进行高保真数字人模型的训练。

### 第二阶段：视频生成 (Video Generation)

此模块负责让数字人“动起来”。
* **音频驱动**: 接收一段音频（无论是克隆生成的还是上传的），驱动训练好的数字人模型生成口型同步的说话视频。
* **推理引擎**: 基于 `backend/video_generator.py`，调用 **GeneFace++** 预训练模型进行推理渲染。

### 第三阶段：人机对话 (Human-Computer Interaction)

此模块赋予数字人“灵魂”与“声音”，实现了完整的语音交互闭环。

#### 1. 智能语音克隆 (Voice Cloning)
* **RVC (Retrieval-based Voice Conversion)**: 
    * 作为本项目的核心语音合成引擎，RVC 提供了高质量、低延迟的变声与克隆能力。
    * **运行方式**: 通过 Docker 容器化运行。后端会自动启动容器进行推理，确保环境一致性。
    * **处理流程**: `chat_engine` 会自动将长文本切分为短句，**循环启动 Docker 容器**分别克隆每一句话，最后将生成的音频片段**拼接**成完整的语音文件。

#### 2. 多模态大模型集成 (LLM Integration)
* **多模型支持**: 系统集成了 **ZhipuAI (GLM-4/GLM-4.5)** 和 **Google Gemini (Pro/Flash)** API。
* **智能交互**: 
    * 支持多轮对话记忆，能够联系上下文进行交流。
    * 通过精心设计的 Prompt（系统提示词），控制 AI 的回复长度与风格，使其更适合口语化表达。

#### 3. 自动化 ASR 流程 (Auto Speech Recognition)
* **全自动转录**: 用户录音上传后，后端自动调用 ASR 接口将其转录为文本。
* **智能字幕生成**: 当用户上传参考音频时，系统会自动检测是否存在对应的字幕文件；若缺失，则自动调用 ASR 生成字幕，实现了“上传即用”的便捷体验。

---

## 🏗️ 人机对话系统架构

该部分展示了用户与系统进行实时语音交互的完整数据流：

```mermaid
graph TD
    User[用户 (Web前端)] -->|1. 录音/上传参考音频| Server[Flask 主服务 (app.py)]
    Server -->|2. 音频转码 & 存储| IO[IO 文件系统 (io/)]
    
    subgraph "Backend Core (backend/)"
        IO -->|3. 读取用户录音| Pipeline[llm_asr_pipeline.py]
        Pipeline -->|4. 调用 ASR API| Text[生成用户文本]
        Text -->|5. 发送历史对话| LLM[大语言模型 (GLM/Gemini)]
        LLM -->|6. 生成回复文本| Response[AI 回复]
        
        Response -->|7. 文本处理| ChatEngine[chat_engine.py]
    end
    
    subgraph "Voice Cloning Engine (RVC Docker)"
        ChatEngine -->|8. 切分长文本| TextChunks[文本片段]
        TextChunks -->|9. 循环调用 Docker| Docker_Run[启动 RVC 容器]
        Docker_Run -->|10. 生成片段音频| AudioChunks[音频片段]
        AudioChunks -->|11. 拼接| Final_Audio[最终音频 (io/output)]
    end
    
    Final_Audio -->|12. 返回路径| Server
    Server -->|13. 播放/驱动视频| User
```

---

## 📂 目录结构与后端说明

```text
genefaceplusplus_ui/
├── app.py                  # Flask 主入口，处理 HTTP 请求与路由
├── .env                    # 环境变量配置文件 (API Keys)
├── backend/                # 核心业务逻辑模块
│   ├── chat_engine.py      # [人机对话] 主引擎。负责切分长文本、循环调用 RVC Docker 容器并拼接音频。
│   ├── llm_asr_pipeline.py # [人机对话] ASR 与 LLM 封装。负责调用智谱/Gemini API。
│   ├── model_trainer.py    # [模型训练] 负责调度 GeneFace++ 的训练任务。
│   └── video_generator.py  # [视频生成] 负责调用 GeneFace++ 进行视频生成。
├── geneface/               # [待补充] GeneFace++ 核心代码与模型文件
│   └── ...                 # (请相关负责人在此处补充 GeneFace 文件夹的具体结构与说明)
├── io/                     # 数据存储中心 (位于项目根目录)
│   ├── input/              # 存放参考音频与文本
│   │   ├── audio/          # 参考音频 (.wav)
│   │   └── text/           # 参考文本 (.txt)
│   ├── history/            # 对话历史与日志
│   ├── output/             # 合成结果
│   └── temp/               # 临时文件
├── RVC/                    # RVC 模型相关文件
│   ├── models_zh/          # RVC 预训练模型权重 (需下载)
│   ├── Dockerfile          # RVC 镜像构建文件
│   └── ...
├── static/                 # 前端资源 (JS, CSS)
└── templates/              # HTML 页面模版
```

---

## 🐳 Docker 镜像说明与运行

本项目采用双镜像架构，以隔离不同模块的复杂依赖。

### 镜像 1: 模型训练与视频生成 (GeneFace++)
*(此处预留给其他成员填写 GeneFace++ 镜像的详细说明)*

### 镜像 2: 语音克隆 (RVC)
* **镜像名称**: `rvc-app`
* **功能**: 负责 RVC 语音克隆推理。
* **Dockerfile 位置**: `RVC/Dockerfile`
* **构建命令**:
  请进入 `RVC` 目录执行构建：
  ```bash
  cd RVC
  docker build -t rvc-app .
  ```

#### RVC 容器启动流程详解
后端 `chat_engine.py` 会自动执行类似以下的命令来启动一次性容器进行推理。对于长文本，此过程会**循环执行**多次：

```bash
docker run --rm --gpus all \
  -v ./io:/io \
  -v ./RVC/models_zh:/app/models_zh \
  rvc-app \
  --ref /io/input/audio/ref_audio.wav \
  --text-file /io/temp/chunk_0.txt \
  --out /io/temp/chunk_0.wav
```

**逻辑说明：**
1.  **切分**: 将 LLM 返回的长文本按标点切分为多个短句。
2.  **循环**: 针对每个短句，启动一个新的 `rvc-app` 容器进行推理。
3.  **拼接**: 所有容器运行完毕后，使用 `pydub` 将生成的音频片段拼接成完整的输出文件。

---

## 🚀 快速启动指南

### 1. 环境准备
确保本地安装了 **Python 3.9+** (Conda)、**CUDA** 和 **FFmpeg**。同时确保 **Docker** 已安装并支持 NVIDIA GPU (`nvidia-container-toolkit`)。

```bash
# Ubuntu/Debian 安装系统依赖
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
```

### 2. 安装依赖

```bash
# 创建环境
conda create -n voice_chat python=3.9
conda activate voice_chat

# 安装核心依赖
pip install flask pydub requests python-dotenv zhipuai werkzeug
```

### 3. 配置与启动

1.  **配置 API Key**: 在 `.env` 文件中填入智谱或 Gemini 的 Key。
2.  **准备模型**: 
    * 需要将 RVC 预训练模型放入 `RVC/models_zh/` 目录下。
    * **模型下载链接**: [百度网盘](https://pan.baidu.com/s/1kEEMK-O1bfIRVgjLWC_Z9Q?pwd=cegc) (提取码: `cegc`)
    * 下载 `models_zh.zip` 后，请解压并将内容覆盖至 `RVC/models_zh/` 文件夹。
3.  **启动服务**:
    ```bash
    python app.py
    ```
    * 服务端口: **5000**
    * 访问地址: `http://127.0.0.1:5000` (或服务器 IP)

---

## 🧪 探索与实验

### 关于 CosyVoice
我们在开发过程中尝试引入了 **CosyVoice** (阿里的零样本语音克隆模型) 作为第二种克隆方案。
* **现状**: 代码库中保留了 CosyVoice 的相关接口与逻辑。
* **说明**: 鉴于 CosyVoice 目前对计算资源要求较高且在部分环境下推理速度不如预期，本项目目前的生产环境默认使用 **RVC** 进行极其稳定的实时克隆。CosyVoice 作为一个实验性功能保留，供后续研究使用。

---

## 📝 团队分工说明
* **人机对话/语音克隆**: 负责 ASR、LLM 接入、RVC Docker 化封装、长文本切分逻辑及整体 Web 交互。
* **模型训练/视频生成**: 负责 GeneFace++ 的模型训练与视频渲染逻辑 (详见 backend 对应模块)。