# RVC 语音克隆项目

本项目使用 RVC (Retrieval-based Voice Conversion) 技术，通过一个参考音频和一个文本文件，生成用参考音频音色朗读该文本的语音。

项目已完全容器化，依赖 Docker 进行环境隔离和部署。

## 环境准备

请确保你的系统已经安装了 [Docker](https://www.docker.com/products/docker-desktop/)。

如果你的计算机拥有支持 CUDA 的 NVIDIA 显卡，请确保 Docker 已配置为使用 GPU，这将极大提升处理速度。

## 文件结构

在运行前，请确保你的项目包含以下结构：

```
.
├── io/
│   ├── input/
│   │   ├── input.wav  <-- 你的参考音频
│   │   └── input.txt  <-- 你的输入文本
│   └── output/
│       └── (这里将存放生成的音频)
├── models_zh/
│   └── (存放所需的模型文件)
├── RVC.py
├── Dockerfile
├── requirements.txt
└── README.md
```

-   `io/input/`: 存放所有输入文件。
    -   `audio/`: 存放参考音频（如 `.wav` 文件）。
    -   `text/`: 存放包含目标朗读内容的文本文件（`.txt`，请使用 UTF-8 编码）。
-   `io/output/`: 用于存放程序生成的音频文件。
-   `models_zh/`: 存放运行 RVC 所需的预训练模型。

## 使用步骤

### 第一步：构建 Docker 镜像

首次运行或在修改了项目代码（如 `RVC.py`）之后，你需要构建 Docker 镜像。在项目的根目录打开终端（如 PowerShell, Bash 等），并执行以下命令：

```bash
docker build -t rvc-app .
```

此命令会根据 `Dockerfile` 创建一个名为 `rvc-app` 的本地镜像。

### 第二步：准备输入文件

1.  将你的参考音频文件（*.wav）放入 `io/input/` 目录。
2.  将你的文本文件（*.txt）放入 `io/input/` 目录。

### 第三步：运行容器

使用以下命令来启动容器并开始语音合成。此命令会将本地的 `io` 和 `models_zh` 文件夹挂载到容器内部，使得容器可以访问你的文件和模型。

**请根据你的文件名修改命令末尾的参数。**

```powershell
打开io和models_zh的上级目录
# 在 PowerShell 中运行
docker run --rm --gpus all`
  -v ${PWD}/io:/app/io `
  -v ${PWD}/models_zh:/app/models_zh `
  rvc-app `
  --ref /app/io/input/zhb.wav `
  --text-file /app/io/input/test.txt `
  --out /app/io/output/generated_audio.wav
```
```
docker run --rm --gpus all -v ./io:/io -v ./models_zh:/app/models_zh rvc-app --ref /io/input/zhb.wav --text-file /io/input/test.txt --out /io/output/generated_audio.wav
```

**命令参数解释:**

-   `--rm`: 容器运行结束后自动删除，保持环境整洁。
-   `-v ${PWD}/io:/app/io`: 将本地的 `io` 文件夹挂载到容器的 `/app/io` 路径。
-   `-v ${PWD}/models_zh:/app/models_zh`: 将本地的 `models_zh` 文件夹挂载到容器的 `/app/models_zh` 路径。
-   `rvc-app`: 要运行的镜像名称。
-   `--ref`: 指定参考音频在**容器内**的路径。
-   `--text-file`: 指定输入文本在**容器内**的路径。
-   `--out`: 指定输出音频在**容器内**的路径和文件名。

### 第四步：查看结果

命令执行完毕后，你可以在本地的 `io/output/` 文件夹中找到生成的音频文件（在上面的例子中是 `generated_audio.wav`）。

## 开发流程须知

-   **当你修改了 Python 代码** (例如 `RVC.py`): 你**必须**重新运行 `docker build` 命令来更新镜像。
-   **当你只是更换了输入文件或模型** (即 `io` 或 `models_zh` 文件夹中的内容): 你**无需**重新构建镜像，直接运行 `docker run` 命令即可。
```