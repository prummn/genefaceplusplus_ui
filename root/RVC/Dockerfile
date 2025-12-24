# ----------------------------------------------------
# 阶段 1: 使用一个标准的Python 3.9基础镜像
# python:3.9-slim 是一个轻量级版本，能有效减小镜像体积
# ----------------------------------------------------
FROM python:3.9-slim

# ----------------------------------------------------
# 设置一些环境变量，优化Python运行
# ----------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE 1  # 防止python生成.pyc文件
ENV PYTHONUNBUFFERED 1         # 让python的输出直接打印到终端，方便查看日志

# ----------------------------------------------------
# 在容器内部创建一个工作目录 /app
# ----------------------------------------------------
WORKDIR /app

COPY pip.conf /etc/pip.conf
# ----------------------------------------------------
# 先只复制依赖文件，并安装它们
# 这样做能利用Docker的缓存机制，如果依赖不变，后续构建会更快
# ----------------------------------------------------
COPY requirements.txt .
RUN echo 'Acquire::https::Verify-Peer "false"; Acquire::https::Verify-Host "false";' > /etc/apt/apt.conf.d/99-no-verify-ssl
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir numpy
RUN pip install torch==2.3.0 --timeout=3600 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install torchvision==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install torchaudio==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir --no-dependencies chatterbox_tts==0.1.3
RUN pip install -r requirements.txt

# ----------------------------------------------------
# 复制项目的所有代码到工作目录
# ----------------------------------------------------
COPY . .

# ----------------------------------------------------
# 容器启动时要执行的命令
# 使用 ENTRYPOINT 可以方便地向脚本传递命令行参数
# ----------------------------------------------------
ENTRYPOINT ["python", "RVC.py"]