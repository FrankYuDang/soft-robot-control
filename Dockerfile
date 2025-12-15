# 1. Base Image: 使用官方轻量级 Python 3.9 镜像
# 这里的 "slim" 版本比完整版小很多，适合生产环境
FROM python:3.9-slim

# 2. Set Working Directory: 在容器内部创建一个工作目录
WORKDIR /app

# 3. Environment Variables: 设置环境变量
# PYTHONDONTWRITEBYTECODE: 不生成 .pyc 文件 (节省空间)
# PYTHONUNBUFFERED: 这里的日志立即打印，不要缓存 (方便我们在 AWS Logs 里看)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. Install Dependencies: 先只复制 requirements.txt
# 为什么？因为 Docker 有 Layer Caching 机制。
# 只要你的依赖包没变，这一步就会被缓存，下次 build 会飞快。
COPY requirements.txt .

# 安装依赖，--no-cache-dir 减小镜像体积
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy Code: 把剩下的代码复制进去
# 注意：我们通常会用 .dockerignore 忽略掉 data 文件夹，防止镜像过大
COPY . .

# 6. Entrypoint: 容器启动时默认执行的命令
# 这里默认执行训练脚本，你也可以改成 bash 进去调试
# CMD ["python", "scripts/train.py"]

# --- 修改重点 START ---

# 1. Expose Port: 声明容器内部开放 8000 端口
EXPOSE 8000

# 2. CMD: 启动 Uvicorn 服务器
# --host 0.0.0.0:这是必须的！意味着监听所有网卡，允许外部访问。
# --port 8000: 监听 8000 端口
# src.app:app: 指向 src/app.py 文件里的 app 对象
# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
# --- 修改重点 END ---