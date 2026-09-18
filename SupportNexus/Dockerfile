# 选择一个带 Python 3.12、但不含多余工具的基础环境。
FROM python:3.12-slim

# 让容器日志立即输出到 Docker，并避免在镜像里生成 .pyc 文件。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app

# 后续命令及应用启动时的默认工作目录。
WORKDIR /app

# 先只复制依赖清单并安装依赖。只要 requirements.txt 没变，Docker 就能复用这一层缓存。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 创建一个普通用户和可写的主目录；Chroma 首次下载本地 Embedding 模型时会使用 HOME。
RUN addgroup --system app \
    && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /home/app \
    && chown -R app:app /home/app

# 复制后端源码、Skills 与示例知识文件，并交给普通用户读取和写入运行期目录。
COPY --chown=app:app . ./

USER app

# 这是给阅读者和编排工具看的文档性声明；实际端口映射会在 Compose 中定义。
EXPOSE 8000

# 容器启动时执行 FastAPI 服务。开发热重载会在后续 Compose 开发配置中按需开启。
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
