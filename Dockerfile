# 使用官方 Python 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 复制所有项目文件
COPY . .

# 暴露端口（docker-compose 会覆盖）
EXPOSE 8000 7860

# 默认命令由 docker-compose 覆盖
CMD ["python", "main.py"]