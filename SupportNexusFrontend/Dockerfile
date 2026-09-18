# ── 构建阶段：使用 Node 将 Vue 源码编译为静态文件 ───────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app

# 先安装依赖；package-lock.json 未变化时可复用这一层缓存。
COPY package.json package-lock.json ./
RUN npm ci

# 再复制源码并生成 dist/。
COPY . ./
RUN npm run build

# ── 运行阶段：只保留静态文件与 Nginx，不携带 Node 和源码 ──────────────────────
FROM nginx:alpine AS runtime

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
