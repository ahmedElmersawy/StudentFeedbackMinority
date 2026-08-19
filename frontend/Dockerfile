# ── Build stage ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci --prefer-offline

COPY frontend/ .

# VITE_API_URL="" → nginx proxies API calls; set to full URL for split deploy
ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# ── Runtime: nginx serving static files ──────────────────────────────────────
FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

# nginx/nginx.conf (default, full config replacing /etc/nginx/nginx.conf)
# proxies /api to a "backend" upstream — only resolvable under Docker
# Compose's shared network. For a split deploy where the frontend calls
# VITE_API_URL directly (e.g. separate Railway services), override both
# build args to use frontend/nginx.conf (a bare server block dropped into
# conf.d, stock nginx.conf left in place) — it just serves the static SPA.
ARG NGINX_CONF=nginx/nginx.conf
ARG NGINX_DEST=/etc/nginx/nginx.conf
COPY ${NGINX_CONF} ${NGINX_DEST}

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
