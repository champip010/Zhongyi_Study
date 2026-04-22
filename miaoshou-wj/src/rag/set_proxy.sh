#!/bin/bash

# 设置代理服务器地址和端口
PROXY_HOST="172.16.103.9"
PROXY_PORT="7897"
PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"

echo "🚀 开始设置代理环境变量..."

# 先清除所有代理设置
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

# 设置 HTTP/HTTPS 代理（标准环境变量）
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"

# 设置 NO_PROXY 排除本地地址
export NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


echo " 清除代理请使用: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy"
