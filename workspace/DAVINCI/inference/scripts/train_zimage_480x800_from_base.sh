#!/usr/bin/env bash
# 兼容旧工作区训练入口 → example/train_t2i_1w_480x800.sh
exec "$(dirname "$0")/../../example/train_t2i_1w_480x800.sh" "$@"
