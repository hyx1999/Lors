#!/bin/bash

# 指定要搜索的目录
search_dir="/data2/models/hyx"

# 指定正则表达式模式
pattern="Llama-.*-hf_(wanda|sparsegpt).2of4_.*"

# 使用 find 查找符合正则表达式的目录
echo "查找并列出符合正则表达式的目录..."
find "$search_dir" -type d -regextype posix-extended -regex ".*/$pattern"

# 如果你确定要删除这些目录，可以这样做：
# 注意：在真正执行删除前，先检查输出是否正确
# read -p "你确定要删除上述目录吗？(y/N) " response
# if [[ $response =~ ^[Yy]$ ]]; then
#     echo "开始删除..."
#     # 使用 find 和 rm 删除目录
#     find "$search_dir" -type d -regextype posix-extended -regex ".*/$pattern" -exec rm -rf {} +
# else
#     echo "取消删除操作。"
# fi
