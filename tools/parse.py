import re

# 表格字符串
table_str = """
|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.3430|±  |0.0139|
|             |       |none  |     0|acc_norm|↑  |0.3729|±  |0.0141|
|arc_easy     |      1|none  |     0|acc     |↑  |0.6545|±  |0.0098|
|             |       |none  |     0|acc_norm|↑  |0.6267|±  |0.0099|
|boolq        |      2|none  |     0|acc     |↑  |0.7774|±  |0.0073|
|hellaswag    |      1|none  |     0|acc     |↑  |0.4356|±  |0.0049|
|             |       |none  |     0|acc_norm|↑  |0.5901|±  |0.0049|
|openbookqa   |      1|none  |     0|acc     |↑  |0.2220|±  |0.0186|
|             |       |none  |     0|acc_norm|↑  |0.3320|±  |0.0211|
|rte          |      1|none  |     0|acc     |↑  |0.6173|±  |0.0293|
|winogrande   |      1|none  |     0|acc     |↑  |0.6630|±  |0.0133|
"""

# 定义一个函数来解析表格字符串
def parse_table(table_str: str):
    # 使用正则表达式分割字符串，去除多余的空白字符
    lines = [re.split(r'\s*\|\s*', line.strip()) for line in table_str.strip().split('\n')]
    return lines

# 解析表格字符串
parsed_table = parse_table(table_str)

# print(parsed_table)

# 提取所有 "acc" 的值
acc_values = [row[7] for row in parsed_table if row[5] == 'acc']

print(acc_values)

acc_values = list(map(float, acc_values))
print(sum(acc_values) / len(acc_values))