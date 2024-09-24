import os
import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['LoRA', 'SP-LoRA', 'SP-LoRA(GC)', 'SP-LoRA(NO)', 'SPP(GC)', 'SPP(NO)']
memory_usage = [26012, 26012, 38380, 44236, 25830, 'OOM']  # 内存占用 (MB)
training_speed = [1.13, 1.28, 1.40, 1.26, 2.20, 0.00]  # 训练速度 (s/it)

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(10, 5))

bar_width = 0.35  # 调整柱子的宽度

# 设置第一个 y 轴（内存占用）
# ax1.set_xlabel('Methods', fontsize=12)
ax1.set_ylabel('Memory Usage (GB)', fontsize=14)
ax1.set_xticks(np.arange(len(methods)))
ax1.set_xticklabels(methods, fontsize=14) # ha='right'
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# 绘制内存占用的柱状图
memory_colors = ['tab:blue' if mem != 'OOM' else 'tab:red' for mem in memory_usage]
memory_values = [mem / 1024 if isinstance(mem, (int, float)) else 49140 / 1024 for mem in memory_usage]
ax1.bar(np.arange(len(methods)) - bar_width/2, memory_values, width=bar_width, color=memory_colors, label='Memory Usage')

#  13138

# 在SPP-Naive上方添加OOM标签
ax1.text(len(methods) - 1 - bar_width/2, memory_values[-1] + 0.5, 'OOM', ha='center', va='bottom', color='tab:red', fontsize=12)

# 添加基准内存使用的横线
baseline_memory = 13138 / 1024
ax1.set_ylim(bottom=baseline_memory)
# ax1.axhline(y=baseline_memory, color='black', linestyle='--', label='Memory Usage of Model Weights')

# # 添加基准内存使用的文本
# ax1.text(len(methods) - 0.5, baseline_memory + 10, 'Llama-2-7B Weight Memory Usage: 13138 MB', color='black', fontsize=10, va='bottom')

# 创建第二个 y 轴（训练速度）
ax2 = ax1.twinx()
ax2.set_ylabel('Average time usage per batch (s)', fontsize=14)
# ax2.tick_params(axis='y', labelcolor='tab:orange')

# 绘制训练速度的柱状图
ax2.bar(np.arange(len(methods)) + bar_width/2, training_speed, width=bar_width, color='tab:orange', alpha=0.7, label='Time usage')

# 调整图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# 显示图形
plt.tight_layout()
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
plt.savefig("images/mem_speed.png")
