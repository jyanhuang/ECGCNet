import matplotlib.pyplot as plt

# 数据
time_ms = [796, 745, 821, 834]
accuracy_percent = [98.41, 98.35, 98.00, 81.58]

# 设置图形大小
plt.figure(figsize=(10, 6))

# 画柱状图，每个柱状图用不同颜色表示
bars = plt.bar(time_ms, accuracy_percent, color=['#BF6874', '#965A97', '#8EB05A', '#F5BF4C'],width=2.9)

# 设置横纵坐标标签和标题
plt.xlabel('Time (ms)')
plt.ylabel('Overall Accuracy (%)')

# 设置纵坐标范围和刻度
plt.ylim(80, 100)


# 设置横坐标范围和刻度
plt.xlim(740, 850)
plt.xticks(range(740, 850, 20))

# 添加图例
legend_labels = ['w/o quantification','quantification Conv1', 'quantification Conv2', 'quantification Conv1 + Conv2']
plt.legend(bars, legend_labels,bbox_to_anchor=(0.45, 0.78))
# 添加横向线
for y in range(70, 100, 5):
    plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
# 显示图形
plt.savefig('quantification.png', dpi=1000, bbox_inches='tight')
plt.show()
