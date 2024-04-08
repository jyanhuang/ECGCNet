import matplotlib.pyplot as plt

# 数据
random_seeds = [42, 72, 126, 168, 210, 252, 294, 336, 378, 420]
accuracies = [98.41, 98.65, 98.42, 98.58, 98.48, 98.60, 98.65, 98.72, 98.64, 98.16]

# 画折线图
plt.plot(random_seeds, accuracies, marker='o', linestyle='-',color='#0F81C5')


# 添加标题和标签

plt.xlabel('Random Seed')
plt.ylabel('Overall Accuracy (%)')
# 设置纵坐标范围
plt.ylim(90, 100)
# 关闭网格线
plt.grid(False)

# 添加横向线
for y in range(90, 101, 1):
    plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
# 去掉顶部和右侧的轴线和框线
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_linestyle('--')
plt.gca().spines['left'].set_visible(False)

# 显示图形

plt.savefig('seed.png', dpi=1000, bbox_inches='tight')
plt.show()
