# 定义范围边界
bins = [1, 100, 200, 300, 400, 500, 600, 700, float('inf')]
bin_labels = ['1–100', '100–200', '200–300', '300–400', '400–500', '500–600', '600–700', '700+']

# 初始化计数器
length_counts = {label: 0 for label in bin_labels}

# 读取文件并统计序列长度
file_path = "../GraphPPIS/Test_315-28.fa"
file_path = "../GraphPPIS/UBtest_31-6.fa"
sequence_lengths = []

with open(file_path, 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 3):  # 每三行一组
        if i + 2 < len(lines):  # 确保有三行
            binary_line = lines[i + 2].strip()  # 第三行是二进制行
            length = len(binary_line)  # 二进制行的长度即序列长度
            sequence_lengths.append(length)

# 将长度划分到各个范围
for length in sequence_lengths:
    for j in range(len(bins) - 1):
        if bins[j] <= length < bins[j + 1]:
            length_counts[bin_labels[j]] += 1
            break

# 输出结果
print("序列长度分布统计：")
for label, count in length_counts.items():
    print(f"{label}: {count} 个序列")

# 可选：打印总序列数以验证
total_sequences = len(sequence_lengths)
print(f"\n总序列数: {total_sequences}")
