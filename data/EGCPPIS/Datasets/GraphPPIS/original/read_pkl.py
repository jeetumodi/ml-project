import pickle

# 指定 pkl 文件路径
file_path = "UBtest_31-6.pkl"
file_path = "Test_315-28.pkl"

# 以二进制读取模式打开文件并加载数据
with open(file_path, "rb") as f:
    data = pickle.load(f)

# # 查看加载的内容
# print("文件内容类型:", type(data))
# print("文件内容:", data)

# 转换函数
def convert_to_fasta(data_dict):
    result = []
    for key, value in data_dict.items():
        sequence = value[0]  # 提取序列字符串
        binary_list = value[1]  # 提取二进制列表
        binary_str = ''.join(map(str, binary_list))  # 将二进制列表转为字符串
        # 构造目标格式
        formatted = f'>{key}\n{sequence}\n{binary_str}\n'
        result.append(formatted)
    return ''.join(result)

# 转换为目标格式
output = convert_to_fasta(data)

# 保存到同名的 .fa 文件
output_file = "Test_315-28.fa"
with open(output_file, 'w') as f:
    f.write(output)

print(f"数据已保存到 {output_file}")
