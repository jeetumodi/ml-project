import os
import subprocess
import time

# 执行nvidia-smi命令来获取GPU信息
def get_gpu_memory():
    try:
        # 获取 GPU 信息的输出
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 处理输出
        output = result.stdout.strip().split('\n')
        gpu_info = []
        for line in output:
            memory_info = line.split(', ')
            total_memory = int(memory_info[0])  # 总内存，单位MB
            free_memory = int(memory_info[1])   # 空闲内存，单位MB
            used_memory = int(memory_info[2])   # 已用内存，单位MB
            gpu_info.append((total_memory, free_memory, used_memory))
        return gpu_info
    except Exception as e:
        print(f"Error occurred while getting GPU info: {e}")
        return []

def main():
    while True:
        gpu_info = get_gpu_memory()

        for idx, (total_memory, free_memory, used_memory) in enumerate(gpu_info):
            free_memory_gb = free_memory / 1024  # 转换为GB
            if free_memory_gb > 10:  # 如果空闲内存大于10GB
                print(f"GPU {idx} has more than 10GB of free memory. Running 'python ppi_main.py'...")
                # 执行指定的Python脚本
                subprocess.run(["python", "ppi_main.py"])

        # 每隔100秒再检查一次
        print("Waiting for 100 seconds before the next check...")
        time.sleep(100)

if __name__ == "__main__":
    main()
