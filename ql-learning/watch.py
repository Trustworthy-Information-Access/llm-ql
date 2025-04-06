import subprocess
import re
import os
import time

def get_gpu_utilizations():
    # 调用nvidia-smi获取GPU信息
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    # 去除每行末尾的换行符，并将其转换为整数列表
    utilizations = [int(line.strip("%")) for line in result.stdout.split('\n')[:-1]]
    print("result:", result)
    print(result.stdout.split('\n'))
    print(utilizations)
    return utilizations

def check_and_run():
    # 获取GPU利用率列表
    gpu_utils = get_gpu_utilizations()
    print(f"Current GPU utilizations: {gpu_utils}")
    
    # 检查所有GPU的利用率是否都为0
    if all(util == 0 for util in gpu_utils):
        print("All GPUs have 0% utilization, running run.py...")
        # 使用nohup命令在后台运行run.py，并忽略输入
        os.system('sh /root/paddlejob/workspace/env_run/llm-index/examples/llm-pretrain/run.sh > /root/paddlejob/workspace/env_run/output/run-w/o-s.log')

if __name__ == "__main__":
    # 可以设置一个循环来定期检查GPU利用率
    while True:
        try:
            check_and_run()
        except Exception as e:
            print(f"An error occurred: {e}")
        # 这里设置检查间隔为10秒，可以根据需要调整
        time.sleep(200)