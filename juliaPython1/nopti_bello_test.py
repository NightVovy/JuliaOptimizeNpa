import subprocess
import json

# 读取 nopti.py 输出的 JSON 数据
def read_nopti_output():
    # 假设 nopti.py 输出的结果已经保存在 'nopti_output.json'
    with open('nopti_output.json', 'r') as file:
        data = json.load(file)
    return data

# 调用 bello.py 进行计算
def call_bello(data):
    # 将从 nopti.py 获取的数据传递给 bello.py
    # 假设 bello.py 接受命令行参数并通过文件或标准输入/输出交互
    # 在这里，我们通过一个简单的示例调用 bello.py，将数据传递给它
    # 假设 bello.py 是一个可以直接通过 subprocess 运行的脚本

    # 构造输入参数，这里假设 bello.py 可以直接从命令行接收参数或读取 JSON 文件
    command = ["python", "bello.py"]
    
    # 我们可以将数据作为 JSON 传递给 bello.py（或者通过标准输入）
    # 假设 bello.py 读取标准输入，这里将数据传递给它
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 将数据传递给 bello.py
    input_data = json.dumps(data).encode('utf-8')
    stdout, stderr = process.communicate(input=input_data)

    # 处理 bello.py 返回的结果
    if stderr:
        print(f"Error: {stderr.decode('utf-8')}")
    return stdout.decode('utf-8')

# 验证数据是否正确传输和计算
def verify_data_flow():
    # 获取 nopti.py 的输出数据
    nopti_data = read_nopti_output()

    # 输出从 nopti.py 获取的数据，验证是否读取成功
    print("Nopti Output:")
    print(nopti_data)

    # 调用 bello.py 处理这些数据
    bello_output = call_bello(nopti_data)

    # 输出 bello.py 的计算结果
    print("Bello Output:")
    print(bello_output)

# 运行测试
if __name__ == "__main__":
    verify_data_flow()
