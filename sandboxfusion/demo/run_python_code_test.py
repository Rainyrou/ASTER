import base64
import json
import os

import requests

# =========== 执行代码: 打印输出 ===========
response = requests.post('http://127.0.0.1:8080/run_code', json={
    'code': 'print("hello!")',
    'language': 'python',
})
print(json.dumps(response.json(), indent=2, ensure_ascii=False))


# =========== 执行代码: 公式计算 ===========
response = requests.post('http://127.0.0.1:8080/run_code', json={
    'code':
"""
# 斐波那契数的比奈公式
import math

def fibonacci(n):
    return ((1 + math.sqrt(5)) ** n - (1 - math.sqrt(5)) ** n) / (2 ** n * math.sqrt(5))

result = fibonacci(100)
print(f"斐波那契数列的第100项是：{result:.10f}")
""",
    'language': 'python',
})
print(json.dumps(response.json(), indent=2, ensure_ascii=False))


# =========== 上传文件 + 执行代码 ===========
# 项目根目录
root_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"root_dir: {root_dir}")

# 读取本地文件并转换为base64
with open(os.path.join(root_dir, 'flag.txt'), 'rb') as f:
    content = f.read()
    base64_content = base64.b64encode(content).decode('utf-8')

# 发送请求，包含要上传的文件
response = requests.post('http://127.0.0.1:8080/run_code', json={
    'code': 'print(open("flag.txt").read())',
    'language': 'python',
    'files': {'flag.txt': base64_content}
})

# 打印上传的文件内容
print(json.dumps(response.json(), indent=2))
# print(response.json())


# =========== 执行代码 + 下载文件 ===========
# 执行代码并请求下载文件
response = requests.post('http://127.0.0.1:8080/run_code', json={
    'code': 'open("flag.txt", "w").write("this is a secret")',
    'language': 'python',
    'fetch_files': ['flag.txt']
})

# 打印响应结果
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# 解码并打印下载的文件内容
print(base64.b64decode(response.json()['files']['flag.txt']).decode('utf-8'))  # this is a secret
# print(response.json())