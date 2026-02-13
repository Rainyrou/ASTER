import subprocess
import sys

def install_packages(packages):
    # 设置国内镜像源
    index_url = "http://mirrors.aliyun.com/pypi/simple/"
    trusted_host = "mirrors.aliyun.com"

    # 构建 pip install 命令
    command = ["pip", "install"]

    # 添加需要的参数
    command.extend(packages)
    command.extend(["-i", index_url])
    command.extend(["--trusted-host", trusted_host])

    # 运行命令
    try:
        subprocess.check_call(command)
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing packages: {e}")

# 示例用法
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python install_packages.py package1 package2 ...")
    else:
        install_packages(sys.argv[1:])