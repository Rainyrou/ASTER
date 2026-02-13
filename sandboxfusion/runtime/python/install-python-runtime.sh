#!/bin/bash
set -o errexit

USE_OFFICIAL_SOURCE=0
for arg in "$@"
do
    if [ "$arg" = "us" ]; then
        USE_OFFICIAL_SOURCE=1
    fi
done

rm -f ~/.condarc
conda create -n sandbox-runtime -y python=3.10

# 激活conda虚拟环境
conda_env_name="sandbox-runtime"
source activate ${conda_env_name}

if [ "$(uname -s)" = "Linux" ]; then
    # 使用conda安装编译pyqt
    conda install -c conda-forge pyqt -y

    # 使用conda安装编译gmpy2
    conda install gmpy2 -c conda-forge -y

    # 安装其他依赖
    pip install -r ./requirements.txt --ignore-requires-python -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
else
    pip install -r ./requirements_windows.txt --ignore-requires-python -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
fi

# for NaturalCodeBench python problem 29
python -c "import nltk; nltk.download('punkt')"

# for CIBench nltk problems
python -c "import nltk; nltk.download('stopwords')"

pip cache purge
conda clean --all -y
