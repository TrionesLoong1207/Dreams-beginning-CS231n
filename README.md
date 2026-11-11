# CS231n independent study

本仓库用于独立完成 CS231n 三份作业，记录自己的实现、实验和学习笔记。
题目和 starter code 来自 [Stanford CS231n](https://cs231n.github.io/)。
当前全部解题 TODO 尚未填写；环境适配不代表作业已完成。

## 本地启动

环境目录：`.venv`；Python 3.12。不会修改已有 Conda 环境。

在 PowerShell 中运行：

```powershell
cd D:\cs231n
.\.venv\Scripts\python.exe -m jupyterlab
```

也可运行 `powershell -ExecutionPolicy Bypass -File .\start-jupyter.ps1`。
在浏览器中打开 `assignment1/knn.ipynb`，选择本项目的 Python 内核。
使用 VS Code 时，打开整个 `D:\cs231n` 文件夹，手动确认 notebook 内核为
`D:\cs231n\.venv\Scripts\python.exe`。

作业一、二的开头环境单元已适配本地路径，不必挂载 Google Drive。
每个 notebook 使用独立内核，避免导入其他作业目录下的同名 `cs231n` 包。
作业三实际目录是 `assignment3/assignment3`，其中的 Colab、Bash、GPU 和
DataLoader 单元需要在开始作业三时继续适配；当前不宣称整套作业三已验证。
`collect_submission.ipynb` 保留原始 Colab 打包流程，本地 PDF 导出尚未配置。

## 复现环境

需要 Python 3.12、Git；作业二编译需要 MSVC C++ Build Tools。
以下命令在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe scripts\prepare_cifar10.py
.\.venv\Scripts\python.exe scripts\prepare_local_notebooks.py
```

PyTorch 与 torchvision 的版本配对来自 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v271)。

`requirements-local.txt` 是本地基础依赖。不要混用作业三原有的旧版 requirements。
作业三的 CLIP、TensorFlow Datasets、视频解码等额外依赖在开始相应题目时另行安装。
COCO 数据、模型权重、emoji 和 DAVIS 数据也需要按对应 notebook 准备。
如有 `requirements-local.lock.txt`，它记录本次安装的精确版本，可用于复现。

## 已验证范围

- Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1。
- PyTorch 2.7.1 + CUDA 12.6、torchvision 0.22.1；RTX 4070 Laptop 上实际 GPU 运算通过。
- 作业二提供的 Cython 扩展已编译，导入与输入输出检查通过。
- JupyterLab 的本机网页服务和认证 API 通过检查。
- 在真实 Jupyter 内核中执行了 kNN 的环境、导入、数据加载单元，读取 50,000 张训练图片和 10,000 张测试图片。
- `pip check` 通过。仅修改作业一、二共 11 个基础设施单元；解题单元、推导、文字答案均保留原样。

再次检查可执行 `.\.venv\Scripts\python.exe scripts\check_environment.py`。
尚未验证其他 notebook 的完整执行；COCO 等其他数据、作业三额外依赖与本地提交 PDF 流程仍待准备。

## Git 与 GitHub

已保留原始 starter code 的本地提交，方便查看自己的修改。
建议先创建一个 **Private** GitHub 仓库，暂不勾选初始化 README、License 或 .gitignore。
本地的 README、忽略规则已经准备好。
创建后，在仓库根目录填入实际地址并执行：

```powershell
git remote add origin https://github.com/YOUR_USERNAME/cs231n-study.git
git push -u origin main
```

GitHub 登录需要使用浏览器认证或 SSH；不要把访问令牌写进文件或远程地址。
以后每完成一个可解释、已验证的部分：

```powershell
git status
git diff
git add assignment1/knn.ipynb assignment1/cs231n/classifiers/k_nearest_neighbor.py
git commit -m "Complete kNN exercise and validation"
git push
```

环境、缓存、数据集、权重、原始压缩包、编译产物均已忽略。
notebook 的文字回答、输出、曲线会随 `.ipynb` 保存，可另外保存精简的实验图片和报告。
每次提交前检查输出，避免把本机隐私、令牌或超大媒体文件放进仓库。

## 学习与辅助记录

实现代码、推导和文字回答由本人独立完成。
本次 AI 协助范围：文件检查、作业要求梳理、学习方法建议、本地环境与 Git 基础设施。
后续辅助情况应如实补充，并按 notebook 的声明要求填写。
使用讲解视频时，记录自己参考的内容与尚未掌握的问题。
