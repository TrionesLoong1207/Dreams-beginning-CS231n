# CS231n independent study

本仓库用于独立完成 CS231n 三份作业，记录自己的实现、实验和学习笔记。
题目和 starter code 来自 [Stanford CS231n](https://cs231n.github.io/)。
仓库保存了三份作业的实现、实验输出与学习记录。部分实验依赖外部数据和预训练权重；具体运行结果与验证范围请查看各 notebook。

## 作业内容

- [Assignment 1](assignment1/)：图像分类、kNN、线性分类器和全连接网络。
- [Assignment 2](assignment2/)：卷积网络、归一化、PyTorch 与循环神经网络图像描述。
- [Assignment 3](assignment3/assignment3/)：Transformer、ViT、对比学习、CLIP、DINO 与扩散模型。

每份作业包含对应的 notebook、代码和实验记录；原始题目及 starter code 版权归课程提供方所有。

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
作业三实际目录是 `assignment3/assignment3`。部分 notebook 包含本机路径、Colab 或 Bash 单元，换环境运行时可能需要调整。
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
- 上述环境检查记录于本地环境准备阶段；各作业的具体运行结果保留在 notebook 中。

再次检查可执行 `.\.venv\Scripts\python.exe scripts\check_environment.py`。
没有在一次全新环境中完整重跑所有 notebook；COCO、DAVIS 等数据、作业三额外依赖与本地提交 PDF 流程需按实际环境准备。

## Git 与 GitHub

已保留原始 starter code 的本地提交，方便查看自己的修改。
仓库已配置 GitHub 远程地址。后续在仓库根目录使用以下命令推送更新：

```powershell
git push
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

本仓库记录个人学习过程，使用 AI 辅助理解题目、讲解公式、排查问题，并参考实现建议完成部分代码和文字回答；同时使用 AI 协助本地环境和 Git 工作。具体课程提交应遵守对应的课程和学术规范。
使用讲解视频时，记录自己参考的内容与尚未掌握的问题。

## 中英对照说明

三个作业直接在原文件内增加了中文：notebook 的题面与说明保留英文后追加中文，
代码注释与函数文档字符串也保留英文并追加中文，没有另建一套作业文件。
初版机器翻译存在术语及语义错误，现已追加一轮按题目语境进行的校正：
重写任务指令、常用注释和接口说明，并重写作业一的主要说明段落。
27 处随堂问题的题意此前已逐项校对。英文保留用于核对公式、函数名和路径。
例如 best value of k 表示按交叉验证表现选择的“k 的最佳取值”，不表示 k 数值最大；
implement 表示“实现”，fold 表示“数据分组/验证折”，score 表示模型的“评分”。
原文中若有形状或表述不一致，中文明确指出，而不擅自替换数学含义。
后续学习过程中又补充了题目实现、实验和文字回答。

最初添加翻译时曾保存本地 Git 检查点；后续作业实现和实验已修改代码及 notebook 输出。

如果 Jupyter 仍打开着修改前的旧标签页，请重新从磁盘打开文件，以读取新增的中文。
旧页面有未保存的个人修改时，先复制保留，避免用旧内容覆盖磁盘上的新版本。
