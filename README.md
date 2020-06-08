# ConstructionHunter

### Directories
* config/                    配置文件
* data/                       数据集
    * dictionary/      字典文件（代码生成）
    * output/           输出数据（代码生成）
    * test/                待标注数据（人工准备）
    * train/               训练集数据（人工准备）
    * validation/      验证集数据（代码生成）
* images/                  模型结构图（代码生成）
* libraries/                 程序依赖库（人工准备）
    * bert_model
    * stanford-corenlp
* log/                         日志文件（代码生成）
* model/                    模型文件（代码生成）
* scripts/                   脚本文件
* src/                         源代码文件
* .gitignore
* requirements.txt   Python Packages
* README.md



### Preparation
**安装 Python Packages 依赖**
```shell script
# Python 3+
cd /path/to/project/
pip install -r requirements.txt
```



**安装 Stanford-CoreNLP**

1. 打开浏览器，输入网址：
```shell script
https://stanfordnlp.github.io/CoreNLP/download.html
```
2. 下载 `CoreNLP 4.0.0`，解压文件，重命名为`stanford-corenlp`，并置于`/libraries`下；
3. 下载 `Chinese 4.0.0`，解压文件，重命名为`stanford-chinese-corenlp-2020-03-15-models.jar`，并置于`/libraries/stanford-corenlp`下；
4. 安装 `java`。



**下载 bert-model**

1. 打开浏览器，输入网址：
```shell script
https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD
```
2. 下载喜欢的`bert`预训练语言模型，并置于`/libraries`下。



**运行脚本**

```shell script
cd /path/to/project/scripts

```



### Usage

| 系统     | 模型                                   | 备注                      |
| -------- | -------------------------------------- | ------------------------- |
| Zeus     | 基于强知识驱动的复句型构式自动识别模型 | ````                      |
| Poseidon | 基于Bert的复句型构式自动识别模型       |                           |
| Hades    | 基于Bi-LSTM的复句型构式自动识别模型    | 要求管理员运行 / **sudo** |



**若使用基于强知识驱动的复句型构式自动识别模型：**

```bash
cd /path/to/project/src
python main.py --system=zeus --mode=predict
```



**若使用基于基于Bert的复句型构式自动识别模型：**

```bash
cd /path/to/project/src
python main.py --system=poseidon --mode=predict
```



**若使用基于Bi-LSTM的复句型构式自动识别模型：**

```bash
cd /path/to/project/src
sudo python main.py --system=hades --mode=predict
```



### Attention

**Poseidon 模型训练硬件要求**

| GPU配置                          | SEQUENCE LENGTH | BATCH SIZE |
| -------------------------------- | --------------- | ---------- |
| Quadro RTX8000 [48G]   RAM: 128G | 180             | 32         |
| Titan RTX [24G]   RAM: 256G      | 180             | 16         |
| GPU: 2080Ti [11G]   RAM: 63G     | 180             | 1          |

* `SEQUENCE LENGTH`与语料句长相关，勿使句长大于`SEQUENCE LENGTH`，否则会报错；
* `BATCH SIZE`理论上取值越大，网络收敛越快（批量随机梯度下降）；GPU配置越高，模型效果越佳。