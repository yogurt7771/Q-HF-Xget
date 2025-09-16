# Xget Hugging Face 下载加速器

🚀 **Hugging Face模型下载工具**，通过[Xget](https://github.com/xixu-me/Xget)和[hf-mirror](https://hf-mirror.com/)加速大文件，可无代理下载。

## ✨ 特性

- 🔗 **智能下载策略**：LFS大文件使用Xget加速，普通文件使用hf-mirror镜像
- 🚫 **无Git依赖**：不需要安装Git，直接下载文件
- ⚡ **并发下载**：支持多线程并发下载，提高效率
- 🔄 **断点续传**：支持下载中断后继续下载
- ✅ **完整性验证**：自动验证文件大小和SHA256哈希值
- 🎯 **灵活过滤**：支持包含/排除特定文件模式
- 📊 **详细统计**：显示下载进度、速度、成功率等统计信息

## 📦 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yogurt7771/Q-HF-Xget.git
cd Q-HF-Xget
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python hfxget.py --help
```

## 🚀 快速开始

### 下载模型

```bash
# 下载DialoGPT模型
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models/dialogpt

# 下载指定分支的模型
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models/dialogpt --revision v1.0
```

### 下载数据集

```bash
# 下载SQuAD数据集
python hfxget.py download squad --repo-type dataset --local-dir ./data/squad
```

### 下载Space应用

```bash
# 下载Hugging Face Space应用
python hfxget.py download microsoft/DialoGPT-medium --repo-type space --local-dir ./spaces/dialogpt
```

## 📖 详细用法

### 基本语法

```bash
python hfxget.py download <repo_id> --local-dir <local_directory> [选项]
```

### 必需参数

- `repo_id`: 仓库ID，格式为 `username/repo-name`
- `--local-dir`: 本地保存目录路径

### 可选参数

#### 仓库相关

- `--repo-type {model,dataset,space}`: 仓库类型（默认：model）
- `--revision <revision>`: 分支/标签/提交（默认：main）

#### 下载配置

- `--max-workers <num>`: 并发下载数（默认：4）
- `--downloader {requests}`: 下载核心（默认：requests）

#### 文件过滤

- `--include <pattern1> <pattern2> ...`: 只下载包含指定模式的文件
- `--exclude <pattern1> <pattern2> ...`: 排除包含指定模式的文件

#### 服务器配置

- `--hf-mirror-url <url>`: HF镜像URL（默认：<https://hf-mirror.com）>
- `--xget-url <url>`: Xget基础URL（默认：<https://xget.xi-xu.me/hf）>

## 💡 使用示例

### 1. 基础下载

```bash
# 下载完整的BERT模型
python hfxget.py download bert-base-uncased --local-dir ./models/bert

# 下载到当前目录
python hfxget.py download microsoft/DialoGPT-small --local-dir .
```

### 2. 高级过滤

```bash
# 只下载PyTorch模型文件
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --include "*.bin" "*.safetensors"

# 排除特定文件
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --exclude "*.h5" "*.onnx"

# 只下载配置文件
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./config --include "config.json" "tokenizer.json"
```

### 3. 性能优化

```bash
# 使用8个并发线程
python hfxget.py download microsoft/DialoGPT-large --local-dir ./models --max-workers 8
```

### 4. 下载特定版本

```bash
# 下载特定标签
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --revision v1.0

# 下载特定提交
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --revision abc123def
```

### 5. 数据集下载

```bash
# 下载GLUE数据集
python hfxget.py download glue --repo-type dataset --local-dir ./data/glue

# 下载特定数据集子集
python hfxget.py download squad --repo-type dataset --local-dir ./data --include "train*" "dev*"
```

## 🔧 下载策略

### 文件分类

- **LFS文件**：使用Xget加速下载
  - 模型权重文件（.bin, .safetensors, .ckpt等）
  - 大型数据文件（.tar.gz, .zip等）
  - 其他大文件

- **普通文件**：使用hf-mirror镜像下载
  - 配置文件（config.json, tokenizer.json等）
  - 小文件（README.md, .gitattributes等）

### 下载流程

1. 🔍 获取仓库文件列表和元数据
2. 📊 分析文件大小和类型
3. 🎯 智能选择下载源（Xget/hf-mirror）
4. ⚡ 并发下载文件
5. ✅ 验证文件完整性
6. 📈 显示下载统计

## 🛠️ 故障排除

### 常见问题

#### 1. 依赖库缺失

```bash
# 如果提示缺少requests库
pip install requests

# 安装所有依赖
pip install -r requirements.txt
```

#### 2. 网络连接问题

```bash
# 使用不同的镜像源
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --hf-mirror-url https://hf-mirror.com

# 减少并发数
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --max-workers 1
```

#### 3. 权限问题

```bash
# 确保有写入权限
mkdir -p ./models
chmod 755 ./models
```

#### 4. 内存不足

```bash
# 减少并发数
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --max-workers 2
```

### 调试模式

```bash
# 使用requests下载器（更稳定）
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --downloader requests

# 只下载小文件测试
python hfxget.py download microsoft/DialoGPT-medium --local-dir ./models --include "*.json" "*.txt"
```

## 📊 输出说明

### 下载进度

```plaintext
使用下载核心: requests
HF 镜像: https://hf-mirror.com
Xget 加速: https://xget.xi-xu.me/hf
正在获取 model microsoft/DialoGPT-medium 的文件列表...
找到 15 个文件:
  🔗 LFS文件 (Xget): 3
  📄 普通文件 (hf-mirror): 12

需要下载: 3 个文件
已完整: 12 个文件
开始下载 3 个文件

pytorch_model.bin: 100%|████████| 355M/355M [00:45<00:00, 7.8MB/s]
config.json: 100%|████████| 1.2k/1.2k [00:00<00:00, 2.1kB/s]
```

### 下载统计

```plaintext
📊 下载统计:
  ✅ 成功: 3
    🔗 Xget下载: 1
    🪞 镜像下载: 2
  ❌ 失败: 0
  📁 总计: 3
  💾 下载量: 0.33 GB
  ⏱️  用时: 45.2 秒
  🚀 平均速度: 7.5 MB/s
  🔧 下载核心: requests
```

## 🔗 相关链接

- [Hugging Face Hub](https://huggingface.co/)
- [hf-mirror镜像站](https://hf-mirror.com/)
- [Xget加速服务](https://xget.xi-xu.me/)
- [Xget项目](https://github.com/xixu-me/Xget)
- [huggingface_hub文档](https://huggingface.co/docs/hub/index)

## 📝 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## ⚠️ 注意事项

1. **网络环境**：确保网络连接稳定，建议使用稳定的网络环境
2. **存储空间**：大模型文件可能占用数GB空间，请确保有足够的存储空间
3. **下载速度**：下载速度取决于网络环境和服务器负载
4. **文件完整性**：程序会自动验证文件完整性，如有问题会重新下载
5. **并发限制**：过高的并发数可能导致网络拥塞，建议根据网络环境调整

---

**享受快速下载Hugging Face模型的乐趣！** 🎉
