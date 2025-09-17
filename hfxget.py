#!/usr/bin/env python3
"""
Xget Hugging Face 下载加速器
用于替代 hf download 命令，通过 Xget 加速下载，避免网络问题

策略：
1. 使用 HF API 获取完整文件列表
2. 小文件从 hf-mirror.com 下载
3. LFS 文件从 Xget 下载
4. 验证文件完整性
"""

import argparse
import hashlib
import sys
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from huggingface_hub import HfApi
from tqdm import tqdm


class DownloaderInterface(ABC):
    """下载器抽象接口"""

    @abstractmethod
    def download_file(self, url, local_path, resume=True, max_retries=3):
        """下载单个文件"""
        pass

    @abstractmethod
    def get_name(self):
        """获取下载器名称"""
        pass


class RequestsDownloader(DownloaderInterface):
    """基于 requests 库的下载器"""

    def get_name(self):
        return "requests"

    def download_file(self, url, local_path, resume=True, max_retries=3):
        """使用 requests 下载文件"""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests 库未安装，请运行: pip install requests")

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果不使用断点续传，直接下载到目标位置
        if not resume:
            temp_path = local_path
            if local_path.exists():
                print(f"覆盖现有文件: {local_path.name}")
        else:
            # 使用 .incomplete 后缀的临时文件
            temp_path = local_path.with_suffix(local_path.suffix + ".incomplete")
            if local_path.exists():
                print(f"文件已存在，跳过: {local_path.name}")
                return True

        # 配置请求会话
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
        )

        # 禁用SSL警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # 配置重试策略
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        for attempt in range(max_retries + 1):
            headers = session.headers.copy()
            mode = "wb"
            initial_pos = 0

            # 检查断点续传
            if resume and temp_path.exists() and temp_path != local_path:
                initial_pos = temp_path.stat().st_size
                headers["Range"] = f"bytes={initial_pos}-"
                mode = "ab"
                print(
                    f"断点续传: {local_path.name} (从 {initial_pos / (1024*1024):.1f} MB 开始)"
                )

            try:
                response = session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(30, 60),
                    verify=True,
                    allow_redirects=True,
                )

                if response.status_code == 416 and resume:
                    if temp_path.exists() and temp_path != local_path:
                        temp_path.rename(local_path)
                        print(f"文件已完整，重命名: {local_path.name}")
                        return True
                    return True

                response.raise_for_status()
                total_size = (
                    int(response.headers.get("content-length", 0)) + initial_pos
                )

                with open(temp_path, mode) as f:
                    with tqdm(
                        desc=local_path.name,
                        total=total_size,
                        initial=initial_pos,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                # 重命名临时文件
                if resume and temp_path != local_path:
                    temp_path.rename(local_path)

                return True

            except Exception as e:
                if attempt < max_retries:
                    print(
                        f"下载失败 (尝试 {attempt + 1}/{max_retries + 1}): {local_path.name}"
                    )
                    print(f"错误: {str(e)}")
                    time.sleep(2**attempt)
                    continue
                else:
                    print(f"下载 {local_path.name} 失败: {str(e)}")
                    if resume and temp_path.exists() and temp_path != local_path:
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                    return False

        return False


class XgetHFDownloader:
    def __init__(
        self,
        xget_base_url="https://xget.xi-xu.me/hf",
        hf_mirror_url="https://hf-mirror.com",
        downloader_type="requests",
    ):
        self.xget_base_url = xget_base_url
        self.hf_mirror_url = hf_mirror_url
        self.hf_api = HfApi(endpoint=hf_mirror_url)

        # LFS 文件大小阈值 (100MB)
        self.lfs_size_threshold = 100 * 1024 * 1024

        # 选择下载器
        if downloader_type == "requests":
            self.downloader = RequestsDownloader()
        else:
            raise ValueError(f"不支持的下载器类型: {downloader_type}")

        print(f"使用下载核心: {self.downloader.get_name()}")
        print(f"HF 镜像: {self.hf_mirror_url}")
        print(f"Xget 加速: {self.xget_base_url}")

    def get_repo_file_list(
        self, repo_id, repo_type="model", revision="main"
    ):
        """获取仓库文件列表和详细信息"""
        try:
            print(f"正在获取 {repo_type} {repo_id} 的文件列表...")

            repo_info = self.hf_api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)

            files_info = []
            for sibling in repo_info.siblings:
                file_info = {
                    "filename": sibling.rfilename,
                    "size": sibling.size,
                    "lfs": sibling.lfs,
                    "blob_id": sibling.blob_id,
                }
                files_info.append(file_info)

            return files_info

        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return []

    def is_lfs_file(self, file_info):
        """判断文件是否为 LFS 文件"""
        # 如果 API 直接标记了 LFS
        print(file_info)
        if file_info.get("lfs"):
            return True

        return False

    def build_download_url(
        self, repo_id, filename, repo_type="model", revision="main", is_lfs=False
    ):
        """构建下载 URL"""
        # 确保文件名在URL中使用正斜杠
        url_filename = filename.replace("\\", "/")

        if is_lfs:
            # LFS 文件使用 Xget
            if repo_type == "dataset":
                hf_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{url_filename}?download=true"
            elif repo_type == "space":
                hf_url = f"https://huggingface.co/spaces/{repo_id}/resolve/{revision}/{url_filename}?download=true"
            else:  # model
                hf_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{url_filename}?download=true"

            download_url = hf_url.replace("https://huggingface.co", self.xget_base_url)
            url_type = "Xget"
        else:
            # 普通文件使用 hf-mirror
            download_url = None
            url_type = "hf-mirror"
        hf_mirror_param = {"repo_id": repo_id, "filename": filename, "revision": revision, "repo_type": repo_type}

        return download_url, url_type, hf_mirror_param

    def verify_file_integrity(
        self, file_path, expected_size=None, expected_sha256=None
    ):
        """验证文件完整性"""
        if not file_path.exists():
            return False

        # 检查文件大小
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                print(
                    f"文件大小不匹配 {file_path.name}: 期望 {expected_size}, 实际 {actual_size}"
                )
                return False

        # 对小文件验证 SHA256（如果提供了哈希值）
        if expected_sha256 and expected_size and expected_size < 500 * 1024 * 1024:
            try:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)

                actual_sha256 = sha256_hash.hexdigest()
                if actual_sha256 != expected_sha256:
                    print(
                        f"SHA256不匹配 {file_path.name}: 期望 {expected_sha256}, 实际 {actual_sha256}"
                    )
                    return False
            except Exception as e:
                print(f"SHA256验证失败 {file_path.name}: {e}")
                return False

        return True

    def download_and_verify_file(self, url, local_dir, local_path, file_info, url_type, hf_mirror_param):
        """下载文件并验证完整性"""
        print(f"正在下载: {local_path.name} (使用 {url_type})")

        if url_type == "Xget":
            try:
                success = self.downloader.download_file(url, local_path, resume=True)
                if not success:
                    print(f"Xget 下载失败: {local_path.name}")
            except Exception as e:
                print(f"Xget 下载失败: {local_path.name}，{e}\n{traceback.format_exc()}")
                success = False
            if success:
                # 验证文件完整性
                expected_size = file_info.get("size")
                if expected_size and local_path.exists():
                    if not self.verify_file_integrity(local_path, expected_size):
                        print(f"下载完成但验证失败，删除文件: {local_path.name}")
                        try:
                            local_path.unlink()
                        except OSError:
                            pass
                        return False

                size_mb = (
                    (local_path.stat().st_size / (1024 * 1024)) if local_path.exists() else 0
                )
                print(f"✓ 下载成功: {local_path.name} ({size_mb:.1f} MB)")
                return True
        try:
            self.hf_api.hf_hub_download(**hf_mirror_param, local_dir=local_dir, resume_download=True)
            return True
        except Exception as e:
            print(f"下载失败: {local_path.name}，{e}\n{traceback.format_exc()}")
            return False

    def download_repo(
        self,
        repo_id,
        local_dir,
        repo_type="model",
        revision="main",
        max_workers=4,
        include_patterns=None,
        exclude_patterns=None,
    ):
        """下载整个仓库"""
        # 验证仓库ID
        try:
            self.hf_api.repo_info(repo_id, repo_type=repo_type, revision=revision)
        except Exception as e:
            print(f"无效的仓库ID: {e}")
            return False

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # 获取文件列表
        files_info = self.get_repo_file_list(repo_id, repo_type, revision)

        if not files_info:
            print("未找到文件或获取文件列表失败")
            return False

        # 过滤文件
        if include_patterns:
            files_info = [
                f
                for f in files_info
                if any(pattern in f["filename"] for pattern in include_patterns)
            ]

        if exclude_patterns:
            files_info = [
                f
                for f in files_info
                if not any(pattern in f["filename"] for pattern in exclude_patterns)
            ]

        # 分类文件
        lfs_files = []
        regular_files = []

        for file_info in files_info:
            if self.is_lfs_file(file_info):
                lfs_files.append(file_info)
            else:
                regular_files.append(file_info)

        print(f"找到 {len(files_info)} 个文件:")
        print(f"  🔗 LFS文件 (Xget): {len(lfs_files)}")
        print(f"  📄 普通文件 (hf-mirror): {len(regular_files)}")

        # 检查哪些文件需要下载
        files_to_download = []
        files_already_complete = 0

        for file_info in files_info:
            filename = file_info["filename"]
            local_path = local_dir / filename
            is_lfs = self.is_lfs_file(file_info)

            # 检查文件是否需要下载
            needs_download = False
            reason = ""

            if not local_path.exists():
                needs_download = True
                reason = "文件不存在"
            else:
                # 检查文件完整性
                expected_size = file_info.get("size")
                if expected_size and self.verify_file_integrity(
                    local_path, expected_size
                ):
                    print(
                        f"✓ 文件完整: {filename} ({expected_size / (1024*1024):.1f} MB)"
                    )
                    files_already_complete += 1
                    continue
                else:
                    needs_download = True
                    reason = "文件不完整或大小不匹配"

            if needs_download:
                url, url_type, hf_mirror_param = self.build_download_url(
                    repo_id, filename, repo_type, revision, is_lfs
                )
                print(f"→ 需要下载: {filename} ({reason}) - 使用 {url_type}")
                files_to_download.append((url, local_path, file_info, url_type, hf_mirror_param))

        print(f"\n需要下载: {len(files_to_download)} 个文件")
        print(f"已完整: {files_already_complete} 个文件")

        if not files_to_download:
            print("所有文件已下载完成")
            return True

        print(f"开始下载 {len(files_to_download)} 个文件")

        # 并发下载文件
        successful_downloads = 0
        failed_downloads = 0
        total_bytes_downloaded = 0
        start_time = time.time()
        xget_downloads = 0
        mirror_downloads = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.download_and_verify_file, url, local_dir, local_path, file_info, url_type, hf_mirror_param
                ): (url, local_path, file_info, url_type, hf_mirror_param)
                for url, local_path, file_info, url_type, hf_mirror_param in files_to_download
            }

            with tqdm(
                total=len(files_to_download),
                desc="文件下载进度",
                unit="文件",
                position=0,
            ) as main_pbar:
                for future in as_completed(future_to_task):
                    url, local_path, file_info, url_type = future_to_task[future]
                    try:
                        success = future.result()
                        if success:
                            successful_downloads += 1
                            if url_type == "Xget":
                                xget_downloads += 1
                            else:
                                mirror_downloads += 1
                            if local_path.exists():
                                total_bytes_downloaded += local_path.stat().st_size
                        else:
                            failed_downloads += 1
                            print(f"失败: {file_info['filename']}")
                    except Exception as e:
                        print(f"下载任务异常: {e}")
                        failed_downloads += 1
                    finally:
                        main_pbar.update(1)

                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            avg_speed = total_bytes_downloaded / elapsed_time
                            main_pbar.set_postfix(
                                {
                                    "成功": successful_downloads,
                                    "失败": failed_downloads,
                                    "平均速度": f"{avg_speed / (1024*1024):.1f} MB/s",
                                }
                            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_speed = total_bytes_downloaded / total_time if total_time > 0 else 0

        print(f"\n📊 下载统计:")
        print(f"  ✅ 成功: {successful_downloads}")
        print(f"    🔗 Xget下载: {xget_downloads}")
        print(f"    🪞 镜像下载: {mirror_downloads}")
        print(f"  ❌ 失败: {failed_downloads}")
        print(f"  📁 总计: {len(files_to_download)}")
        print(f"  💾 下载量: {total_bytes_downloaded / (1024*1024*1024):.2f} GB")
        print(f"  ⏱️  用时: {total_time:.1f} 秒")
        print(f"  🚀 平均速度: {avg_speed / (1024*1024):.1f} MB/s")
        print(f"  🔧 下载核心: {self.downloader.get_name()}")

        return failed_downloads == 0


def main():
    parser = argparse.ArgumentParser(
        description="Xget Hugging Face 下载加速器（无Git依赖版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python xget_hf.py download microsoft/DialoGPT-medium --local-dir ./model
  python xget_hf.py download squad --repo-type dataset --local-dir ./data
  python xget_hf.py download microsoft/DialoGPT-medium --max-workers 8 --downloader requests

下载策略:
  1. 使用 HF API 获取完整文件列表和信息
  2. 小文件从 hf-mirror.com 快速下载
  3. LFS 大文件从 Xget 加速下载
  4. 验证文件完整性（文件大小验证）
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    download_parser = subparsers.add_parser("download", help="下载模型/数据集/空间")
    download_parser.add_argument("repo_id", help="仓库ID，格式: username/repo-name")
    download_parser.add_argument("--local-dir", required=True, help="本地目录路径")
    download_parser.add_argument(
        "--repo-type",
        choices=["model", "dataset", "space"],
        default="model",
        help="仓库类型 (默认: model)",
    )
    download_parser.add_argument(
        "--revision", default="main", help="分支/标签/提交 (默认: main)"
    )
    download_parser.add_argument(
        "--max-workers", type=int, default=4, help="并发下载数 (默认: 4)"
    )
    download_parser.add_argument("--include", nargs="*", help="包含的文件模式")
    download_parser.add_argument("--exclude", nargs="*", help="排除的文件模式")
    download_parser.add_argument(
        "--hf-mirror-url",
        default="https://hf-mirror.com",
        help="HF 镜像URL，用于普通文件 (默认: https://hf-mirror.com)",
    )
    download_parser.add_argument(
        "--xget-url",
        default="https://xget.xi-xu.me/hf",
        help="Xget 基础URL，用于 LFS 文件 (默认: https://xget.xi-xu.me/hf)",
    )
    download_parser.add_argument(
        "--downloader",
        choices=["requests"],
        default="requests",
        help="下载核心",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "download":
        # 检查下载器可用性
        if args.downloader == "requests" and not REQUESTS_AVAILABLE:
            print("错误: requests 库未安装，请运行: pip install requests")
            return 1

        try:
            downloader = XgetHFDownloader(
                args.xget_url, args.hf_mirror_url, args.downloader
            )
        except Exception as e:
            print(f"初始化下载器失败: {e}")
            return 1

        success = downloader.download_repo(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            repo_type=args.repo_type,
            revision=args.revision,
            max_workers=args.max_workers,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )

        return 0 if success else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
