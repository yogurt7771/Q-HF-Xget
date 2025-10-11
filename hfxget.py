#!/usr/bin/env python3
"""
Xget Hugging Face 下载加速器
用于替代 hf download 命令，通过 Xget 加速下载，避免网络问题
"""

import argparse
import signal
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
import urllib3
from huggingface_hub import HfApi
from huggingface_hub._local_folder import \
    read_download_metadata as hf_read_download_metadata
from huggingface_hub._local_folder import \
    write_download_metadata as hf_write_download_metadata
from huggingface_hub.file_download import hf_hub_url
from huggingface_hub.utils import build_hf_headers
from huggingface_hub.utils.sha import git_hash, sha_fileobj
from tqdm import tqdm

# 全局变量用于跟踪中断状态
interrupted = False

# 应用元信息
APP_NAME = "hfxget"
APP_VERSION = "1.0"

# 统一管理 aria2 tqdm 的显示位置，避免多线程冲突
_aria2_position_lock = threading.Lock()
_aria2_active_positions = set()


def _aria2_acquire_position():
    with _aria2_position_lock:
        pos = 1
        while pos in _aria2_active_positions:
            pos += 1
        _aria2_active_positions.add(pos)
        return pos


def _aria2_release_position(position):
    with _aria2_position_lock:
        _aria2_active_positions.discard(position)


def build_default_hf_headers(token=None):
    return build_hf_headers(
        token=token,
        library_name=APP_NAME,
        library_version=APP_VERSION,
    )


def signal_handler(signum, frame):
    """处理Ctrl+C中断信号"""
    global interrupted
    interrupted = True
    print("\n\n⚠️  检测到中断信号 (Ctrl+C)，正在停止下载...")
    print("请等待当前下载任务完成...")


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)


@dataclass
class DownloadResult:
    success: bool
    status_code: int | None = None
    message: str | None = None


class DownloaderInterface(ABC):
    """下载器抽象接口"""

    @abstractmethod
    def download_file(
        self, url: str, local_path: str, resume: bool = True
    ) -> DownloadResult:
        """下载单个文件"""
        pass

    @abstractmethod
    def get_name(self):
        """获取下载器名称"""
        pass


class RequestsDownloader(DownloaderInterface):
    """基于 requests 库的下载器"""
    def __init__(self, headers):
        self.default_headers = headers

    def get_name(self):
        return "requests"

    def download_file(self, url, local_path, resume=True):
        """使用 requests 下载文件"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not resume:
            temp_path = local_path
            if local_path.exists():
                print(f"♻️  覆盖现有文件: {local_path.name}")
        else:
            temp_path = local_path.with_suffix(local_path.suffix + ".incomplete")

        session = requests.Session()
        session.headers.update(self.default_headers)

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = session.headers.copy()
        mode = "wb"
        initial_pos = 0

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

            if response.status_code == 416:
                if temp_path.exists() and temp_path != local_path:
                    local_path.unlink(missing_ok=True)
                    temp_path.rename(local_path)
                    print(f"✅ 本地已完整，重命名: {local_path.name}")
                    return DownloadResult(success=True)
                return DownloadResult(success=True)

            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0)) + initial_pos

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
                        if interrupted:
                            print(f"\n⏹️  下载被中断: {local_path.name}")
                            return DownloadResult(success=False, message="interrupted")
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            if temp_path != local_path:
                local_path.unlink(missing_ok=True)
                temp_path.rename(local_path)

            return DownloadResult(success=True)

        except Exception as e:
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code in [401, 403, 404]
            ):
                print(f"🚫 HTTP {e.response.status_code}: {local_path.name}")
                temp_path.unlink(missing_ok=True)
                return DownloadResult(
                    success=False, status_code=e.response.status_code, message=str(e)
                )
            else:
                print(f"❌ 下载失败: {local_path.name}")
            print(f"原因: {str(e)}")
            return DownloadResult(success=False, message=str(e))


class HFDownloader:
    def __init__(
        self,
        lfs_base_url="https://xget.xi-xu.me/hf",
        hf_base_url="https://xget.xi-xu.me/hf",
        hf_downloader_type="requests",
        lfs_downloader_type="requests",
        token=None,
    ):
        self.lfs_base_url = lfs_base_url
        self.hf_base_url = hf_base_url
        self.hf_api = HfApi(endpoint=hf_base_url, token=token)
        self.header = build_default_hf_headers(token=token)

        # LFS 文件大小阈值 (50MB)
        self.lfs_size_threshold = 50 * 1024 * 1024

        # 选择下载器
        self.hf_downloader: DownloaderInterface | None = None
        if hf_downloader_type == "requests":
            self.hf_downloader = RequestsDownloader(self.header)
        else:
            raise ValueError(f"不支持的下载器类型: {hf_downloader_type}")

        self.lfs_downloader: DownloaderInterface | None = None
        if lfs_downloader_type == "requests":
            self.lfs_downloader = RequestsDownloader(self.header)
        else:
            raise ValueError(f"不支持的下载器类型: {lfs_downloader_type}")

        print(f"🪞  HF 地址: {self.hf_base_url}")
        print(f"🛠️  HF 下载核心: {self.hf_downloader.get_name()}")
        print(f"🚀  LFS 地址: {self.lfs_base_url}")
        print(f"🛠️  LFS 下载核心: {self.lfs_downloader.get_name()}")

    def get_repo_file_list(self, repo_id, repo_type="model", revision="main"):
        """获取仓库文件列表和详细信息"""
        try:
            print(f"📡 获取文件列表: {repo_type} {repo_id} @ {revision}")

            repo_info = self.hf_api.repo_info(
                repo_id, repo_type=repo_type, revision=revision, files_metadata=True
            )
            
            self.commit_hash = repo_info.sha
            print(f"🔖 提交哈希: {self.commit_hash}")

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
            # 检查是否是401错误，如果是则不重试
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == 401
            ):
                print(f"🚫 访问受限 (401): {repo_id} | {e}")
                return []
            print(f"❌ 获取文件列表失败: {e}")
            return []

    def is_lfs_file(self, file_info):
        """判断文件是否为 LFS 文件"""
        return file_info.get("lfs") is not None

    def build_download_url(
        self, repo_id, filename, repo_type="model", revision="main", is_lfs=False
    ):
        """构建下载 URL"""
        # 确保文件名在URL中使用正斜杠
        url_filename = filename.replace("\\", "/")

        if is_lfs:
            # LFS 文件使用 LFS 地址
            if repo_type == "dataset":
                hf_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{url_filename}?download=true"
            elif repo_type == "space":
                hf_url = f"https://huggingface.co/spaces/{repo_id}/resolve/{revision}/{url_filename}?download=true"
            else:  # model
                hf_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{url_filename}?download=true"

            download_url = hf_url.replace("https://huggingface.co", self.lfs_base_url)
            url_type = "LFS"
        else:
            # 普通文件
            # download_url = hf_hub_url(
            #     repo_id,
            #     filename,
            #     repo_type=repo_type,
            #     revision=revision,
            #     endpoint=self.hf_base_url,
            # )
            # url_type = "HF"
            if repo_type == "dataset":
                hf_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{url_filename}?download=true"
            elif repo_type == "space":
                hf_url = f"https://huggingface.co/spaces/{repo_id}/resolve/{revision}/{url_filename}?download=true"
            else:  # model
                hf_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{url_filename}?download=true"

            download_url = hf_url.replace("https://huggingface.co", self.hf_base_url)
            url_type = "HF-URL"

        hf_mirror_param = {
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
            "repo_type": repo_type,
        }

        return download_url, url_type, hf_mirror_param

    def _extract_expected_etag(self, file_info):
        """从文件信息中解析期望的 ETag。"""

        lfs_info = file_info.get("lfs")
        if lfs_info is not None:
            sha256 = lfs_info.get("sha256")
            return sha256

        return file_info.get("blob_id")

    def verify_file_integrity(
        self, local_dir, file_path: Path, file_info, force_regenerate_etag=False
    ):
        """通过元数据验证文件完整性。"""

        if not file_path.exists():
            return False

        expected_size = file_info.get("size")
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                print(
                    f"❌ 文件大小不匹配: {file_path.name} | 期望 {expected_size}, 实际 {actual_size}"
                )
                return False

        metadata = None
        filename = file_info.get("filename")

        if force_regenerate_etag:
            self._write_local_metadata(local_dir, file_info)

        try:
            metadata = hf_read_download_metadata(Path(local_dir), filename)
        except Exception as e:
            print(f"⚠️  读取元数据失败 {file_path.name}: {e}")

        if metadata is None:
            self._write_local_metadata(local_dir, file_info)
            try:
                metadata = hf_read_download_metadata(Path(local_dir), filename)
            except Exception as e:
                print(f"⚠️  读取元数据失败 {file_path.name}: {e}")

        if metadata is None:
            print(f"⚠️  未找到有效元数据 {file_path.name}")
            return False

        expected_etag = self._extract_expected_etag(file_info)
        if expected_etag and metadata.etag != expected_etag:
            print(
                f"❌ ETag 不匹配: {file_path.name} | 期望 {expected_etag}, 实际 {metadata.etag}"
            )
            return False

        return True

    def _write_local_metadata(self, local_dir, file_info):
        """将下载的文件元数据写入本地缓存目录。"""

        filename = file_info.get("filename")
        if self.is_lfs_file(file_info):
            with open(Path(local_dir) / filename, "rb") as f:
                etag = sha_fileobj(f).hex()
        else:
            with open(Path(local_dir) / filename, "rb") as f:
                etag = git_hash(f.read())

        try:
            hf_write_download_metadata(
                Path(local_dir), filename, self.commit_hash, etag
            )
        except Exception as e:
            print(f"⚠️  写入元数据失败 {file_info['filename']}: {e}")

    def download_and_verify_file(
        self,
        url: str,
        local_dir: Path,
        local_path: Path,
        file_info,
        url_type,
        hf_mirror_param,
        max_attempts=5,
    ):
        """下载文件并验证完整性"""

        attempt = 0

        download_success = False
        performed_download = False
        while True:
            if interrupted:
                print(f"⏹️  下载被中断，跳过: {local_path.name}")
                return {"success": False, "downloaded": False, "url_type": url_type}

            if self.verify_file_integrity(
                local_dir,
                local_path,
                file_info,
                force_regenerate_etag=performed_download,
            ):
                print(f"✅ 已存在且通过校验: {local_path.name}")
                return {
                    "success": True,
                    "downloaded": performed_download,
                    "url_type": url_type,
                }
            else:
                local_path.unlink(missing_ok=True)

            if attempt >= max_attempts:
                # 下载失败
                print(f"🚫 达到最大重试次数，放弃下载: {local_path.name}")
                return {"success": False, "downloaded": False, "url_type": url_type}

            attempt += 1
            attempt_note = f"{attempt}/{max_attempts}次尝试"
            print(f"📥 开始下载: {local_path.name} | 来源: {url_type} | {attempt_note} | URL: {url}")
            final_source = url_type

            if url_type in ["LFS"]:
                try:
                    download_result = self.lfs_downloader.download_file(url, local_path)
                    performed_download = True

                    if download_result.success:
                        download_success = True
                    else:
                        status_code = download_result.status_code
                        if status_code in {401, 403, 404}:
                            print(
                                f"🔀 LFS 下载错误：{status_code=}, 尝试切换 HF 下载: {local_path.name}"
                            )
                            local_path.unlink(missing_ok=True)
                            control_file = Path(str(local_path) + ".aria2")
                            control_file.unlink(missing_ok=True)
                            url_type = "HF"
                        else:
                            message = download_result.message
                            if message:
                                print(
                                    f"⚠️ LFS 下载未完成: {local_path.name} | {message}"
                                )
                            else:
                                print(f"⚠️ LFS 下载未完成: {local_path.name}")
                except Exception as e:
                    performed_download = True
                    print(f"❌ LFS 下载异常: {local_path.name} | {e}")
                    print(traceback.format_exc())
            elif url_type in ["HF-URL"]:
                try:
                    download_result = self.hf_downloader.download_file(url, local_path)
                    performed_download = True

                    if download_result.success:
                        download_success = True
                    else:
                        status_code = download_result.status_code
                        if status_code in {401, 403, 404}:
                            print(
                                f"🔀 HF-URL 下载错误：{status_code=}, 尝试切换 HF 下载: {local_path.name}"
                            )
                            local_path.unlink(missing_ok=True)
                            control_file = Path(str(local_path) + ".aria2")
                            control_file.unlink(missing_ok=True)
                            url_type = "HF"
                        else:
                            message = download_result.message
                            if message:
                                print(
                                    f"⚠️ HF-URL 下载未完成: {local_path.name} | {message}"
                                )
                            else:
                                print(f"⚠️ HF-URL 下载未完成: {local_path.name}")
                except Exception as e:
                    performed_download = True
                    print(f"❌ LFS 下载异常: {local_path.name} | {e}")
                    print(traceback.format_exc())
            elif url_type in ["HF"]:
                try:
                    self.hf_api.hf_hub_download(
                        **hf_mirror_param, local_dir=local_dir, resume_download=True
                    )
                    download_success = True
                    performed_download = True
                except Exception as e:
                    performed_download = True
                    if (
                        hasattr(e, "response")
                        and e.response is not None
                        and e.response.status_code in [401, 403, 404]
                    ):
                        print(
                            f"🚫 HF 访问受限 ({e.response.status_code}): {local_path.name} | {e}"
                        )
                        return {
                            "success": False,
                            "downloaded": performed_download,
                            "url_type": url_type,
                        }
                    print(f"❌ HF下载异常: {local_path.name} | {e}")
            else:
                print(f"❌ 未知下载类型: {url_type} | {local_path.name}")
                return {"success": False, "downloaded": False, "url_type": url_type}

            if not download_success:
                if attempt < max_attempts:
                    wait_seconds = 2
                    print(
                        f"🔁 准备重试: {local_path.name} | 下一次尝试 {attempt + 1}/{max_attempts} | 等待 {wait_seconds}s"
                    )
                    time.sleep(wait_seconds)
                    continue
                print(f"🚫 放弃下载: {local_path.name} | 已达最大重试次数")
                return {
                    "success": False,
                    "downloaded": performed_download,
                    "url_type": final_source,
                }

            if download_success:
                if local_path.exists():
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    print(f"✅ 下载结束: {local_path.name} | {size_mb:.3f} MB")
                else:
                    print(f"✅ 下载结束: {local_path.name}")

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
        print(f"🔍 验证仓库: {repo_type} {repo_id} @ {revision}")
        try:
            self.hf_api.repo_info(repo_id, repo_type=repo_type, revision=revision)
        except Exception as e:
            print(f"❌ 仓库信息无效: {e}")
            return False

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # 获取文件列表
        files_info = self.get_repo_file_list(repo_id, repo_type, revision)

        if not files_info:
            print("❌ 未找到文件或无法获取文件列表")
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

        total_files = len(files_info)
        print(
            f"📂 文件统计: 共 {total_files} 个 | LFS: {len(lfs_files)} | 普通: {len(regular_files)}"
        )

        files_to_download = []

        for file_info in files_info:
            filename = file_info["filename"]
            local_path = local_dir / filename
            is_lfs = self.is_lfs_file(file_info)

            url, url_type, hf_mirror_param = self.build_download_url(
                repo_id, filename, repo_type, revision, is_lfs
            )
            source_icon = "🔗" if url_type == "LFS" else "🪞"
            print(f"{source_icon} 排队: {filename} | 来源: {url_type} | fileinfo: {file_info}")
            files_to_download.append(
                (url, local_path, file_info, url_type, hf_mirror_param)
            )

        print(f"\n🧾 任务总数: {len(files_to_download)}")

        if not files_to_download:
            print("✅ 所有文件均已通过校验，无需下载")
            return True

        print("🚀 启动下载任务")

        # 并发下载文件
        successful_downloads = 0
        failed_downloads = 0
        total_bytes_downloaded = 0
        start_time = time.time()
        lfs_downloads = 0
        hf_downloads = 0
        verified_without_downloads = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.download_and_verify_file,
                    url,
                    local_dir,
                    local_path,
                    file_info,
                    url_type,
                    hf_mirror_param,
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
                    # 检查是否被中断
                    if interrupted:
                        print(f"\n⚠️  检测到中断信号，正在取消剩余下载任务...")
                        # 取消所有未完成的任务
                        for f in future_to_task:
                            f.cancel()
                        break

                    try:
                        url, local_path, file_info, url_type, hf_mirror_param = (
                            future_to_task[future]
                        )
                        result = future.result()
                        if isinstance(result, dict):
                            if result.get("success"):
                                successful_downloads += 1
                                if result.get("downloaded"):
                                    if result.get("url_type") == "LFS":
                                        lfs_downloads += 1
                                    else:
                                        hf_downloads += 1
                                    if local_path.exists():
                                        total_bytes_downloaded += (
                                            local_path.stat().st_size
                                        )
                                else:
                                    verified_without_downloads += 1
                            else:
                                failed_downloads += 1
                                print(f"❌ 任务失败: {file_info['filename']}")
                        elif result:
                            successful_downloads += 1
                            if url_type == "LFS":
                                lfs_downloads += 1
                            else:
                                hf_downloads += 1
                            if local_path.exists():
                                total_bytes_downloaded += local_path.stat().st_size
                        else:
                            failed_downloads += 1
                            print(f"❌ 任务失败: {file_info['filename']}")
                    except Exception as e:
                        print(f"💥 任务异常: {local_path.name} | {e}")
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
                                    "已验证": verified_without_downloads,
                                    "平均速度": f"{avg_speed / (1024*1024):.1f} MB/s",
                                }
                            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_speed = total_bytes_downloaded / total_time if total_time > 0 else 0

        print(f"\n📊 下载统计:")
        print(f"  ✅ 成功: {successful_downloads}")
        print(f"    🔄 已存在验证: {verified_without_downloads}")
        print(f"    🔗 Xget下载: {lfs_downloads}")
        print(f"    🪞 镜像下载: {hf_downloads}")
        print(f"  ❌ 失败: {failed_downloads}")
        print(f"  📁 总计: {len(files_to_download)}")
        print(f"  💾 下载量: {total_bytes_downloaded / (1024*1024*1024):.2f} GB")
        print(f"  ⏱️  用时: {total_time:.1f} 秒")
        print(f"  🚀 平均速度: {avg_speed / (1024*1024):.1f} MB/s")

        return failed_downloads == 0


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Face 下载加速器",
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
        "--hf-url",
        default="https://xget.xi-xu.me/hf",
        help="HF 镜像URL，用于普通文件",
    )
    download_parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face 访问令牌 (可选)",
    )
    download_parser.add_argument(
        "--lfs-url",
        default="https://xget.xi-xu.me/hf",
        help="Xget 基础URL，用于 LFS 文件 (默认: https://xget.xi-xu.me/hf)",
    )
    download_parser.add_argument(
        "--hf-downloader",
        choices=["requests"],
        default="requests",
        help="下载核心",
    )
    download_parser.add_argument(
        "--lfs-downloader",
        choices=["requests"],
        default="requests",
        help="下载核心",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "download":
        try:
            downloader = HFDownloader(
                args.lfs_url, args.hf_url, args.hf_downloader, args.lfs_downloader, args.token
            )
        except Exception as e:
            print(f"❌ 初始化下载器失败: {e}")
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

        # 检查是否被中断
        if interrupted:
            print(f"\n⚠️  下载被用户中断 (Ctrl+C)")
            print("已下载的文件将保留在本地目录中")
            return 130  # 标准的中断退出码

        return 0 if success else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
