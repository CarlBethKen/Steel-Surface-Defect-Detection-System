#!/usr/bin/env python3
"""
钢铁缺陷检测系统 - 一键启动器
运行此文件即可启动整个系统并打开网页
"""

import os
import sys
import time
import webbrowser
import subprocess
import socket
import signal
from pathlib import Path


class SystemLauncher:
    """系统启动器"""

    def __init__(self):
        self.project_dir = Path(__file__).parent.absolute()
        self.host = "localhost"
        self.port = 8000
        self.server_process = None
        self.system_url = f"http://{self.host}:{self.port}/"

    def print_header(self):
        """打印启动头"""
        print("\n" + "="*60)
        print("🚀 钢铁缺陷检测系统")
        print("="*60 + "\n")

    def check_python(self):
        """检查Python版本"""
        print("📌 检查Python环境...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python版本过低 (当前: {version.major}.{version.minor})")
            print("   需要 Python 3.8 或以上")
            return False
        print(f"✓ Python {version.major}.{version.minor} 已就绪\n")
        return True

    def check_dependencies(self):
        """检查依赖"""
        print("📌 检查依赖...")
        dependencies = ["fastapi", "uvicorn", "sqlalchemy", "cv2", "torch"]
        missing = []

        for dep in dependencies:
            try:
                if dep == "cv2":
                    import cv2
                else:
                    __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            print(f"⚠️  缺失依赖: {', '.join(missing)}")
            print(f"   请运行: pip install {' '.join(missing)}")
            return False

        print("✓ 所有依赖已安装\n")
        return True

    def check_port_available(self):
        """检查端口是否可用"""
        print(f"📌 检查端口 {self.port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((self.host, self.port))
        sock.close()

        if result == 0:
            print(f"❌ 端口 {self.port} 已被占用")
            print("   请停止其他应用或更改端口号")
            return False

        print(f"✓ 端口 {self.port} 可用\n")
        return True

    def check_files(self):
        """检查必要文件"""
        print("📌 检查文件...")

        required_files = [
            "app.py",
            "templates/system.html",
            "requirements.txt"
        ]

        required_dirs = [
            "core",
            "templates",
            "static",
            "models"
        ]

        # 检查文件
        for file in required_files:
            file_path = self.project_dir / file
            if not file_path.exists():
                print(f"❌ 找不到文件: {file}")
                return False

        # 检查目录
        for dir_name in required_dirs:
            dir_path = self.project_dir / dir_name
            if not dir_path.exists():
                print(f"❌ 找不到目录: {dir_name}")
                return False

        print("✓ 所有文件已就绪\n")
        return True

    def start_server(self):
        """启动FastAPI服务器"""
        print("📌 启动后端服务器...")

        try:
            # 启动Uvicorn服务器
            self.server_process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "app:app",
                    "--host", self.host,
                    "--port", str(self.port),
                    "--reload"
                ],
                cwd=str(self.project_dir),
                stdout=None,  # 让日志直接输出到终端
                stderr=None
            )

            print(f"✓ 服务器进程已启动 (PID: {self.server_process.pid})\n")
            return True
        except Exception as e:
            print(f"❌ 启动服务器失败: {e}\n")
            return False

    def wait_for_server(self, max_attempts=30):
        """等待服务器就绪"""
        print("📌 等待服务器就绪...")

        import urllib.request
        import urllib.error

        for attempt in range(max_attempts):
            try:
                response = urllib.request.urlopen(
                    self.system_url + "api/statistics",
                    timeout=2
                )
                if response.status == 200:
                    print("✓ 服务器已就绪\n")
                    return True
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass

            if attempt < max_attempts - 1:
                print(f"  检查中... ({attempt + 1}/{max_attempts})")
                time.sleep(1)

        print("⚠️  服务器启动可能较慢，继续等待...\n")
        return True

    def open_browser(self):
        """打开浏览器"""
        print("📌 打开网页...")

        # 打开 FastAPI 渲染的系统页面
        url = f"http://{self.host}:{self.port}/system"
        try:
            webbrowser.open(url)
            print("✓ 网页已打开\n")
            return True
        except Exception as e:
            print(f"⚠️  无法自动打开浏览器: {e}")
            print(f"   请手动访问: {url}\n")
            return True

    def print_info(self):
        """打印系统信息"""
        print("="*60)
        print("✅ 系统已启动！")
        print("="*60)
        print()
        print("🌐 系统地址:")
        print(f"   http://{self.host}:{self.port}/")
        print()
        print("📌 使用说明:")
        print("   1. 系统网页已打开")
        print("   2. 点击'启动系统'进入应用")
        print("   3. 点击'进入系统'开始使用")
        print()
        print("⏹️  停止系统:")
        print("   按 Ctrl+C 停止系统")
        print()
        print("="*60)
        print()

    def handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        print("\n\n📌 正在关闭系统...")
        self.shutdown()
        print("✓ 系统已关闭")
        sys.exit(0)

    def shutdown(self):
        """关闭系统"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def run(self):
        """运行启动流程"""
        try:
            self.print_header()

            # 只检查端口和文件
            if not self.check_files():
                return False

            if not self.check_port_available():
                return False

            if not self.start_server():
                return False

            self.open_browser()

            self.print_info()

            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)

            while True:
                time.sleep(1)

            return True

        except KeyboardInterrupt:
            print("\n\n📌 正在关闭系统...")
            self.shutdown()
            print("✓ 系统已关闭")
            return True

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            self.shutdown()
            return False


def main():
    """主函数"""
    launcher = SystemLauncher()

    # 如果运行失败则等待用户按键再退出
    if not launcher.run():
        print("\n按任意键退出...")
        input()
        sys.exit(1)


if __name__ == "__main__":
    main()

