import datetime
import http.server
import ipaddress
import os
import socketserver
import ssl
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


HOST = "127.0.0.1"
PORT = 8443
HTML_NAME = "MER2dashboard.html"

def get_base_dir():
    if getattr(sys, "frozen", False):
        # PyInstaller 打包後執行 exe
        return Path(sys.executable).resolve().parent
    else:
        # 直接執行 .py
        return Path(__file__).resolve().parent

BASE_DIR = get_base_dir()

HTML_FILE = BASE_DIR / HTML_NAME

# 憑證放 temp，避免 onefile 解壓目錄被清掉或唯讀問題
RUNTIME_DIR = Path(tempfile.gettempdir()) / "mer2_https_runtime"
CERT_FILE = RUNTIME_DIR / "cert.pem"
KEY_FILE = RUNTIME_DIR / "key.pem"


def ensure_html_exists():
    if not HTML_FILE.exists():
        print(f"[錯誤] 找不到檔案：{HTML_FILE}")
        input("按 Enter 結束...")
        sys.exit(1)


def ensure_certificates():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    if CERT_FILE.exists() and KEY_FILE.exists():
        return

    print("[資訊] 正在用 Python 自動產生自簽 HTTPS 憑證...")

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "TW"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Local HTTPS"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    now = datetime.datetime.utcnow()

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"[完成] 憑證已建立：{CERT_FILE}")
    print(f"[完成] 私鑰已建立：{KEY_FILE}")


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def open_browser(url: str):
    time.sleep(1)
    webbrowser.open(url)


def main():
    ensure_html_exists()
    ensure_certificates()

    # 切到 html 所在目錄，讓 dashboard 內引用的相對路徑資源可正常載入
    os.chdir(HTML_FILE.parent)

    handler = http.server.SimpleHTTPRequestHandler

    try:
        with ReusableTCPServer((HOST, PORT), handler) as httpd:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

            url = f"https://{HOST}:{PORT}/{HTML_FILE.name}"

            print("=" * 60)
            print("HTTPS 本機伺服器已啟動")
            print(url)
            print("=" * 60)
            print("第一次開啟時，瀏覽器可能會顯示憑證警告。")
            print("這是自簽憑證的正常現象，手動選擇繼續即可。")
            print("關閉這個黑色視窗，就會停止伺服器。")
            print("=" * 60)

            threading.Thread(target=open_browser, args=(url,), daemon=True).start()
            httpd.serve_forever()

    except OSError as e:
        print(f"[錯誤] 無法啟動伺服器：{e}")
        print(f"可能是 {PORT} port 已被占用。")
        input("按 Enter 結束...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[資訊] 伺服器已停止。")


if __name__ == "__main__":
    main()
