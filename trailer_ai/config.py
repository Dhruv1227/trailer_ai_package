import os
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(path: str = ".env"):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            return
load_dotenv()
WRITE_SUBS = os.getenv("TRAILER_WRITE_SUBS", "1")
COOKIES_PATH = os.getenv("TRAILER_COOKIES", "").strip() or None
SLEEP_REQUESTS = os.getenv("TRAILER_SLEEP_REQUESTS", "").strip()
MAX_SLEEP_INTERVAL = os.getenv("TRAILER_MAX_SLEEP_INTERVAL", "").strip()
API_TOKEN = os.getenv("TRAILER_API_TOKEN", "").strip() or None
