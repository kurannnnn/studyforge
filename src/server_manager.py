"""
Starts and stops llama.cpp's llama-server for this project.

ensure_running() does everything that is needed, and only what is missing:
  1. if the server already answers, it does nothing;
  2. if llama.cpp is not installed yet, it downloads it (once);
  3. it finds the GGUF model in the Hugging Face cache (downloading it once);
  4. it starts the server and waits until it is ready.
"""

import glob
import json
import os
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import requests

from src.config import (
    GGUF_FILENAME,
    GGUF_PATH,
    GGUF_REPO_ID,
    SERVER_CTX_PER_SLOT,
    SERVER_SLOTS,
    SERVER_URL,
)

ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = ROOT / "llama_server"
LOG_FILE = ROOT / "llama_server.log"

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
CUDA_TAG = "win-cuda-12.4-x64"  # the CUDA 12.4 build runs on any recent NVIDIA driver
PORT = SERVER_URL.rsplit(":", 1)[-1]

_slots_cache = None


def is_up() -> bool:
    try:
        return requests.get(f"{SERVER_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def get_slots() -> int:
    """How many requests the running server handles at once."""

    global _slots_cache

    if _slots_cache is None:
        try:
            props = requests.get(f"{SERVER_URL}/props", timeout=3).json()
            _slots_cache = max(1, int(props.get("total_slots", 1)))
        except Exception:
            return 1

    return _slots_cache


def _running_context():
    """Context size per slot of the running server, or None if it can't be read."""

    try:
        props = requests.get(f"{SERVER_URL}/props", timeout=3).json()
        return int(props["default_generation_settings"]["n_ctx"])
    except Exception:
        return None


def _find_exe():
    found = list(SERVER_DIR.rglob("llama-server.exe"))
    return found[0] if found else None


def _download_llama_cpp(log) -> None:

    log("Downloading llama.cpp (one time only, roughly 1 GB)...")

    request = urllib.request.Request(
        RELEASES_API, headers={"User-Agent": "ai-learning-companion"}
    )
    releases = json.load(urllib.request.urlopen(request, timeout=60))

    # Recent builds are marked "pre-release", so look through the list
    # and take the newest one that has the Windows CUDA files.
    for release in releases:
        assets = {a["name"]: a["browser_download_url"] for a in release["assets"]}
        program = [
            n for n in assets
            if n.endswith(f"bin-{CUDA_TAG}.zip") and not n.startswith("cudart")
        ]
        runtime = [n for n in assets if n.startswith("cudart") and CUDA_TAG in n]
        if program and runtime:
            break
    else:
        raise RuntimeError(
            f"No recent llama.cpp release has a {CUDA_TAG} build. Download one "
            f"from the llama.cpp releases page and unzip it into {SERVER_DIR}."
        )

    SERVER_DIR.mkdir(exist_ok=True)

    for name in program + runtime:
        target = SERVER_DIR / name
        log(f"Downloading {name} ...")
        urllib.request.urlretrieve(assets[name], target)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(SERVER_DIR)
        target.unlink()


def _find_model(log) -> str:

    if GGUF_PATH and os.path.exists(GGUF_PATH):
        return GGUF_PATH

    pattern = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--" + GGUF_REPO_ID.replace("/", "--"), "snapshots", "*", GGUF_FILENAME,
    )
    hits = glob.glob(pattern)

    if hits:
        return hits[0]

    log("Downloading the model file (one time only, about 2.5 GB)...")

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=GGUF_REPO_ID, filename=GGUF_FILENAME)


def _command(exe: Path, model: str, slots: int) -> list[str]:

    return [
        str(exe),
        "-m", model,
        "-ngl", "99",                              # every layer on the GPU
        "-np", str(slots),                         # requests handled at once
        "-c", str(SERVER_CTX_PER_SLOT * slots),    # total context, split per slot
        "-b", "512", "-ub", "512",                 # small compute buffers for a 4 GB card
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--host", "127.0.0.1",
        "--port", PORT,
    ]


def gpu_memory():
    """(used MiB, total MiB) from nvidia-smi, or None if unavailable."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        used, total = (int(v) for v in out.split(","))
        return used, total
    except Exception:
        return None


def _try_start(exe: Path, model: str, slots: int, log) -> bool:

    handle = open(LOG_FILE, "w", encoding="utf-8")

    process = subprocess.Popen(
        _command(exe, model, slots),
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(exe.parent),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    started = time.time()

    while time.time() - started < 240:

        if process.poll() is not None:
            return False

        if is_up():
            return True

        time.sleep(1.5)

    process.kill()
    return False


def ensure_running(log=print) -> dict:
    """Make sure llama-server is up. Returns {"slots": n, "gpu": (used, total)}."""

    global _slots_cache

    if is_up():

        context = _running_context()

        # A server left running from an older setup may have too small a context.
        # Replace it so the current settings take effect.
        if context is None or context >= SERVER_CTX_PER_SLOT:
            return {"slots": get_slots(), "gpu": gpu_memory()}

        log(f"The running server has a context of {context} tokens per slot; restarting it with {SERVER_CTX_PER_SLOT}.")
        _slots_cache = None
        stop_server()
        time.sleep(3)

    exe = _find_exe()

    if exe is None:
        _download_llama_cpp(log)
        exe = _find_exe()

        if exe is None:
            raise RuntimeError("llama-server.exe was not found after downloading llama.cpp.")

    model = _find_model(log)

    log("Starting the model server...")

    # Try the configured number of slots first, then a single slot, which
    # needs the least GPU memory.
    for slots in dict.fromkeys([SERVER_SLOTS, 1]):

        if _try_start(exe, model, slots, log):
            _slots_cache = slots
            return {"slots": slots, "gpu": gpu_memory()}

        log(f"The server did not start with {slots} slot(s).")

    tail = "".join(open(LOG_FILE, encoding="utf-8", errors="ignore").readlines()[-20:])

    raise RuntimeError(
        "llama-server could not start. If Windows says a file is blocked, "
        "see Smart App Control. Last lines of llama_server.log:\n" + tail
    )


def stop_server() -> None:
    """Stop llama-server so it releases the GPU."""

    if os.name == "nt":
        command = ["taskkill", "/IM", "llama-server.exe", "/F"]
    else:
        command = ["pkill", "-f", "llama-server"]

    try:
        subprocess.run(command, capture_output=True, timeout=15)
    except Exception:
        pass
