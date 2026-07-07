"""Hardware-aware helpers: pick the right llama-server binary and size GPU
offload to the actual GPU on this box.

Two jobs:
  1. ``find_llama_server()`` — prefer a CUDA-enabled build (``build-cuda``) over
     the CPU-only ``build`` so experts use the GPU automatically once the CUDA
     build exists, with no code change.
  2. ``plan_gpu_layers()`` — query free VRAM via ``nvidia-smi`` and decide how
     many transformer layers to offload for a given GGUF so a model that does
     not fully fit still uses the GPU for as much as fits (instead of blindly
     passing 99 and OOMing, or 0 and running pure CPU).
"""

import os
import struct
import subprocess
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_DIR = os.path.join(REPO_ROOT, "llama.cpp")

# Candidate llama-server locations, most-preferred first. CUDA builds win.
_BINARY_CANDIDATES = [
    os.path.join(LLAMA_DIR, "build-cuda", "bin", "Release", "llama-server.exe"),
    os.path.join(LLAMA_DIR, "build-cuda", "bin", "llama-server.exe"),
    os.path.join(LLAMA_DIR, "build", "bin", "Release", "llama-server.exe"),
    os.path.join(LLAMA_DIR, "build", "bin", "llama-server.exe"),
]

# GGUF value-type sizes for the fixed-width types we may skip over.
_GGUF_SCALAR_SIZE = {
    0: 1, 1: 1,   # int8/uint8
    2: 2, 3: 2,   # int16/uint16
    4: 4, 5: 4,   # int32/uint32
    6: 4,         # float32
    7: 1,         # bool
    10: 8, 11: 8, # int64/uint64
    12: 8,        # float64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def find_llama_server(explicit: Optional[str] = None) -> Optional[str]:
    """Return the best available llama-server.exe path.

    Order: explicit arg, ``LLAMA_SERVER_BIN`` env, then the candidate list
    (CUDA builds before the CPU build). Returns None if none exist.
    """
    for cand in [explicit, os.environ.get("LLAMA_SERVER_BIN"), *_BINARY_CANDIDATES]:
        # Require a non-empty file: an interrupted link can leave a 0-byte stub.
        if cand and os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return cand
    return None


def has_cuda_build() -> bool:
    """True if the selected binary comes from a CUDA build dir."""
    b = find_llama_server()
    return bool(b and "build-cuda" in b.replace("\\", "/"))


def _cuda_dll_dirs() -> list:
    """Directories holding the CUDA runtime DLLs (cudart/cublas).

    CUDA 13 relocated these to ``<toolkit>\\bin\\x64`` (older layouts used
    ``bin``). We include both so ggml-cuda.dll can load regardless of version.
    """
    dirs = []
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_PATH_V13_3")
    roots = [cuda_path] if cuda_path else []
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(base):
        # Newest version dir first.
        for name in sorted(os.listdir(base), reverse=True):
            roots.append(os.path.join(base, name))
    for root in roots:
        if not root:
            continue
        for sub in (os.path.join("bin", "x64"), "bin"):
            d = os.path.join(root, sub)
            if os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    return dirs


def launch_env() -> dict:
    """Return an environment dict for spawning llama-server with CUDA DLLs.

    Prepends the CUDA runtime DLL dirs and the binary's own dir to PATH so the
    server (and ggml-cuda.dll) resolve their dependencies without a global
    PATH change. Safe to use even on the CPU build (dirs simply won't exist).
    """
    env = dict(os.environ)
    extra = list(_cuda_dll_dirs())
    binary = find_llama_server()
    if binary:
        extra.append(os.path.dirname(binary))
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def gpu_free_vram_mb() -> Optional[int]:
    """Free VRAM on GPU 0 in MB via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _read_gguf_block_count(path: str) -> Optional[int]:
    """Parse a GGUF header enough to find ``*.block_count`` (layer count).

    Returns None if the file isn't GGUF or the key is absent. Best-effort: we
    walk the metadata KV table skipping values by type until we hit the key.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            # tensor_count and metadata_kv_count are u64 in v2/v3.
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", errors="replace")

            def skip_value(vtype: int) -> None:
                if vtype in _GGUF_SCALAR_SIZE:
                    f.read(_GGUF_SCALAR_SIZE[vtype])
                elif vtype == _GGUF_STRING:
                    n = struct.unpack("<Q", f.read(8))[0]
                    f.read(n)
                elif vtype == _GGUF_ARRAY:
                    elem_type = struct.unpack("<I", f.read(4))[0]
                    count = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(count):
                        skip_value(elem_type)
                else:
                    raise ValueError(f"unknown gguf value type {vtype}")

            if version < 2:
                return None
            for _ in range(kv_count):
                key = read_str()
                vtype = struct.unpack("<I", f.read(4))[0]
                if key.endswith(".block_count"):
                    if vtype in (4, 5):  # int32/uint32
                        return struct.unpack("<I", f.read(4))[0]
                    if vtype in (10, 11):  # int64/uint64
                        return struct.unpack("<Q", f.read(8))[0]
                    skip_value(vtype)
                    return None
                skip_value(vtype)
    except (OSError, struct.error, ValueError):
        return None
    return None


def plan_gpu_layers(
    gguf_path: str,
    default: int = 99,
    reserve_mb: int = 2048,
) -> int:
    """Decide ``--n-gpu-layers`` for a model given current free VRAM.

    - No CUDA build or no GPU info → 0 (pure CPU; offload flag is ignored anyway
      on a CPU-only binary, but 0 keeps logs honest).
    - Model + KV/context reserve fits in free VRAM → ``default`` (offload all).
    - Otherwise scale layers by the fraction of the model that fits, using the
      GGUF's real block count (falls back to ``default`` if unreadable).
    """
    if not has_cuda_build():
        return 0
    free_mb = gpu_free_vram_mb()
    if free_mb is None:
        return default

    try:
        model_mb = os.path.getsize(gguf_path) / (1024 * 1024)
    except OSError:
        return default

    budget_mb = max(0, free_mb - reserve_mb)
    if model_mb <= budget_mb:
        return default  # whole model fits, offload everything

    blocks = _read_gguf_block_count(gguf_path)
    if not blocks:
        # Unknown layer count: offload a proportional guess against a nominal 48.
        blocks = 48
    fraction = budget_mb / model_mb if model_mb else 0
    layers = int(blocks * fraction)
    return max(0, min(layers, blocks))
