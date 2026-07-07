"""Bridge Ollama's model store to our custom llama.cpp runtime.

Design: Ollama is used only as the model *catalog and download manager*. The
actual inference is done by our own `llama-server` binary pointed straight at the
GGUF blob that Ollama stored on disk. An Ollama "model" is an OCI-style manifest
whose `application/vnd.ollama.image.model` layer is a plain GGUF file living under
`<store>/blobs/sha256-<digest>`. This module turns an Ollama tag (e.g.
``gemma4:26b``) into that blob path so llama.cpp can load it directly — no copy,
no conversion, and Ollama keeps managing the bytes.
"""

import os
import json
from typing import Dict, List, Optional

MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
PARAMS_MEDIA_TYPE = "application/vnd.ollama.image.params"
TEMPLATE_MEDIA_TYPE = "application/vnd.ollama.image.template"

DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"


def find_ollama_root() -> Optional[str]:
    """Locate the Ollama model store.

    Order: explicit ROUTER_OLLAMA_ROOT, then OLLAMA_MODELS, then the known
    E:\\Ollama store, then the per-user default. Returns None if none exist.
    """
    candidates = [
        os.environ.get("ROUTER_OLLAMA_ROOT"),
        os.environ.get("OLLAMA_MODELS"),
        "E:\\Ollama",
        os.path.join(os.path.expanduser("~"), ".ollama", "models"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "manifests")):
            return c
    return None


def _parse_tag(tag: str) -> tuple:
    """Split a model reference into (registry, namespace, name, tag).

    Accepts forms like ``gemma4:26b``, ``library/gemma4:26b``,
    ``hf.co/unsloth/Qwen3-Coder-Next-GGUF:UD-Q4_K_XL``.
    """
    registry, namespace, name = DEFAULT_REGISTRY, DEFAULT_NAMESPACE, tag
    version = DEFAULT_TAG

    ref = tag
    if ":" in ref.split("/")[-1]:
        ref, version = ref.rsplit(":", 1)

    parts = ref.split("/")
    if len(parts) == 1:
        name = parts[0]
    elif len(parts) == 2:
        namespace, name = parts
    else:
        registry, namespace, name = parts[0], parts[1], "/".join(parts[2:])

    return registry, namespace, name, version


class OllamaResolver:
    """Resolves Ollama tags to on-disk GGUF blob paths."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or find_ollama_root()

    @property
    def available(self) -> bool:
        return bool(self.root)

    def _manifest_path(self, tag: str) -> Optional[str]:
        if not self.root:
            return None
        registry, namespace, name, version = _parse_tag(tag)
        path = os.path.join(
            self.root, "manifests", registry, namespace, name, version
        )
        return path if os.path.isfile(path) else None

    def _blob_path(self, digest: str) -> str:
        # digest is "sha256:<hex>"; on disk it's "sha256-<hex>"
        return os.path.join(self.root, "blobs", digest.replace(":", "-"))

    def resolve_gguf(self, tag: str) -> Optional[str]:
        """Return the absolute GGUF blob path for an Ollama tag, or None."""
        manifest_path = self._manifest_path(tag)
        if not manifest_path:
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == MODEL_MEDIA_TYPE:
                blob = self._blob_path(layer["digest"])
                return blob if os.path.isfile(blob) else None
        return None

    def resolve_params(self, tag: str) -> Dict:
        """Return the Ollama-stored generation params for a tag (may be empty)."""
        manifest_path = self._manifest_path(tag)
        if not manifest_path:
            return {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == PARAMS_MEDIA_TYPE:
                blob = self._blob_path(layer["digest"])
                try:
                    with open(blob, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError):
                    return {}
        return {}

    def list_models(self) -> List[str]:
        """List every Ollama tag present in the store as 'name:version'."""
        if not self.root:
            return []
        manifests_root = os.path.join(self.root, "manifests")
        found = []
        for dirpath, _, files in os.walk(manifests_root):
            for fname in files:
                # Reconstruct name:tag from the path under manifests/<reg>/<ns>/<name>/<tag>
                rel = os.path.relpath(os.path.join(dirpath, fname), manifests_root)
                parts = rel.replace("\\", "/").split("/")
                if len(parts) >= 4:
                    name = "/".join(parts[2:-1])
                    version = parts[-1]
                    found.append(f"{name}:{version}")
        return sorted(found)


# Module-level singleton for convenience.
resolver = OllamaResolver()
