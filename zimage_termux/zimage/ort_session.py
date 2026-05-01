"""Helper for building ONNX Runtime sessions on the QNN HTP backend.

Every session created here is verified to actually be running on
QNNExecutionProvider after construction. CPU fallback is disabled at the
session level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .npu import HtpInfo, NpuUnavailable


def make_session(
    model_path: Path,
    info: HtpInfo,
    *,
    extra_provider_opts: Mapping[str, str] | None = None,
) -> Any:
    """Build an InferenceSession on the QNN HTP backend.

    `model_path` may be either an ONNX file or a pre-compiled QNN context
    binary (.qnn.bin) — the QNN EP detects the format from the file header.
    """
    import onnxruntime as ort

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {model_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess_opts.log_severity_level = 3

    provider_opts: dict[str, str] = {
        "backend_path": str(info.backend_lib),
        "htp_arch": info.htp_arch,
        "enable_htp_fp16_precision": "1",
        "qnn_context_priority": "high",
    }
    if model_path.suffix == ".bin":
        provider_opts["qnn_context_cache_enable"] = "1"
        provider_opts["qnn_context_cache_path"] = str(model_path)
    if extra_provider_opts:
        provider_opts.update(extra_provider_opts)

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=[("QNNExecutionProvider", provider_opts)],
        )
    except Exception as exc:
        raise NpuUnavailable(
            f"Failed to build QNN session for {model_path}: {exc}"
        ) from exc

    providers = session.get_providers()
    if not providers or providers[0] != "QNNExecutionProvider":
        raise NpuUnavailable(
            f"Session for {model_path.name} is not on QNN "
            f"(providers={providers}). Refusing to continue."
        )
    return session
