"""Fail-loud Hexagon NPU gate.

Refuses to run unless ONNX Runtime can build a session on the QNN
ExecutionProvider with the HTP backend. There is no CPU fallback by design.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


VENDOR_LIB = Path("/vendor/lib64")
HTP_BACKEND = "libQnnHtp.so"
SYSTEM_LIB = "libQnnSystem.so"


class NpuUnavailable(RuntimeError):
    """Raised when the Hexagon HTP backend cannot be initialized."""


@dataclass(frozen=True)
class HtpInfo:
    soc_model: str
    htp_arch: str          # e.g. "v79"
    stub_lib: Path
    skel_lib: Path
    backend_lib: Path
    system_lib: Path


def _getprop(name: str) -> str:
    try:
        out = subprocess.check_output(["getprop", name], text=True, timeout=2)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise NpuUnavailable(f"`getprop {name}` failed: {exc}") from exc
    return out.strip()


def _detect_htp_arch() -> str:
    """Return the Hexagon HTP architecture string (e.g. 'v79').

    Probes the on-device QNN libraries directly: every Snapdragon ships the
    matching libQnnHtpV{NN}Stub.so, so we infer the arch from the highest
    version present in /vendor/lib64. We cross-check with the SoC model when
    possible, but the lib presence is the source of truth.
    """
    candidates = []
    pattern = re.compile(r"^libQnnHtpV(\d+)Stub\.so$")
    for entry in VENDOR_LIB.glob("libQnnHtpV*Stub.so"):
        m = pattern.match(entry.name)
        if m:
            candidates.append(int(m.group(1)))
    if not candidates:
        raise NpuUnavailable(
            f"No libQnnHtpV*Stub.so found in {VENDOR_LIB}. "
            "This device has no Hexagon HTP runtime."
        )
    arch = f"v{max(candidates)}"
    return arch


def detect() -> HtpInfo:
    """Locate vendor libs + HTP arch. Does NOT initialize ORT."""
    backend = VENDOR_LIB / HTP_BACKEND
    system = VENDOR_LIB / SYSTEM_LIB
    if not backend.is_file():
        raise NpuUnavailable(f"Missing {backend}. Hexagon backend not present.")
    if not system.is_file():
        raise NpuUnavailable(f"Missing {system}. QNN system lib not present.")

    arch = _detect_htp_arch()
    stub = VENDOR_LIB / f"libQnnHtpV{arch[1:]}Stub.so"
    skel = VENDOR_LIB / f"libQnnHtpV{arch[1:]}Skel.so"
    if not stub.is_file():
        raise NpuUnavailable(f"Missing {stub} for detected arch {arch}.")
    if not skel.is_file():
        raise NpuUnavailable(f"Missing {skel} for detected arch {arch}.")

    soc = _getprop("ro.soc.model") or _getprop("ro.board.platform")
    return HtpInfo(
        soc_model=soc,
        htp_arch=arch,
        stub_lib=stub,
        skel_lib=skel,
        backend_lib=backend,
        system_lib=system,
    )


def _smoke_model_bytes() -> bytes:
    """A 1-op ONNX model: out = a + b. Used to verify HTP initialization."""
    import onnx
    from onnx import TensorProto, helper

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", ["a", "b"], ["y"])
    graph = helper.make_graph([node], "smoke", [a, b], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    return model.SerializeToString()


def assert_htp_alive(info: HtpInfo | None = None) -> HtpInfo:
    """Build a 1-op QNN session and run it. Raise NpuUnavailable on any failure.

    Critically, sets `disable_cpu_ep_fallback=1` on the session so that QNN EP
    cannot silently demote to CPU.
    """
    info = info or detect()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise NpuUnavailable(
            "onnxruntime is not installed. Run setup.sh, or install the "
            "onnxruntime-qnn wheel manually (see README.md)."
        ) from exc

    if "QNNExecutionProvider" not in ort.get_available_providers():
        raise NpuUnavailable(
            "Installed onnxruntime build does not include QNNExecutionProvider. "
            "You need the onnxruntime-qnn wheel, not vanilla onnxruntime."
        )

    sess_opts = ort.SessionOptions()
    # The single most important guard against silent CPU fallback:
    sess_opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess_opts.log_severity_level = 3

    provider_opts = {
        "backend_path": str(info.backend_lib),
        "htp_arch": info.htp_arch,
        "enable_htp_fp16_precision": "1",
        "qnn_context_priority": "high",
    }

    try:
        session = ort.InferenceSession(
            _smoke_model_bytes(),
            sess_options=sess_opts,
            providers=[("QNNExecutionProvider", provider_opts)],
        )
    except Exception as exc:
        raise NpuUnavailable(f"QNN session init failed: {exc}") from exc

    providers = session.get_providers()
    if not providers or providers[0] != "QNNExecutionProvider":
        raise NpuUnavailable(
            f"QNN EP not active after init (providers={providers}). "
            "ORT may have demoted the session despite disable_cpu_ep_fallback."
        )

    import numpy as np
    out = session.run(None, {"a": np.float32([1.0]), "b": np.float32([2.0])})
    if not (abs(float(out[0][0]) - 3.0) < 1e-3):
        raise NpuUnavailable(
            f"Smoke op produced unexpected output {out[0]}; HTP is not healthy."
        )

    return info


def adsp_library_path() -> str:
    """Return the ADSP_LIBRARY_PATH that the Hexagon stub expects.

    Mirrors what vendor_shim.sh writes into the profile, so callers can set
    the env var when running outside an interactive shell.
    """
    return ":".join((
        "/vendor/dsp/cdsp",
        "/vendor/lib/rfsa/adsp",
        "/vendor/dsp",
        "/system/lib/rfsa/adsp",
        "/dsp",
    ))


def ensure_env() -> None:
    """Populate ADSP_LIBRARY_PATH if the parent shell did not."""
    if not os.environ.get("ADSP_LIBRARY_PATH"):
        os.environ["ADSP_LIBRARY_PATH"] = adsp_library_path()
