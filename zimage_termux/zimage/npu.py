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


def _search_dirs() -> list[Path]:
    """Where to look for QNN runtime libraries.

    Stock Samsung firmware does NOT ship the QNN libraries in /vendor/lib64
    — Qualcomm's QAIRT runtime libs (libQnnHtp*.so, libQnnSystem.so) come
    from the QAIRT SDK and apps that use the Hexagon NPU bundle them in
    their APK. So we search Termux's $PREFIX/lib first (where the user
    sideloads them — see compile/bundle_qnn_runtime.sh), and fall back to
    /vendor/lib64 in case a vendor build does ship them.
    """
    out: list[Path] = []
    prefix = os.environ.get("PREFIX")
    if prefix:
        out.append(Path(prefix) / "lib")
    out.append(VENDOR_LIB)
    return [d for d in out if d.is_dir()]


def _find_lib(name: str) -> Path | None:
    for d in _search_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


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


def _detect_htp_arch() -> tuple[str, Path]:
    """Return (htp_arch, dir_where_stub_was_found).

    Every Snapdragon arch ships a matching libQnnHtpV{NN}Stub.so. We pick
    the highest-numbered stub across all search dirs, and remember which
    directory it came from so the matching skel/system libs are taken
    from the same source.
    """
    pattern = re.compile(r"^libQnnHtpV(\d+)Stub\.so$")
    best: tuple[int, Path] | None = None
    searched: list[Path] = []
    for d in _search_dirs():
        searched.append(d)
        for entry in d.glob("libQnnHtpV*Stub.so"):
            m = pattern.match(entry.name)
            if not m:
                continue
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, d)
    if best is None:
        raise NpuUnavailable(
            "No libQnnHtpV*Stub.so found in: "
            + ", ".join(str(p) for p in searched)
            + ". Sideload the QAIRT runtime libs into "
            + (os.environ.get("PREFIX", "$PREFIX") + "/lib")
            + " (see compile/bundle_qnn_runtime.sh and the README)."
        )
    return f"v{best[0]}", best[1]


def detect() -> HtpInfo:
    """Locate QNN libs + HTP arch. Does NOT initialize ORT."""
    arch, source_dir = _detect_htp_arch()
    backend = source_dir / HTP_BACKEND
    system = source_dir / SYSTEM_LIB
    stub = source_dir / f"libQnnHtpV{arch[1:]}Stub.so"
    skel = source_dir / f"libQnnHtpV{arch[1:]}Skel.so"
    for p, label in (
        (backend, HTP_BACKEND),
        (system, SYSTEM_LIB),
        (stub, stub.name),
        (skel, skel.name),
    ):
        if not p.is_file():
            raise NpuUnavailable(
                f"Missing {p}. Sideload it from QAIRT SDK into {source_dir} "
                "(see compile/bundle_qnn_runtime.sh and the README)."
            )

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
