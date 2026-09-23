# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vllm_omni.model_executor.models.cosyvoice3 import flow_estimator_trt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _temporary_plans(plan_path: Path) -> list[Path]:
    return list(plan_path.parent.glob(f"{plan_path.name}.tmp.*"))


def test_write_plan_cleans_up_after_replace_failure(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    plan_path.write_bytes(b"existing plan")
    replace_error = OSError("replace failed")

    def fail_replace(source, destination):
        raise replace_error

    monkeypatch.setattr(flow_estimator_trt.os, "replace", fail_replace)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is replace_error
    assert plan_path.read_bytes() == b"existing plan"
    assert _temporary_plans(plan_path) == []


def test_write_plan_preserves_replace_error_when_cleanup_fails(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    replace_error = OSError("replace failed")

    def fail_replace(source, destination):
        raise replace_error

    def fail_unlink(path):
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(flow_estimator_trt.os, "replace", fail_replace)
    monkeypatch.setattr(flow_estimator_trt.os, "unlink", fail_unlink)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is replace_error
    assert len(_temporary_plans(plan_path)) == 1


def test_write_plan_cleans_up_after_write_failure(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    write_error = OSError("write failed")
    real_open = open

    class FailingWriter:
        def __init__(self, path, mode):
            self.file = real_open(path, mode)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.file.close()

        def write(self, data):
            self.file.write(data[:1])
            raise write_error

    monkeypatch.setattr(flow_estimator_trt, "open", FailingWriter, raising=False)

    with pytest.raises(OSError) as exc_info:
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert exc_info.value is write_error
    assert not plan_path.exists()
    assert _temporary_plans(plan_path) == []


def test_write_plan_does_not_remove_a_colliding_temporary_file(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    token = "0" * 32
    temporary_path = Path(f"{plan_path}.tmp.{flow_estimator_trt.os.getpid()}.{token}")
    temporary_path.write_bytes(b"another writer")
    monkeypatch.setattr(flow_estimator_trt.uuid, "uuid4", lambda: flow_estimator_trt.uuid.UUID(hex=token))

    with pytest.raises(FileExistsError):
        flow_estimator_trt._write_plan_atomically(b"new plan", str(plan_path))

    assert temporary_path.read_bytes() == b"another writer"
    assert not plan_path.exists()


def test_write_plan_supports_concurrent_publication(tmp_path, monkeypatch):
    plan_path = tmp_path / "flow.plan"
    payloads = (b"a" * 4096, b"b" * 4096)
    barrier = threading.Barrier(len(payloads))
    source_paths = []
    source_paths_lock = threading.Lock()
    real_replace = flow_estimator_trt.os.replace

    def synchronized_replace(source, destination):
        with source_paths_lock:
            source_paths.append(Path(source))
        barrier.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(flow_estimator_trt.os, "replace", synchronized_replace)

    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [
            executor.submit(flow_estimator_trt._write_plan_atomically, payload, str(plan_path)) for payload in payloads
        ]
        for future in futures:
            future.result(timeout=10)

    assert len(set(source_paths)) == len(payloads)
    assert plan_path.read_bytes() in payloads
    assert _temporary_plans(plan_path) == []


class TestExportCacheKey:
    """Exported ONNX bakes in the checkpoint's weights, so its cache path must
    change with the checkpoint even when the export directory is shared."""

    def test_fingerprint_differs_across_checkpoint_dirs(self, tmp_path):
        a, b = tmp_path / "ckpt_a", tmp_path / "ckpt_b"
        for d in (a, b):
            d.mkdir()
            (d / "flow.pt").write_bytes(b"same bytes")
        assert flow_estimator_trt.flow_checkpoint_fingerprint(str(a)) != flow_estimator_trt.flow_checkpoint_fingerprint(
            str(b)
        )

    def test_fingerprint_changes_when_weights_change_in_place(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "flow.pt").write_bytes(b"v1")
        before = flow_estimator_trt.flow_checkpoint_fingerprint(str(d))
        assert before == flow_estimator_trt.flow_checkpoint_fingerprint(str(d))
        (d / "flow.pt").write_bytes(b"v2 longer")
        assert flow_estimator_trt.flow_checkpoint_fingerprint(str(d)) != before

    def test_fingerprint_is_stable_without_weights(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        assert flow_estimator_trt.flow_checkpoint_fingerprint(str(d)) == flow_estimator_trt.flow_checkpoint_fingerprint(
            str(d)
        )

    def test_fingerprint_changes_with_export_version(self, tmp_path, monkeypatch):
        before = flow_estimator_trt.flow_checkpoint_fingerprint(str(tmp_path))
        monkeypatch.setattr(flow_estimator_trt, "_CHUNK_MASK_EXPORT_VERSION", 999)
        assert flow_estimator_trt.flow_checkpoint_fingerprint(str(tmp_path)) != before

    def test_onnx_path_carries_the_key(self, tmp_path):
        keyed = flow_estimator_trt.chunk_mask_estimator_onnx_path(str(tmp_path), fp16=True, cache_key="abc123")
        plain = flow_estimator_trt.chunk_mask_estimator_onnx_path(str(tmp_path), fp16=True, cache_key=None)
        assert keyed != plain
        assert keyed.endswith(".abc123.onnx") and "chunk_mask.autocast_fp16" in keyed
        assert plain.endswith("flow.decoder.estimator.chunk_mask.autocast_fp16.onnx")
