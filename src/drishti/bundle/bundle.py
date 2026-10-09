"""The project bundle: an on-disk, versioned, content-hashed directory that is the ONLY interface
between stages. Because stages exchange data through the bundle, any stage can run locally or on a
cloud GPU and the next stage continues from what it wrote (docs/implementation/02-SYSTEM-DESIGN.md).
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from .hashing import sha256_file
from .manifest import BUNDLE_VERSION, InputRef, Manifest

MANIFEST_NAME = "manifest.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


class Bundle:
    """Filesystem-backed run bundle. Construct via :meth:`create` or :meth:`open`."""

    def __init__(self, path: Path, manifest: Manifest) -> None:
        self.path = path
        self.manifest = manifest

    # -- lifecycle -------------------------------------------------------------
    @classmethod
    def create(
        cls,
        output_root: str | Path,
        run_id: str | None = None,
        dataset_name: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        dataset_snapshot: dict[str, Any] | None = None,
        seeds: dict[str, int] | None = None,
    ) -> Bundle:
        run_id = run_id or new_run_id()
        path = Path(output_root) / run_id
        path.mkdir(parents=True, exist_ok=True)
        for sub in ("inputs", "logs", "report"):
            (path / sub).mkdir(exist_ok=True)
        manifest = Manifest(
            bundle_version=BUNDLE_VERSION,
            drishti_version=__version__,
            run_id=run_id,
            created_utc=_utc_now(),
            dataset_name=dataset_name,
            config_snapshot=config_snapshot or {},
            dataset_snapshot=dataset_snapshot or {},
            seeds=seeds or {},
        )
        b = cls(path, manifest)
        b.save_manifest()
        return b

    @classmethod
    def open(cls, path: str | Path) -> Bundle:
        path = Path(path)
        mpath = path / MANIFEST_NAME
        if not mpath.is_file():
            raise FileNotFoundError(f"no bundle manifest at {mpath}")
        manifest = Manifest.model_validate_json(mpath.read_text(encoding="utf-8"))
        return cls(path, manifest)

    # -- manifest --------------------------------------------------------------
    def save_manifest(self) -> None:
        tmp = self.path / (MANIFEST_NAME + ".tmp")
        tmp.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path / MANIFEST_NAME)  # atomic

    # -- paths / artifacts -----------------------------------------------------
    def stage_dir(self, name: str) -> Path:
        d = self.path / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def artifact_path(self, relpath: str | Path) -> Path:
        return self.path / relpath

    def relpath(self, abspath: str | Path) -> str:
        return str(Path(abspath).relative_to(self.path)).replace(os.sep, "/")

    def write_json(self, relpath: str | Path, obj: Any) -> str:
        dest = self.path / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return self.relpath(dest)

    def read_json(self, relpath: str | Path) -> Any:
        return json.loads((self.path / relpath).read_text(encoding="utf-8"))

    def exists(self, relpath: str | Path) -> bool:
        return (self.path / relpath).exists()

    # -- inputs ----------------------------------------------------------------
    def register_input(self, path: str | Path | None, role: str, required: bool) -> InputRef:
        """Hash + record an input file. Missing required input -> ValueError (fail loudly)."""
        if path is None:
            if required:
                raise ValueError(f"required input '{role}' is missing (no path provided)")
            ref = InputRef(path="", role=role, present=False)
            self.manifest.inputs.append(ref)
            return ref
        p = Path(path)
        if not p.is_file():
            if required:
                raise ValueError(f"required input '{role}' not found at: {p}")
            ref = InputRef(path=str(p), role=role, present=False)
            self.manifest.inputs.append(ref)
            return ref
        ref = InputRef(
            path=str(p),
            role=role,
            sha256=sha256_file(p),
            bytes=p.stat().st_size,
            present=True,
        )
        self.manifest.inputs.append(ref)
        return ref
