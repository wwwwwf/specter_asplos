"""Create the pinned Specter environment using external custom wheels."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile


CUSTOM_WHEELS = (
    "flash_attn-2.8.1-cp311-cp311-linux_x86_64.whl",
    "gptqmodel-4.0.0.dev0-cp311-cp311-linux_x86_64.whl",
    "transformers-4.53.3-py3-none-any.whl",
    "vllm-0.9.2-cp38-abi3-linux_x86_64.whl",
)
PYTHON_VERSION = "3.11.13"
REQUIREMENTS = Path(__file__).with_name("requirements.lock.txt")
BUILD_CONSTRAINTS = Path(__file__).with_name("build-constraints.txt")
WHEEL_HASHES = Path(__file__).with_name("wheels.sha256")
MIN_FREE_BYTES = 5 * 1024**3


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Environment paths must be non-empty strings.")
    path = Path(os.path.expandvars(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def wheel_hashes():
    expected = {}
    for line in WHEEL_HASHES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if (name in expected or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"Invalid wheel checksum entry: {line}")
        expected[name] = digest
    if set(expected) != set(CUSTOM_WHEELS):
        raise ValueError("The checksum manifest must contain exactly the four custom wheels.")
    return expected


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlaps(first, second):
    return first == second or first in second.parents or second in first.parents


def load_cache_plan(config_file, venv, wheelhouse):
    config_file = config_file.expanduser().resolve()
    with config_file.open("rb") as handle:
        cache = resolve_path(tomllib.load(handle)["paths"]["cache"], config_file.parent)
    source_root = REQUIREMENTS.parent.parent.resolve()
    paths = {name: (cache / name).resolve() for name in ("uv", "tmp", "python")}
    for path in (cache, *paths.values()):
        for protected in (source_root, venv, wheelhouse):
            if overlaps(path, protected):
                raise ValueError(f"Keep installation caches separate from source, environment, and wheels: {path}")
    for name, path in paths.items():
        if any(overlaps(path, other) for key, other in paths.items() if key != name):
            raise ValueError("The uv cache, temporary directory, and Python installation directory must be separate.")
    return paths


def storage_available(path):
    parent = path
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(f"Expected a directory for installation storage: {parent}")
    return parent, shutil.disk_usage(parent).free


def load_plan(config_file):
    config_file = config_file.expanduser().resolve()
    with config_file.open("rb") as handle:
        settings = tomllib.load(handle)["environment"]
    python_version = settings["python"]
    if python_version != PYTHON_VERSION:
        raise ValueError(f"The pinned environment requires Python {PYTHON_VERSION}.")
    venv = resolve_path(settings["venv"], config_file.parent)
    wheelhouse = resolve_path(settings["wheelhouse"], config_file.parent)
    source_root = REQUIREMENTS.parent.parent.resolve()
    if venv == source_root or source_root in venv.parents or venv in source_root.parents:
        raise ValueError("Keep the virtual environment outside the source directory.")
    if wheelhouse == source_root or source_root in wheelhouse.parents:
        raise ValueError("Keep the wheelhouse outside the source directory.")
    if venv == wheelhouse or venv in wheelhouse.parents or wheelhouse in venv.parents:
        raise ValueError("The virtual environment and wheelhouse must be separate.")

    expected_hashes = wheel_hashes()
    resolved_lines = []
    wheel_names = []
    entries = 0
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries += 1
        if stripped.startswith("./wheelhouse/"):
            name = stripped.removeprefix("./wheelhouse/")
            if name not in CUSTOM_WHEELS:
                raise ValueError(f"Unexpected custom wheel in requirements: {name}")
            wheel = wheelhouse / name
            if not wheel.is_file() or not zipfile.is_zipfile(wheel):
                raise ValueError(f"Missing or invalid custom wheel: {wheel}")
            actual_hash = sha256_file(wheel)
            if actual_hash != expected_hashes[name]:
                raise ValueError(f"SHA-256 mismatch for {wheel}: expected {expected_hashes[name]}, got {actual_hash}")
            wheel_names.append(name)
            line = wheel.as_uri()
        resolved_lines.append(line)
    if sorted(wheel_names) != sorted(CUSTOM_WHEELS) or entries != 194:
        raise ValueError("Expected 194 pinned packages, including four custom wheels.")

    interpreter = venv / "bin" / "python"
    if venv.exists():
        if not (venv / "pyvenv.cfg").is_file() or not interpreter.is_file():
            raise ValueError(f"Existing target is not a Python virtual environment: {venv}")
        version = subprocess.check_output(
            [str(interpreter), "-I", "-B", "-c", "import platform; print(platform.python_version())"],
            text=True,
        ).strip()
        if version != PYTHON_VERSION:
            raise ValueError(f"Existing environment uses Python {version}; expected {PYTHON_VERSION}.")
    return venv, interpreter, wheelhouse, "\n".join(resolved_lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to specter.toml.")
    parser.add_argument("--check", action="store_true", help="Validate inputs and show the plan without installing.")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("The custom wheels require Linux x86_64.")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv and make it available on PATH.")
    try:
        venv, interpreter, wheelhouse, requirements = load_plan(args.config)
        caches = load_cache_plan(args.config, venv, wheelhouse)
        if not BUILD_CONSTRAINTS.is_file():
            raise ValueError(f"Missing build constraints: {BUILD_CONSTRAINTS}")
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    create_command = [uv, "venv", str(venv), "--python", PYTHON_VERSION, "--prompt", "specter"]
    sync_command = [uv, "pip", "sync", "--python", str(interpreter), "--link-mode", "copy",
                    "--build-constraints", str(BUILD_CONSTRAINTS.resolve())]
    check_command = [uv, "pip", "check", "--python", str(interpreter)]
    print(f"Environment: {venv}", flush=True)
    print(f"Custom wheels: {wheelhouse} (4 SHA-256 checks passed)", flush=True)
    print(f"uv cache: {caches['uv']}", flush=True)
    print(f"Temporary files: {caches['tmp']}", flush=True)
    print(f"Managed Python: {caches['python']}", flush=True)
    print(f"Build constraints: {BUILD_CONSTRAINTS.resolve()}", flush=True)
    print("Dependencies: 194 pinned packages; the remaining 190 require an online package index.", flush=True)
    try:
        low_storage = []
        for label, path in (("Environment", venv), *caches.items()):
            parent, free = storage_available(path)
            print(f"Available storage for {label}: {free / 1024**3:.1f} GiB (at {parent})", flush=True)
            if free < MIN_FREE_BYTES:
                low_storage.append(str(path))
        if low_storage:
            raise ValueError("Less than 5 GiB free for: " + ", ".join(low_storage)
                             + ". Choose a filesystem with more free space. "
                             "This is a conservative preflight guard, not a measured peak requirement.")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not venv.exists():
        print(shlex.join(create_command), flush=True)
    print(shlex.join(sync_command + ["<resolved requirements>"]), flush=True)
    print(shlex.join(check_command), flush=True)
    if args.check:
        print("Check complete; no environment changes were made.", flush=True)
        return

    process_env = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPATH", "PYTHONHOME", "UV_BUILD_CONSTRAINT")}
    process_env.update({
        "UV_CACHE_DIR": str(caches["uv"]),
        "TMPDIR": str(caches["tmp"]),
        "UV_PYTHON_INSTALL_DIR": str(caches["python"]),
    })
    for path in caches.values():
        path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="specter-install-", dir=caches["tmp"]) as temporary:
        resolved_file = Path(temporary) / "requirements.txt"
        resolved_file.write_text(requirements, encoding="utf-8")
        if not venv.exists():
            subprocess.run(create_command, check=True, env=process_env)
        subprocess.run(sync_command + [str(resolved_file)], check=True, env=process_env)
        subprocess.run(check_command, check=True, env=process_env)
    print("Specter environment is ready.", flush=True)


if __name__ == "__main__":
    main()
