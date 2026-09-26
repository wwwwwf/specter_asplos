"""Create the pinned Specter environment using external custom wheels."""

import argparse
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


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Environment paths must be non-empty strings.")
    path = Path(os.path.expandvars(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


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
            [str(interpreter), "-I", "-c", "import platform; print(platform.python_version())"],
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
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    create_command = [uv, "venv", str(venv), "--python", PYTHON_VERSION, "--prompt", "specter"]
    sync_command = [uv, "pip", "sync", "--python", str(interpreter), "--link-mode", "copy"]
    check_command = [uv, "pip", "check", "--python", str(interpreter)]
    print(f"Environment: {venv}")
    print(f"Custom wheels: {wheelhouse} (4 validated)")
    print("Dependencies: 194 pinned packages; the remaining 190 require an online package index.")
    if not venv.exists():
        print(shlex.join(create_command))
    print(shlex.join(sync_command + ["<resolved requirements>"]))
    print(shlex.join(check_command))
    if args.check:
        print("Check complete; no environment changes were made.")
        return

    process_env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    with tempfile.TemporaryDirectory(prefix="specter-install-") as temporary:
        resolved_file = Path(temporary) / "requirements.txt"
        resolved_file.write_text(requirements, encoding="utf-8")
        if not venv.exists():
            subprocess.run(create_command, check=True, env=process_env)
        subprocess.run(sync_command + [str(resolved_file)], check=True, env=process_env)
        subprocess.run(check_command, check=True, env=process_env)
    print("Specter environment is ready.")


if __name__ == "__main__":
    main()
