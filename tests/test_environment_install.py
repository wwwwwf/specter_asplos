"""Exercise installer isolation and integrity checks without installing packages."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from environment import install


class EnvironmentInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="specter-install-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.wheels = self.root / "wheels"
        self.wheels.mkdir()
        hashes = []
        for name in install.CUSTOM_WHEELS:
            wheel = self.wheels / name
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("test.dist-info/METADATA", name)
            hashes.append(f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {name}")
        self.manifest = self.root / "wheels.sha256"
        self.manifest.write_text("\n".join(hashes) + "\n", encoding="utf-8")
        self.config = self.root / "specter.toml"
        self.write_config()
        self.venv = self.root / "env"
        self.cache = self.root / "cache"
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        manifest_patch = patch.object(install, "WHEEL_HASHES", self.manifest)
        manifest_patch.start()
        self.addCleanup(manifest_patch.stop)

    def write_config(self, cache="./cache"):
        self.config.write_text(
            f'[paths]\ncache = "{cache}"\n'
            '[environment]\npython = "3.11.13"\n'
            'venv = "./env"\nwheelhouse = "./wheels"\n',
            encoding="utf-8",
        )

    def call_main(self, check=False, runner=None, free=64 * 1024**3, version="3.11.13"):
        argv = ["install.py", "--config", str(self.config)] + (["--check"] if check else [])
        with ExitStack() as stack:
            stack.enter_context(patch.object(install.sys, "argv", argv))
            stack.enter_context(patch.object(install.sys, "platform", "linux"))
            stack.enter_context(patch.object(install.platform, "machine", return_value="x86_64"))
            stack.enter_context(patch.object(install.shutil, "which", return_value="/tools/uv"))
            stack.enter_context(patch.object(install.shutil, "disk_usage", return_value=SimpleNamespace(free=free)))
            run = stack.enter_context(patch.object(install.subprocess, "run", side_effect=runner))
            check_output = stack.enter_context(patch.object(install.subprocess, "check_output", return_value=version + "\n"))
            stack.enter_context(redirect_stdout(self.stdout))
            stack.enter_context(redirect_stderr(self.stderr))
            install.main()
            return run, check_output

    def test_install_passes_constraints_and_isolated_storage_to_every_process(self):
        observed = []

        def run(command, **kwargs):
            self.assertTrue(kwargs["check"])
            env = kwargs["env"]
            self.assertEqual(env["UV_CACHE_DIR"], str(self.cache / "uv"))
            self.assertEqual(env["TMPDIR"], str(self.cache / "tmp"))
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(self.cache / "python"))
            for key in ("PYTHONPATH", "PYTHONHOME", "UV_BUILD_CONSTRAINT"):
                self.assertNotIn(key, env)
            if command[1:3] == ["pip", "sync"]:
                position = command.index("--build-constraints")
                self.assertEqual(command[position + 1], str(install.BUILD_CONSTRAINTS.resolve()))
                requirements = Path(command[-1])
                self.assertEqual(requirements.parent.parent, self.cache / "tmp")
                lines = requirements.read_text(encoding="utf-8").splitlines()
                original = install.REQUIREMENTS.read_text(encoding="utf-8").splitlines()
                expected = [
                    (self.wheels / line.removeprefix("./wheelhouse/")).as_uri()
                    if line.startswith("./wheelhouse/") else line
                    for line in original
                ]
                self.assertEqual(lines, expected)
                self.assertEqual(sum(bool(line.strip()) and not line.startswith("#") for line in lines), 194)
            observed.append(command)

        inherited = {"UV_CACHE_DIR": "/old/cache", "TMPDIR": "/old/tmp",
                     "UV_PYTHON_INSTALL_DIR": "/old/python", "UV_BUILD_CONSTRAINT": "/old/constraints",
                     "PYTHONPATH": "/old/modules", "PYTHONHOME": "/old/home"}
        with patch.dict(os.environ, inherited):
            self.call_main(runner=run)
        self.assertEqual([command[1:3] for command in observed],
                         [["venv", str(self.venv)], ["pip", "sync"], ["pip", "check"]])
        self.assertEqual(list((self.cache / "tmp").iterdir()), [])

    def test_check_does_not_create_directories_or_install(self):
        before = {path.relative_to(self.root): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        with patch.object(Path, "mkdir", side_effect=AssertionError("--check must not mkdir")):
            run, version = self.call_main(check=True)
        run.assert_not_called()
        version.assert_not_called()
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.venv.exists())
        after = {path.relative_to(self.root): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        for expected in (str(self.venv), str(self.cache / "tmp"), str(self.cache / "python"),
                         "64.0 GiB", "4 SHA-256 checks passed", "--build-constraints"):
            self.assertIn(expected, self.stdout.getvalue())

    def test_valid_zip_with_wrong_content_is_rejected(self):
        with zipfile.ZipFile(self.wheels / install.CUSTOM_WHEELS[0], "a") as archive:
            archive.writestr("unexpected-change.txt", "changed build")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            install.load_plan(self.config)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.venv.exists())

    def test_missing_wheel_fails_before_install(self):
        (self.wheels / install.CUSTOM_WHEELS[0]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing or invalid custom wheel"):
            install.load_plan(self.config)

    def test_existing_environment_is_reused_only_at_configured_path(self):
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin" / "python").write_bytes(b"mock interpreter")
        (self.venv / "pyvenv.cfg").write_text("version = 3.11.13\n", encoding="utf-8")
        with patch.dict(os.environ, {"VIRTUAL_ENV": str(self.root / "old-env")}):
            run, version = self.call_main()
        version.assert_called_once()
        self.assertEqual(version.call_args.args[0][:3], [str(self.venv / "bin" / "python"), "-I", "-B"])
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--python") + 1], str(self.venv / "bin" / "python"))
            self.assertNotIn("venv", command)

    def test_insufficient_space_fails_without_writes(self):
        with patch.object(Path, "mkdir", side_effect=AssertionError("preflight must not mkdir")):
            with self.assertRaises(SystemExit) as raised:
                self.call_main(check=True, free=2 * 1024**3)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("2.0 GiB", self.stdout.getvalue())
        self.assertIn("not a measured peak requirement", self.stderr.getvalue())

    def test_cache_cannot_overlap_the_selected_environment(self):
        self.write_config(cache="./env/cache")
        with self.assertRaisesRegex(ValueError, "Keep installation caches separate"):
            install.load_cache_plan(self.config, self.venv, self.wheels)

    def test_build_constraints_match_runtime_lock(self):
        def pins(path):
            return dict(line.strip().split("==", 1) for line in path.read_text().splitlines()
                        if line.strip() and not line.startswith("#") and "==" in line)
        constraints = pins(install.BUILD_CONSTRAINTS)
        self.assertEqual(set(constraints), {"torch", "setuptools", "wheel", "packaging", "ninja", "numpy"})
        runtime = pins(install.REQUIREMENTS)
        self.assertEqual(constraints, {name: runtime[name] for name in constraints})


if __name__ == "__main__":
    unittest.main()
