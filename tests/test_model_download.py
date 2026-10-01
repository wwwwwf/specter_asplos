"""Small CPU fixtures; no GitHub requests or model downloads."""
import copy
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from initialization import download_models as downloader


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "downloads"
        self.source = self.root / "source"
        self.contents = [b"first part 12345", b"second part 6789"]
        name = downloader.ARCHIVES["target"]
        self.archive = {
            "name": name, "size_bytes": sum(map(len, self.contents)),
            "sha256": hashlib.sha256(b"".join(self.contents)).hexdigest(), "parts": [
                {"name": f"{name}.part{index:04d}", "size_bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest(),
                 "url": f"https://github.com/example/specter/releases/download/ae-v2/{name}.part{index:04d}"}
                for index, data in enumerate(self.contents, 1)
            ],
        }
        self.manifest = {"schema_version": 1, "archives": [self.archive]}

    def fake_curl(self, command, **kwargs):
        self.assertIn("--continue-at", command)
        self.assertFalse(kwargs["check"])
        path = Path(command[command.index("--output") + 1])
        index = int(command[-1][-4:]) - 1
        offset = path.stat().st_size if path.exists() else 0
        with path.open("ab") as handle:
            handle.write(self.contents[index][offset:])
        return subprocess.CompletedProcess(command, 0, "206" if offset else "200")

    def fake_gh(self, command, **kwargs):
        self.assertEqual(command[:4], ["gh", "release", "download", "ae-v2"])
        self.assertEqual(command[command.index("--repo") + 1], "example/specter")
        self.assertIn("--clobber", command)
        self.assertFalse(kwargs["check"])
        asset = command[command.index("--pattern") + 1]
        index = int(asset[-4:]) - 1
        self.assertEqual(asset, self.archive["parts"][index]["name"])
        path = Path(command[command.index("--output") + 1])
        self.assertEqual(path.name, asset + ".partial")
        path.write_bytes(self.contents[index])
        return subprocess.CompletedProcess(command, 0, "")

    def run_download(self, side_effect=None, github_cli=False):
        fake = side_effect or (self.fake_gh if github_cli else self.fake_curl)
        with redirect_stdout(io.StringIO()), patch.object(downloader.subprocess, "run", side_effect=fake) as run:
            downloader.download(downloader.validate_manifest(self.manifest), self.output, self.source,
                                "gh" if github_cli else "curl", github_cli)
            return run

    def test_resume_parts_and_reassemble_then_skip_verified_archive(self):
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        directory.mkdir(parents=True)
        partial = directory / (self.archive["parts"][0]["name"] + ".partial")
        partial.write_bytes(self.contents[0][:4])
        self.assertEqual(self.run_download().call_count, 2)
        self.assertEqual((self.output / self.archive["name"]).read_bytes(), b"".join(self.contents))
        self.assertFalse(partial.exists())
        self.assertFalse((self.output / ".specter-download.lock").exists())
        self.run_download().assert_not_called()

    def test_404_explains_private_access_without_marking_complete(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP 404.*Private repositories require read access"):
            self.run_download(lambda command, **kwargs: subprocess.CompletedProcess(command, 22, "404"))
        self.assertFalse((self.output / self.archive["name"]).exists())
        self.assertFalse((self.output / ".specter-download.lock").exists())

    def test_interrupted_transfer_resumes_from_retained_partial(self):
        def interrupted(command, **kwargs):
            path = Path(command[command.index("--output") + 1])
            path.write_bytes(self.contents[0][:5])
            return subprocess.CompletedProcess(command, 18, "200")
        with self.assertRaisesRegex(RuntimeError, "rerun to resume"):
            self.run_download(interrupted)
        self.assertFalse((self.output / self.archive["name"]).exists())
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        partial = directory / (self.archive["parts"][0]["name"] + ".partial")
        self.assertEqual(partial.read_bytes(), self.contents[0][:5])
        self.run_download()
        self.assertEqual((self.output / self.archive["name"]).read_bytes(), b"".join(self.contents))

    def test_complete_partial_is_verified_without_downloading_it_again(self):
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        directory.mkdir(parents=True)
        partial = directory / (self.archive["parts"][0]["name"] + ".partial")
        partial.write_bytes(self.contents[0])
        self.assertEqual(self.run_download().call_count, 1)
        self.assertTrue((directory / self.archive["parts"][0]["name"]).exists())

    def test_bad_part_hash_is_retained_as_partial(self):
        def corrupt(command, **kwargs):
            result = self.fake_curl(command, **kwargs)
            path = Path(command[command.index("--output") + 1])
            path.write_bytes(b"x" * path.stat().st_size)
            return result
        with self.assertRaisesRegex(ValueError, "partial retained, not marked complete"):
            self.run_download(corrupt)
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        self.assertTrue((directory / (self.archive["parts"][0]["name"] + ".partial")).exists())
        self.assertFalse((directory / self.archive["parts"][0]["name"]).exists())
        self.assertFalse((self.output / self.archive["name"]).exists())

    def test_bad_full_hash_never_creates_completed_archive(self):
        self.archive["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Reassembled archive failed"):
            self.run_download()
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        self.assertTrue((directory / (self.archive["name"] + ".assembling")).exists())
        self.assertTrue(all((directory / part["name"]).exists() for part in self.archive["parts"]))
        self.assertFalse((self.output / self.archive["name"]).exists())

    def test_existing_archive_is_not_overwritten(self):
        self.output.mkdir()
        path = self.output / self.archive["name"]
        path.write_bytes(b"keep original asset")
        with self.assertRaisesRegex(ValueError, "retained unchanged"):
            self.run_download()
        self.assertEqual(path.read_bytes(), b"keep original asset")

    def test_source_output_and_traversal_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the source"):
            downloader.download([self.archive], self.source / "downloads", self.source, "curl")
        self.assertFalse(self.source.exists())
        manifest = copy.deepcopy(self.manifest)
        manifest["archives"][0]["parts"][0]["name"] = "../source.txt"
        with self.assertRaisesRegex(ValueError, "numbered consecutively"):
            downloader.validate_manifest(manifest)

    def test_non_github_url_and_oversized_release_part_are_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["archives"][0]["parts"][0]["url"] = "https://other.example/file"
        with self.assertRaisesRegex(ValueError, "HTTPS GitHub"):
            downloader.validate_manifest(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["archives"][0]["parts"][0]["size_bytes"] = 2**31
        manifest["archives"][0]["size_bytes"] = sum(p["size_bytes"] for p in manifest["archives"][0]["parts"])
        with self.assertRaisesRegex(ValueError, "split size"):
            downloader.validate_manifest(manifest)

    def test_gh_uses_release_repo_tag_and_restarts_only_incomplete_part(self):
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        directory.mkdir(parents=True)
        first = directory / self.archive["parts"][0]["name"]
        first.write_bytes(self.contents[0])
        partial = directory / (self.archive["parts"][1]["name"] + ".partial")
        partial.write_bytes(self.contents[1][:4])
        run = self.run_download(github_cli=True)
        self.assertEqual(run.call_count, 1)
        self.assertEqual((self.output / self.archive["name"]).read_bytes(), b"".join(self.contents))
        self.assertEqual(first.read_bytes(), self.contents[0])
        self.assertFalse(partial.exists())

    def test_gh_failure_retains_partial_and_retry_downloads_it_again(self):
        def failed(command, **kwargs):
            path = Path(command[command.index("--output") + 1])
            path.write_bytes(self.contents[0][:5])
            return subprocess.CompletedProcess(command, 1, "")
        with self.assertRaisesRegex(RuntimeError, "repository read access.*Partial data was retained"):
            self.run_download(failed, github_cli=True)
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        self.assertEqual((directory / (self.archive["parts"][0]["name"] + ".partial")).read_bytes(),
                         self.contents[0][:5])
        self.assertFalse((directory / self.archive["parts"][0]["name"]).exists())
        self.run_download(github_cli=True)
        self.assertEqual((self.output / self.archive["name"]).read_bytes(), b"".join(self.contents))

    def test_gh_checks_existing_part_hash_instead_of_trusting_filename(self):
        directory = self.output / ".specter-parts" / self.archive["sha256"]
        directory.mkdir(parents=True)
        first = directory / self.archive["parts"][0]["name"]
        first.write_bytes(b"x" * len(self.contents[0]))
        with patch.object(downloader.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Existing part failed verification"):
                downloader.download([self.archive], self.output, self.source, "gh", True)
            run.assert_not_called()
        self.assertEqual(first.read_bytes(), b"x" * len(self.contents[0]))

    def test_default_source_root_is_repository_when_script_is_in_initialization(self):
        (self.source / "environment").mkdir(parents=True)
        (self.source / "environment" / "requirements.lock.txt").write_text("# fixture\n")
        script = self.source / "initialization" / "download_models.py"
        with patch.object(downloader, "__file__", str(script)):
            self.assertEqual(downloader.default_source_root(), self.source)


if __name__ == "__main__":
    unittest.main()
