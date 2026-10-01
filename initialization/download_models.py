"""Download and verify the two Specter DeepSeek checkpoint archives.

Archives are never extracted or installed. Public downloads use curl; private
releases require repository read access and an authenticated GitHub CLI.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ARCHIVES = {"target": "specter-ae-dsv2-target.tar.gz", "draft": "specter-ae-dsv2-draft.tar.gz"}
MAX_PART_BYTES = 1_900_000_000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("archives"), list):
        raise ValueError("Expected manifest schema_version=1 with an archives array.")
    seen = set()
    for archive in data["archives"]:
        if not isinstance(archive, dict) or not isinstance(archive.get("parts"), list):
            raise ValueError("Every archive needs an ordered parts array.")
        name = archive.get("name")
        if name not in ARCHIVES.values() or name in seen:
            raise ValueError(f"Unexpected or duplicate checkpoint archive: {name}")
        seen.add(name)
        items = [archive] + archive.get("parts", [])
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each archive part must be a metadata object.")
            if type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0:
                raise ValueError("Every archive and part needs a positive size_bytes.")
            if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError("Every archive and part needs a lowercase SHA-256 digest.")
        if not archive.get("parts") or sum(p["size_bytes"] for p in archive["parts"]) != archive["size_bytes"]:
            raise ValueError(f"Part sizes do not cover the full archive: {name}")
        for index, part in enumerate(archive["parts"], 1):
            if part.get("name") != f"{name}.part{index:04d}":
                raise ValueError("Parts must be numbered consecutively from part0001 in manifest order.")
            if part["size_bytes"] > MAX_PART_BYTES:
                raise ValueError("A part exceeds the 1,900,000,000-byte release split size.")
            if not isinstance(part.get("url"), str):
                raise ValueError("Each part needs its GitHub release download URL.")
            url = urlsplit(part["url"])
            if (url.scheme != "https" or url.netloc != "github.com" or url.query or url.fragment
                    or not re.fullmatch(r"/[^/]+/[^/]+/releases/download/[^/]+/[^/]+", url.path)):
                raise ValueError("Each part URL must be an HTTPS GitHub release download URL.")
            if unquote(url.path.rsplit("/", 1)[1]) != part["name"]:
                raise ValueError("The release asset name must match the manifest part name.")
    if not seen:
        raise ValueError("The manifest contains no checkpoint archives.")
    return data["archives"]


def child(directory, name):
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError(f"Refusing a symlink or path outside the output directory: {name}")
    return path


def verified(path, expected):
    return path.is_file() and path.stat().st_size == expected["size_bytes"] and sha256(path) == expected["sha256"]


@contextmanager
def output_lock(directory):
    lock = child(directory, ".specter-download.lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("Another download may be active. If no downloader is running, remove the stale "
                           ".specter-download.lock file and retry.") from error
    try:
        handle.write("Specter checkpoint download in progress.\n")
        handle.close()
        yield
    finally:
        handle.close()
        lock.unlink()


def fetch_part(part, directory, client, github_cli=False):
    destination = child(directory, part["name"])
    if destination.exists():
        if verified(destination, part):
            return destination
        raise ValueError(f"Existing part failed verification; retained unchanged: {destination.name}")
    partial = child(directory, part["name"] + ".partial")
    size = partial.stat().st_size if partial.exists() else 0
    if size > part["size_bytes"]:
        raise ValueError(f"Oversized partial retained; move it aside before retrying: {partial.name}")
    if size < part["size_bytes"]:
        print(f"Downloading {part['name']} ({size}/{part['size_bytes']} bytes present)", flush=True)
        if github_cli:
            fields = urlsplit(part["url"]).path.split("/")
            repo = unquote(fields[1]) + "/" + unquote(fields[2])
            tag = unquote(fields[5])
            if size:
                print("GitHub CLI restarts this incomplete part; verified parts are reused.", flush=True)
            result = subprocess.run(
                [client, "release", "download", tag, "--repo", repo, "--pattern", part["name"],
                 "--output", str(partial), "--clobber"],
                stdout=subprocess.PIPE, text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(f"GitHub CLI failed with exit {result.returncode} for {part['name']}. "
                                   "Confirm gh auth login and repository read access. HTTP 401/403/404 can "
                                   "indicate missing access or an unavailable release asset. Partial data "
                                   "was retained; rerun to restart this part.")
        else:
            result = subprocess.run(
                [client, "--location", "--fail", "--retry", "3", "--connect-timeout", "30",
                 "--proto", "=https", "--proto-redir", "=https", "--continue-at", "-",
                 "--progress-bar", "--write-out", "%{http_code}", "--output", str(partial), part["url"]],
                stdout=subprocess.PIPE, text=True, check=False,
            )
            status = result.stdout.strip()[-3:]
            if result.returncode:
                if status in {"401", "403", "404"}:
                    raise RuntimeError(f"HTTP {status} for {part['name']}: the release asset is not publicly accessible "
                                       "or the URL is incorrect. Private repositories require read access; "
                                       "use gh auth login and --github-cli. Partial data was retained.")
                raise RuntimeError(f"curl failed with exit {result.returncode} (HTTP {status or 'unknown'}). "
                                   "Partial data was retained; rerun to resume. If the server rejected byte ranges, "
                                   "move only the named .partial file aside to restart that part.")
    if not verified(partial, part):
        raise ValueError(f"Part size or SHA-256 mismatch; partial retained, not marked complete: {partial.name}. "
                         "Move it aside before retrying if its full expected length is already present.")
    partial.rename(destination)
    return destination


def assemble(archive, parts, directory, output):
    destination = child(output, archive["name"])
    assembling = child(directory, archive["name"] + ".assembling")
    # Only this downloader's unfinished assembly is restarted. Source parts remain intact.
    digest = hashlib.sha256()
    total = 0
    with assembling.open("wb") as merged:
        for part in parts:  # The manifest's explicit order is authoritative.
            with part.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    merged.write(block)
                    digest.update(block)
                    total += len(block)
    if total != archive["size_bytes"] or digest.hexdigest() != archive["sha256"]:
        raise ValueError("Reassembled archive failed verification; parts and unfinished assembly retained.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing archive: {destination.name}")
    assembling.rename(destination)
    print(f"Verified archive: {destination.name}  SHA-256 {archive['sha256']}", flush=True)


def download(archives, output, source_root, client, github_cli=False):
    output = output.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("Choose an output directory outside the source repository.")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        for archive in archives:
            destination = child(output, archive["name"])
            if destination.exists():
                if verified(destination, archive):
                    print(f"Already verified: {destination.name}", flush=True)
                    continue
                raise ValueError(f"Existing archive failed verification; retained unchanged: {destination.name}")
            cache = child(output, ".specter-parts")
            cache.mkdir(exist_ok=True)
            directory = child(cache, archive["sha256"])
            directory.mkdir(exist_ok=True)
            parts = [fetch_part(part, directory, client, github_cli) for part in archive["parts"]]
            assemble(archive, parts, directory, output)


def default_source_root():
    script = Path(__file__).resolve()
    for directory in script.parents:
        if (directory / "environment" / "requirements.lock.txt").is_file():
            return directory
    return script.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Local trusted release manifest JSON.")
    parser.add_argument("--output", required=True, type=Path, help="External directory for verified archives and parts.")
    parser.add_argument("--only", choices=ARCHIVES, help="Download just one checkpoint archive.")
    parser.add_argument("--github-cli", action="store_true", help="Use authenticated gh; incomplete parts restart.")
    parser.add_argument("--source-root", type=Path, default=default_source_root(), help="Repository path to protect.")
    args = parser.parse_args()
    try:
        archives = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        if args.only:
            archives = [archive for archive in archives if archive["name"] == ARCHIVES[args.only]]
            if not archives:
                raise ValueError("The requested checkpoint is absent from the manifest.")
        client_name = "gh" if args.github_cli else "curl"
        client = shutil.which(client_name)
        if client is None:
            raise ValueError(f"{client_name} is required for this download mode.")
        download(archives, args.output, args.source_root, client, args.github_cli)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
