"""Convert a `docker save` archive into an enroot squashfs image (.sqsh) on a machine without Docker.

usage: python scripts/kesai/docker_archive_to_sqsh.py alpasim-base-0.134.0.tar.zst /cephfs/data/nurec/sqsh/alpasim_base_0.134.0.sqsh \
           --work_dir /cephfs/scratch/$USER/sqsh_build [--keep_work] [--processors 32]

Flattens the image layers (OCI whiteouts applied), writes enroot's /etc/environment, /etc/fstab and /etc/rc from the
image config (ENV, VOLUME, WORKDIR, ENTRYPOINT, CMD), the way `enroot import` does, and packs the rootfs with mksquashfs.
work_dir needs about twice the unpacked image size. The sqsh name must be what the AlpaSim wizard looks for:
image basename with ':' and '-' replaced by '_', e.g. alpasim-base:0.134.0 -> alpasim_base_0.134.0.sqsh.
Needs tar, zstd (for .zst archives) and mksquashfs (squashfs-tools).
"""

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

COPY_BUFFER_BYTES = 1 << 20
MAX_SYMLINK_HOPS = 40
OPAQUE_WHITEOUT = ".wh..wh..opq"
WHITEOUT_PREFIX = ".wh."
SQUASH_OPTIONS = ["-noappend", "-comp", "lz4", "-Xhc", "-b", "262144", "-no-xattrs", "-force-uid", "0", "-force-gid", "0"]
EXTRACTABLE_TYPES = (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE)


def extract_archive(archive, archive_dir):
    archive_dir.mkdir(parents=True)
    if archive.suffix == ".zst":
        decompress = subprocess.Popen(["zstd", "-dc", "-T0", str(archive)], stdout=subprocess.PIPE)
        untar = subprocess.run(["tar", "-x", "-C", str(archive_dir)], stdin=decompress.stdout)
        decompress.stdout.close()
        if decompress.wait() != 0 or untar.returncode != 0:
            raise SystemExit(f"extracting {archive} failed (zstd {decompress.returncode}, tar {untar.returncode})")
        return
    subprocess.run(["tar", "-xf", str(archive), "-C", str(archive_dir)], check=True)


def rootfs_path(rootfs, relative_name):
    """Path of an image entry inside rootfs, following parent symlinks within the image like a chroot would."""
    parts = [part for part in relative_name.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise SystemExit(f"refusing image entry with '..': {relative_name}")
    if not parts:
        return rootfs
    current = rootfs
    pending = parts[:-1]
    hops = 0
    while pending:
        candidate = current / pending.pop(0)
        if not candidate.is_symlink():
            current = candidate
            continue
        hops += 1
        if hops > MAX_SYMLINK_HOPS:
            raise SystemExit(f"symlink loop while resolving {relative_name}")
        target = os.readlink(candidate)
        target_parts = [part for part in target.split("/") if part not in ("", ".")]
        if target.startswith("/"):
            current = rootfs
        if ".." in target_parts:
            raise SystemExit(f"refusing symlink with '..' inside the image: {candidate} -> {target}")
        pending = target_parts + pending
    return current / parts[-1]


def remove_existing(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.is_symlink() or path.exists():
        path.unlink()


def apply_whiteouts(layer_members, rootfs):
    removed_count = 0
    for member in layer_members:
        basename = os.path.basename(member.name.rstrip("/"))
        if not basename.startswith(WHITEOUT_PREFIX):
            continue
        parent = rootfs_path(rootfs, member.name).parent
        if basename == OPAQUE_WHITEOUT:
            targets = list(parent.iterdir()) if parent.is_dir() else []
        else:
            targets = [parent / basename[len(WHITEOUT_PREFIX):]]
        for target in targets:
            if target.is_symlink() or target.exists():
                remove_existing(target)
                removed_count += 1
    return removed_count


def extract_layer(layer_tar, rootfs, dir_modes):
    """Extract one layer over rootfs. -> (entries written, entries skipped: devices and fifos)."""
    written_count = skipped_count = 0
    for member in layer_tar.getmembers():
        basename = os.path.basename(member.name.rstrip("/"))
        if basename.startswith(WHITEOUT_PREFIX) or member.name.strip("/") in ("", "."):
            continue
        if member.type not in EXTRACTABLE_TYPES:
            skipped_count += 1
            continue
        destination = rootfs_path(rootfs, member.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if member.isdir():
            if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                destination.unlink()
            destination.mkdir(exist_ok=True)
            destination.chmod((member.mode & 0o7777) | stat.S_IRWXU)  # owner access while later layers write into it
            dir_modes[destination] = member.mode & 0o7777
        elif member.issym():
            remove_existing(destination)
            os.symlink(member.linkname, destination)
        elif member.islnk():
            remove_existing(destination)
            os.link(rootfs_path(rootfs, member.linkname), destination)
        else:
            remove_existing(destination)
            with layer_tar.extractfile(member) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target, COPY_BUFFER_BYTES)
            destination.chmod(member.mode & 0o7777)
        written_count += 1
    return written_count, skipped_count


def write_enroot_config(rootfs, image_config):
    """/etc/environment, /etc/fstab and /etc/rc as `enroot import` generates them from the image config."""
    etc = rootfs / "etc"
    etc.mkdir(exist_ok=True)
    (etc / "environment").write_text("".join(line + "\n" for line in image_config.get("Env") or []))
    volumes = image_config.get("Volumes") or {}
    (etc / "fstab").write_text(
        "".join(f"tmpfs {volume} tmpfs x-create=dir,rw,nosuid,nodev\n" for volume in sorted(volumes))
    )
    workdir = image_config.get("WorkingDir") or "/"
    rootfs_path(rootfs, workdir).mkdir(parents=True, exist_ok=True)
    entrypoint = " ".join(shlex.quote(part) for part in image_config.get("Entrypoint") or [])
    cmd = " ".join(shlex.quote(part) for part in image_config.get("Cmd") or [])
    labels = image_config.get("Labels") or {}
    rc_lines = [f"# {key} {value}" for key, value in sorted(labels.items())] + [
        f"cd {shlex.quote(workdir)} && unset OLDPWD || exit 1",
        "",
        "if [ -s /etc/rc.local ]; then",
        "    . /etc/rc.local",
        "fi",
        "",
        "if [ $# -gt 0 ]; then",
        f'    exec {entrypoint} "$@"'.replace("exec  ", "exec "),
        "else",
        f"    exec {entrypoint} {cmd}".replace("exec  ", "exec "),
        "fi",
    ]
    (etc / "rc").write_text("".join(line + "\n" for line in rc_lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path, help="docker save archive: .tar, .tar.gz or .tar.zst")
    ap.add_argument("output", type=Path, help="squashfs image to write")
    ap.add_argument("--work_dir", type=Path, default=None, help="scratch space, default <output>.work")
    ap.add_argument("--keep_work", action="store_true", help="leave the unpacked archive and rootfs in work_dir")
    ap.add_argument("--processors", type=int, default=os.cpu_count(), help="mksquashfs threads")
    args = ap.parse_args()
    for tool in ("tar", "mksquashfs") + (("zstd",) if args.archive.suffix == ".zst" else ()):
        if shutil.which(tool) is None:
            raise SystemExit(f"missing tool: {tool}")
    if not args.archive.is_file():
        raise SystemExit(f"no archive at {args.archive}")
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    work_dir = args.work_dir or args.output.with_name(args.output.name + ".work")
    if work_dir.exists():
        raise SystemExit(f"work dir {work_dir} already exists; remove it or pass another --work_dir")
    archive_dir, rootfs = work_dir / "archive", work_dir / "rootfs"

    print(f"[docker_archive_to_sqsh] extracting {args.archive} -> {archive_dir}", flush=True)
    extract_archive(args.archive, archive_dir)
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    if len(manifest) != 1:
        raise SystemExit(f"expected one image in the archive, got {[entry.get('RepoTags') for entry in manifest]}")
    image_config = json.loads((archive_dir / manifest[0]["Config"]).read_text())["config"]
    layers = [archive_dir / layer for layer in manifest[0]["Layers"]]
    print(f"[docker_archive_to_sqsh] image {manifest[0].get('RepoTags')}: {len(layers)} layers", flush=True)

    rootfs.mkdir()
    dir_modes = {}
    for layer_idx, layer in enumerate(layers):
        with tarfile.open(layer, "r:*") as layer_tar:
            removed_count = apply_whiteouts(layer_tar.getmembers(), rootfs)
            written_count, skipped_count = extract_layer(layer_tar, rootfs, dir_modes)
        if layer not in layers[layer_idx + 1:]:  # identical layers share one blob
            layer.unlink()
        print(
            f"[docker_archive_to_sqsh] layer {layer_idx + 1}/{len(layers)}: {written_count} entries, "
            f"{removed_count} whiteout removals, {skipped_count} device/fifo entries skipped",
            flush=True,
        )
    write_enroot_config(rootfs, image_config)
    for directory in sorted(dir_modes, key=lambda path: len(path.parts), reverse=True):
        if directory.is_dir() and not directory.is_symlink():  # a later layer may have replaced or removed it
            directory.chmod(dir_modes[directory] | stat.S_IRUSR | stat.S_IXUSR)  # mksquashfs must be able to read it

    partial_output = args.output.with_name(args.output.name + ".partial")
    partial_output.unlink(missing_ok=True)
    print(f"[docker_archive_to_sqsh] mksquashfs {rootfs} -> {args.output}", flush=True)
    subprocess.run(
        ["mksquashfs", str(rootfs), str(partial_output), "-processors", str(args.processors), *SQUASH_OPTIONS],
        check=True,
    )
    partial_output.replace(args.output)
    if not args.keep_work:
        shutil.rmtree(work_dir)
    print(f"[docker_archive_to_sqsh] done: {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
