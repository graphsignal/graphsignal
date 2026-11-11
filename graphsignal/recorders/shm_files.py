"""Reading the per-pid directories other processes publish under /dev/shm."""

import os
import stat
from typing import Optional


def _process_uid(pid) -> Optional[int]:
    try:
        return os.stat(f'/proc/{int(pid)}').st_uid
    except (OSError, TypeError, ValueError):
        return None


def is_owned_dir(path: str, pid=None) -> bool:
    """True when `path` is a real directory (not a symlink) that belongs to
    this user or to the user process `pid` runs as."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode):
        return False
    return st.st_uid in (os.geteuid(), _process_uid(pid))


def read_regular_file(path: str, max_bytes: int, offset: int = 0) -> Optional[bytes]:
    """Up to `max_bytes` of a regular file from `offset`. None when the path
    cannot be opened or is anything but a regular file (a symlink, a FIFO, a
    device)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        if offset:
            os.lseek(fd, offset, os.SEEK_SET)
        chunks = []
        remaining = max_bytes
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)
