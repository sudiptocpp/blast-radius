"""Read and verify changes against the pinned checkout without writing files."""

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Change:
    before_path: str | None
    after_path: str | None
    hunks: list[list[str]]

    @property
    def path(self) -> str:
        return self.before_path or self.after_path


def parse_patch(patch: str) -> list[Change]:
    changes = []
    for block in re.split(r"(?m)^diff --git ", patch)[1:]:
        lines = block.split("\n")
        if lines[-1] == "":
            lines.pop()
        starts = [i for i, line in enumerate(lines) if line.startswith("@@ ")]
        if not starts or any(line.startswith(("old mode ", "new mode ", "rename ", "copy "))
                             for line in lines[:starts[0]]):
            raise ValueError("unsupported file change")
        header = lines[:starts[0]]
        old = _path(next(line[4:] for line in header if line.startswith("--- ")), "a/")
        new = _path(next(line[4:] for line in header if line.startswith("+++ ")), "b/")
        if (old and new and old != new) or not (old or new):
            raise ValueError("unsupported file paths")
        changes.append(Change(old, new, [lines[a:b] for a, b in zip(starts, starts[1:] + [len(lines)])]))
    if not changes:
        raise ValueError("missing file changes")
    return changes


def _path(raw: str, prefix: str) -> str | None:
    raw = raw.split("\t", 1)[0]
    if raw == "/dev/null":
        return None
    if not raw.startswith(prefix):
        raise ValueError("unsupported diff path")
    name = raw[len(prefix):]
    if not name or "\\" in name or ":" in name or PurePosixPath(name).is_absolute() or ".." in name.split("/"):
        raise ValueError("invalid diff path")
    return name


def apply_python(change: Change, checkout: Path) -> tuple[str, str, set[int], dict[int, int]]:
    """Return old/new source, removed old lines and added lines with their old anchors."""
    if change.before_path is None:
        before = ""
        original: list[str] = []
    else:
        before = (checkout / change.before_path).read_text(encoding="utf-8")
        *lines, last = before.split("\n")
        original = [line + "\n" for line in lines]
        if last:
            original.append(last)

    edited = []
    removed = set()
    added = {}
    cursor = 0
    for hunk in change.hunks:
        match = HUNK.match(hunk[0])
        if not match:
            raise ValueError("invalid hunk header")
        old_at, old_count, new_at, new_count = (int(n) if n is not None else 1 for n in match.groups())
        start = old_at - bool(old_count)
        if not cursor <= start <= len(original):
            raise ValueError("invalid hunk location")
        edited.extend(original[cursor:start])
        cursor = start
        output_start = len(edited)
        if output_start != new_at - bool(new_count):
            raise ValueError("invalid target hunk location")
        last_marker = None
        for line in hunk[1:]:
            if line == r"\ No newline at end of file":
                if last_marker == "+":
                    edited[-1] = edited[-1].removesuffix("\n")
                continue
            marker, content = line[:1], line[1:]
            if marker in (" ", "-"):
                if cursor >= len(original) or original[cursor].removesuffix("\n") != content:
                    raise ValueError("hunk does not match source")
                if marker == " ":
                    edited.append(original[cursor])
                cursor += 1
            if marker == "-":
                removed.add(cursor)
            if marker == "+":
                edited.append(content + "\n")
                added[len(edited)] = cursor
            if marker not in (" ", "-", "+"):
                raise ValueError("invalid hunk body")
            last_marker = marker
        if cursor - start != old_count or len(edited) - output_start != new_count:
            raise ValueError("hunk line count mismatch")
    edited.extend(original[cursor:])
    return before, "".join(edited), removed, added


apply_text = apply_python
