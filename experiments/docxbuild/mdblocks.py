r"""Block- and inline-level parser for the subset of Markdown used by docs/MANUSCRIPT.md."""
from __future__ import annotations

import re

# inline: ***bi***, **b**, *i*, `code`
# Emphasis delimiters must not straddle a longer run: "**B4 +*t***" is bold("B4 +" italic("t")),
# and an italic span must not open on the second star of a "**" pair.
INLINE = re.compile(
    r"(\*\*\*.+?\*\*\*(?!\*)"
    r"|\*\*(?!\*).+?\*\*(?!\*)"
    r"|(?<!\*)\*(?!\*)[^\s].*?(?<!\*)\*(?!\*)"
    r"|`[^`]+`)", re.S)


def inline_runs(text: str, bold=False, italic=False) -> list[tuple[str, bool, bool, bool]]:
    """Split text into (chunk, bold, italic, code) runs.

    Recurses into bold and italic spans, so `code` nested inside *emphasis* is still recognised
    rather than leaving literal backticks in the output.
    """
    out = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("***") and part.endswith("***"):
            out += inline_runs(part[3:-3], True, True)
        elif part.startswith("**") and part.endswith("**"):
            out += inline_runs(part[2:-2], True, italic)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out += inline_runs(part[1:-1], bold, True)
        elif part.startswith("`") and part.endswith("`"):
            out.append((part[1:-1], bold, italic, True))
        else:
            out.append((unescape_md(part), bold, italic, False))
    return out


def unescape_md(text: str) -> str:
    """Drop Markdown backslash escapes: the reader should see the character, not the escape."""
    return re.sub(r"\\([*_|`\[\]\\])", r"\1", text)


def _split_row(line: str) -> list[str]:
    # a cell may contain an escaped pipe (\|t\|>2); split on unescaped pipes only
    cells = re.split(r"(?<!\\)\|", line.strip())
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return [c.strip().replace("\\|", "|") for c in cells]


def parse(md: str) -> list[tuple]:
    """Return a flat list of block tuples. Line numbers are 1-based and refer to the source."""
    lines = md.split("\n")
    blocks: list[tuple] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        st = line.strip()

        if not st:
            i += 1
            continue

        if st.startswith("```"):
            start = i + 1
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            i += 1
            blocks.append(("fence", "\n".join(b.strip() for b in buf), start))
            continue

        if st.startswith("> "):
            buf = []
            while i < len(lines) and (lines[i].strip().startswith(">") or
                                      (buf and lines[i].strip().startswith("```"))):
                buf.append(re.sub(r"^\s*> ?", "", lines[i]))
                i += 1
            blocks.append(("quote", parse("\n".join(buf)), i))
            continue

        if st.startswith("#"):
            level = len(st) - len(st.lstrip("#"))
            blocks.append((f"h{level}", st.lstrip("# ").strip(), i + 1))
            i += 1
            continue

        if st.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            header = rows[0]
            body = [r for r in rows[2:]] if len(rows) > 2 else []
            blocks.append(("table", header, body))
            continue

        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", st):
            blocks.append(("rule",))
            i += 1
            continue

        m = re.match(r"^(\s*)([*\-+]|\d+\.)\s+(.*)$", line)
        if m:
            indent = len(m.group(1))
            ordered = m.group(2)[0].isdigit()
            buf = [m.group(3)]
            i += 1
            while i < len(lines) and lines[i].strip() and \
                    not re.match(r"^\s*([*\-+]|\d+\.)\s+", lines[i]) and \
                    not lines[i].strip().startswith(("#", "|", "```", ">")):
                buf.append(lines[i].strip())
                i += 1
            blocks.append(("li", " ".join(buf), ordered, indent, m.group(2)))
            continue

        buf = [st]
        i += 1
        while i < len(lines) and lines[i].strip() and \
                not lines[i].strip().startswith(("#", "|", "```", ">", "---")) and \
                not re.match(r"^\s*([*\-+]|\d+\.)\s+", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        blocks.append(("para", " ".join(buf), i))
    return blocks
