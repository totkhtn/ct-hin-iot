"""Extract CT-HIN entities from v5.2 static-analysis records (``*.json.gz``), one JSON line per file.

    python scripts/extract_entities.py <directory or list.json> entities.jsonl.gz --jobs 4

Entity types (Section III-D of the paper). All entities are read from the ``strings_sample`` field of
the record; function signatures are read from ``cfg_graph``.

  s    normalised strings: strip, drop strings shorter than 4 characters, replace IPv4 addresses with <IP>,
       hexadecimal runs of 8+ digits with <HEX> and decimal numbers with #, truncate to 80 characters,
       then hash (BLAKE2b, 6 bytes, prefix "s:")
  r    raw strings of 6+ characters, hashed (BLAKE2b, 6 bytes), used only in the ablation
  f    semantic function signatures (set of semantic opcode classes of a function plus log2 of its size),
       hashed; used only in the ablation
  ip   IPv4 addresses with port when present; 0.*, 127.*, 255.* and version-like a.b.c.d with all parts < 10
       are dropped
  dom  domain-like tokens (alphanumeric labels and an alphabetic final label of 2-24 characters); common file
       suffixes (.so, .sh, .conf, ...) are dropped. Dotted file names such as s.upd can still match.
  url  http/https/ftp/tftp URLs without query string, lower-cased, truncated to 120 characters

The pre-registered configuration uses s, ip, dom and url. Strings are hashed so that the released
entity files do not reproduce program text; IP addresses, domain-like tokens and URLs are released as
indicators.
"""
import glob
import gzip
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

NORM_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
NORM_HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b")
NORM_NUM = re.compile(r"\d+")
IP = re.compile(r"(?<![\d.])((?:\d{1,3}\.){3}\d{1,3})(?::(\d{1,5}))?(?![\d.])")
URL = re.compile(r"\b((?:https?|ftp|tftp)://[^\s\"'<>]+)", re.I)
DOM = re.compile(r"\b((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})\b", re.I)
FILE_SUFFIXES = (".so", ".sh", ".conf", ".txt", ".bin", ".cfg", ".log", ".pid", ".lock", ".tmp", ".c", ".h", ".o",
                 ".py", ".pl", ".js", ".html", ".htm", ".php", ".xml", ".json", ".ini", ".d", ".mips", ".arm", ".x86",
                 ".mpsl", ".ppc", ".m68k", ".sh4", ".spc", ".arm7", ".arm5", ".arm6", ".i686", ".i586", ".x86_64",
                 ".elf", ".gz", ".zip", ".tar")


def h6(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=6).hexdigest()


def normalise(s: str):
    s = s.strip()
    if len(s) < 4:
        return None
    s = NORM_NUM.sub("#", NORM_HEX.sub("<HEX>", NORM_IP.sub("<IP>", s)))
    return s[:80]


def indicators(strings):
    ip, dom, url, raw = set(), set(), set(), set()
    for x in strings:
        if len(x) >= 6:
            raw.add(h6(x))
        for m in IP.finditer(x):
            octets = [int(v) for v in m.group(1).split(".")]
            if max(octets) > 255 or octets[0] in (0, 127, 255) or max(octets) < 10:
                continue
            ip.add(m.group(1) + (":" + m.group(2) if m.group(2) else ""))
        for m in URL.finditer(x):
            url.add(m.group(1).split("?")[0].rstrip("/").lower()[:120])
        for m in DOM.finditer(x):
            d = m.group(1).lower()
            if d.endswith(FILE_SUFFIXES) or re.fullmatch(r"[\d.]+", d):
                continue
            dom.add(d[:100])
    return sorted(ip), sorted(dom), sorted(url), sorted(raw)


def signatures(record):
    funcs = {}
    for node in ((record.get("cfg_graph") or {}).get("nodes") or []):
        g = funcs.setdefault(node.get("func"), {})
        for k, v in (node.get("sem_tf") or {}).items():
            if k != "invalid":
                g[k] = g.get(k, 0) + v
    out = set()
    for g in funcs.values():
        total = sum(g.values())
        if total >= 6 and len(g) >= 3:
            out.add("|".join(sorted(g)) + "#" + str(int(math.log2(total))))
    return sorted(out)


def extract(path):
    try:
        with gzip.open(path, "rt") as fh:
            rec = json.load(fh)
    except Exception as exc:  # unreadable record
        return {"error": repr(exc), "path": path}
    strings = [s for s in (rec.get("strings_sample") or []) if isinstance(s, str)]
    norm = sorted({n for n in (normalise(s) for s in strings) if n})
    ip, dom, url, raw = indicators(strings)
    return {"sha256": str(rec.get("file_hash", "")).lower(), "s": [h6("s:" + x) for x in norm], "r": raw,
            "f": [h6("f:" + x) for x in signatures(rec)], "ip": ip, "dom": dom, "url": url}


def main():
    src, out = sys.argv[1], sys.argv[2]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 2
    files = json.load(open(src)) if src.endswith(".json") else sorted(
        glob.glob(os.path.join(src, "**", "*.json.gz"), recursive=True))
    errors = 0
    with ProcessPoolExecutor(jobs) as ex, gzip.open(out, "wt") as fo:
        for r in ex.map(extract, files, chunksize=25):
            if "error" in r:
                errors += 1
                continue
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"{len(files) - errors} records written, {errors} unreadable", file=sys.stderr)


if __name__ == "__main__":
    main()
