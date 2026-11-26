"""Rebuild the locked candidate lists from pinned public EMERGE inputs."""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import time

from retrieval import linked_ids, retrieve, snapshot_subset, supported_labels

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REF = HERE / "reference"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(record):
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records_digest(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(canonical(r))
    return h.hexdigest()


def check_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


@contextmanager
def staging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as f:
        temp = Path(f.name)
    try:
        yield temp
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_rows(path, rows):
    with staging(path) as temp, temp.open("wb") as output:
        for r in rows:
            output.write(canonical(r))


def save_json(path, value):
    with staging(path) as temp:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_file(root, rel, expected, url, offline):
    path = root / rel
    if path.is_file():
        check_file(path, expected)
        return path
    if offline:
        raise FileNotFoundError(f"Missing pinned input in offline mode: {path}")
    import requests
    print(f"Fetching {rel}", flush=True)
    with staging(path) as temp, requests.Session() as session:
        size = None
        failures = 0
        while size is None or temp.stat().st_size < size:
            offset = temp.stat().st_size
            end = offset + 32 * 1024 * 1024 - 1
            try:
                with session.get(url, headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
                                 stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        part = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not part or int(part[1]) != offset:
                            raise OSError(f"Unexpected download range for {rel}")
                        size = int(part[3])
                    with temp.open("ab" if response.status_code == 206 else "wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                    if response.status_code != 206:
                        break
                    if temp.stat().st_size != int(part[2]) + 1:
                        raise OSError(f"Short download for {rel}")
                    failures = 0
            except (OSError, requests.RequestException):
                failures += 1
                if failures == 4:
                    raise
                time.sleep(failures)
        check_file(temp, expected)
    return path


def reference():
    release = read_json(REF / "release.json")
    check_file(HERE / "config.json", release["config_sha256"])
    check_file(REF / "cohorts.json", release["sha256"]["cohorts.json"])
    return release, read_json(HERE / "config.json")


def data_root(explicit, years, release):
    if explicit is not None:
        return explicit
    official = REPO / "data"
    files = [f for year in years for f in release["years"][str(year)]["shards"]]
    files.extend(f"kg_snapshots/kg_snapshot_{year}-01-01.tsv.gz" for year in years)
    if all((official / f).is_file() for f in files):
        return official
    return REPO / "output/role_selection/public-data"


def model_encoder(root, release, offline, pretrained=False):
    files = ["config.json", "vocab.txt"]
    if pretrained:
        files.append("model.safetensors")
    for name in files:
        url = f"https://huggingface.co/{release['encoder']}/resolve/{release['encoder_revision']}/{name}?download=true"
        public_file(root, name, release["encoder_sha256"][name], url, offline)
    return root


@dataclass
class Cohort:
    year: int
    rows: list
    links: list


def load_cohort(year, data_dir, cache, release, offline):
    spec = release["years"][str(year)]
    members = read_json(REF / "cohorts.json")[str(year)]
    shards = sorted(spec["shards"])
    wanted = defaultdict(dict)
    for index, (key, shard, line) in enumerate(members):
        wanted[shards[shard]][line] = (index, key)
    if len(set(key for key, _, _ in members)) != len(members):
        raise ValueError("Duplicate identifier in cohort")
    raw = [None] * len(members)
    labels = [None] * len(members)
    url = f"https://huggingface.co/datasets/{release['dataset']}/resolve/{release['dataset_revision']}/"
    for shard in shards:
        path = public_file(data_dir, shard, spec["shards"][shard], url + shard + "?download=true", offline)
        with path.open(encoding="utf-8") as stream:
            for line, value in enumerate(stream, 1):
                if line not in wanted[shard]:
                    continue
                i, key = wanted[shard][line]
                r = json.loads(value)
                if key != r["hash_id"]:
                    raise ValueError(f"Passage identity changed in {shard} at {line}")
                labels[i] = supported_labels(r)
                raw[i] = {k: r[k] for k in ("hash_id", "passage", "anchor_page_id", "revision_id", "mentions")}
                raw[i].update(anchor_page_qid=r.get("anchor_page_qid"), source=f"{shard}:{line}")
    if any(r is None for r in raw):
        raise ValueError(f"Missing passages for {year}")
    path = cache / f"candidates-{year}-{spec['candidate_rows_sha256'][:12]}.jsonl.gz"
    if path.is_file():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(s) for s in stream]
    else:
        graph = f"kg_snapshots/kg_snapshot_{year}-01-01.tsv.gz"
        snapshot = public_file(data_dir, graph, spec["snapshot_sha256"], url + graph + "?download=true", offline)
        print(f"Retrieving {year} old-snapshot candidates for {len(raw)} passages", flush=True)
        subjects = {s for r in raw for s in linked_ids(r)}
        targets = {t for gold in labels for t in gold}
        edges, present = snapshot_subset(snapshot, subjects, targets)
        rows = [{"id": r["hash_id"], "page": r.get("anchor_page_qid") or str(r["anchor_page_id"]),
                 "revision": r["revision_id"], "source": r["source"], "passage": r["passage"],
                 "candidates": retrieve(r, edges),
                 "gold": [[t, op] for t, op in sorted(gold.items()) if t in present]}
                for r, gold in zip(raw, labels, strict=True)]
        if records_digest(rows) != spec["candidate_rows_sha256"]:
            raise ValueError(f"The rebuilt {year} cohort differs from the original candidate lock")
        with staging(path) as temp, gzip.open(temp, "wb") as stream:
            for row in rows:
                stream.write(canonical(row))
    if records_digest(rows) != spec["candidate_rows_sha256"]:
        raise ValueError(f"Candidate integrity check failed for {path}")
    links = []
    for r in raw:
        spans = []
        for m in r["mentions"]:
            a, b, s = m["start_char"], m["end_char"], m["mention_text"]
            if r["passage"][a:b] != s:
                raise ValueError(f"Hyperlink span changed in {r['source']}")
            if m.get("qid") and b > a:
                spans.append((a, b, m["qid"], s))
        links.append(tuple(spans))
    return Cohort(year, rows, links)
