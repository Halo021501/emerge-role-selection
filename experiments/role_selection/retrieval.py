"""Frozen old-graph retrieval used for the P39 paired evaluation."""

from collections import defaultdict
from dataclasses import dataclass
import gzip
import math
import re
import unicodedata


def snapshot_subset(path, wanted, target_ids):
    """Stream six-column snapshot rows once, checking target membership separately."""
    heads = {s.encode() for s in wanted}
    targets = {tuple(s.encode() for s in t): t for t in target_ids}
    relevant = heads | {t[0] for t in targets}
    found, present = defaultdict(list), set()
    with gzip.open(path, "rb") as stream:
        for line in stream:
            head = line.partition(b"\t")[0]
            if head not in relevant:
                continue
            parts = line.rstrip(b"\r\n").split(b"\t", 5)
            if len(parts) != 6:
                raise ValueError(f"Invalid snapshot row in {path.name}")
            key = tuple(parts[:3])
            if key in targets:
                present.add(targets[key])
            if head in heads:
                ids = tuple(s.decode() for s in key)
                found[ids[0]].append(Edge(ids, tuple(s.decode() for s in parts[3:])))
    return found, present

ASSESSOR = "Meta-Llama-3.1-405B_prompt_v1"


TOP_K = 32


OPERATION = {
    "x-triples": ("exists", "triple_assertion"),
    "d-triples": ("deprecate", "triple_deprecation"),
}


TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


STOP = set("a an and as at by for from in is it of on or that the to was were with".split())


def words(text):
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def supported_labels(row, head_before=None):
    by_triple = defaultdict(set)
    for record in row["tkgu_triples"]:
        if head_before and record["triple"][0] >= head_before:
            continue
        for operation, (label, prompt) in OPERATION.items():
            if operation in record["tkgu_operations"] and any(
                item.get("llm_assessment") is True
                and item.get("llm_name") == ASSESSOR
                and item.get("llm_prompt_type") == prompt
                for item in record["llm_assessment"]
            ):
                by_triple[tuple(record["triple"])].add(label)
    return {triple: labels.pop() for triple, labels in by_triple.items() if len(labels) == 1}


def linked_ids(row):
    linked = {mention.get("qid") for mention in row["mentions"]}
    linked.add(row.get("anchor_page_qid"))
    return {qid for qid in linked if isinstance(qid, str) and re.fullmatch(r"Q\d+", qid)}


@dataclass(frozen=True, slots=True)
class Edge:
    ids: tuple
    labels: tuple


def token_match(label, query, query_tokens):
    tokens = words(label)
    if not tokens:
        return 0.0, 0.0
    phrase = " " + " ".join(tokens) + " "
    relevant = set(tokens) - STOP or set(tokens)
    return float(phrase in query), len(relevant & query_tokens) / len(relevant)


def retrieve(row, by_head):
    mentions = linked_ids(row)
    passage = words(row["passage"])
    query = " " + " ".join(passage) + " "
    tokens = set(passage)
    anchor = row.get("anchor_page_qid")
    match = {}
    properties = defaultdict(dict)
    found = {}
    for head in sorted(mentions):
        for edge in by_head.get(head, ()):
            if edge.ids in found:
                continue
            for name in edge.labels:
                if name not in match:
                    match[name] = token_match(name, query, tokens)
            subject, relation, tail = (match[name] for name in edge.labels)
            anchor_match = float(head == anchor)
            tail_linked = float(edge.ids[2] in mentions)
            score = 1.5 * anchor_match + 2 * tail_linked + 0.5 * subject[0]
            score += relation[1] + 1.5 * tail[0] + 0.5 * tail[1]
            properties[edge.ids[:2]][edge.ids[2]] = (tail_linked, tail[0], tail[1], edge.labels[2])
            features = [anchor_match, tail_linked, *subject, *relation, *tail, score]
            found[edge.ids] = (edge, features)
    ranked = sorted(found.values(), key=lambda item: (-item[1][-1], item[0].ids))[:TOP_K]
    result = []
    for rank, (edge, features) in enumerate(ranked, 1):
        alternatives = sorted(
            (
                (quality, qid)
                for qid, quality in properties[edge.ids[:2]].items()
                if qid != edge.ids[2]
            ),
            key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1]),
        )
        features.extend(
            [1 / rank, math.log1p(len(alternatives)), float(any(q[0] for q, _ in alternatives))]
        )
        result.append(
            {
                "ids": edge.ids,
                "labels": edge.labels,
                "alternatives": [item[0][-1] for item in alternatives[:3]],
                "lexical": features,
            }
        )
    return result
