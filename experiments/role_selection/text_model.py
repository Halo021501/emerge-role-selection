"""2019 text/old-triple logistic baseline and its fixed article-fold policies."""

from collections import defaultdict
import hashlib

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def article_folds(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["page"]].append(index)
    def order(page):
        return -len(groups[page]), hashlib.sha256(f"emerge-fusion-1/{page}".encode()).digest()
    folds = np.empty(len(rows), dtype=int)
    loads = [0] * 5
    for page in sorted(groups, key=order):
        i = min(range(5), key=loads.__getitem__)
        folds[groups[page]] = i
        loads[i] += len(groups[page])
    return folds


def describe(c):
    text = " | ".join(c["labels"])
    if c["alternatives"]:
        text += " . Other values: " + "; ".join(c["alternatives"])
    return text


def vectorizer(vocabulary=None):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=16000,
                           sublinear_tf=True, vocabulary=vocabulary)


def columns(rows):
    candidates = [c for r in rows for c in r["candidates"]]
    passages = [r["passage"] for r in rows]
    docs = np.repeat(np.arange(len(rows)), [len(r["candidates"]) for r in rows])
    numeric = np.asarray([c["lexical"] for c in candidates], dtype=np.float32)
    return passages, docs, [describe(c) for c in candidates], numeric


def feature_matrix(vocab, passages, docs, triples, numeric):
    return hstack([vocab.transform(passages)[docs], vocab.transform(triples), csr_matrix(numeric)], format="csr")


def fit_text(rows, path):
    passages, docs, triples, numeric = columns(rows)
    labels = []
    for row in rows:
        gold = {tuple(t): op for t, op in row["gold"]}
        for c in row["candidates"]:
            labels.append({"deprecate": 1, "exists": 0}.get(gold.get(tuple(c["ids"])), -1))
    labels = np.asarray(labels)
    known = np.flatnonzero(labels >= 0)
    vocab = vectorizer().fit(passages + [triples[i] for i in known])
    scaler = StandardScaler().fit(numeric[known])
    features = feature_matrix(vocab, passages, docs, triples, scaler.transform(numeric))
    model = LogisticRegression(C=0.7, class_weight="balanced", max_iter=600, random_state=73)
    model.fit(features[known], labels[known])
    np.savez_compressed(path, terms=np.asarray(vocab.get_feature_names_out(), dtype=str),
                        idf=vocab.idf_, coefficients=model.coef_[0], intercept=model.intercept_,
                        mean=scaler.mean_, scale=scaler.scale_)


def baseline(cohort, weights, release):
    rows = cohort.rows
    passages, docs, triples, numeric = columns(rows)
    with np.load(weights, allow_pickle=False) as params:
        vocab = vectorizer({word: i for i, word in enumerate(params["terms"].tolist())})
        vocab.fit([""])
        vocab.idf_ = params["idf"]
        # StandardScaler uses in-place float32 subtraction and division on these inputs.
        numeric -= params["mean"]
        numeric /= params["scale"]
        features = feature_matrix(vocab, passages, docs, triples, numeric)
        logits = features @ params["coefficients"] + params["intercept"][0]
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite text score")
    folds = article_folds(rows).tolist() if cohort.year == 2020 else [None] * len(rows)
    picks = []
    start = 0
    for row, fold in zip(rows, folds, strict=True):
        stop = start + len(row["candidates"])
        chosen = None
        if stop > start:
            i = int(np.argmax(logits[start:stop]))
            policy = release["text_deployment_policy"] if fold is None else release["text_2020_fold_policies"][fold]
            group = int(row["candidates"][i]["ids"][1] != "P39")
            threshold = policy["thresholds"][group]
            if threshold is not None and (logits[start+i] - policy["mean"]) / policy["scale"] >= threshold:
                chosen = i
        picks.append(chosen)
        start = stop
    return picks, folds
