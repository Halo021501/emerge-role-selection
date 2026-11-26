"""Article-paired outcome counts, transition audits and percentile intervals."""

from collections import Counter, defaultdict
import numpy as np

METHODS = ("baseline", "challenger")


MEASURES = {
    "executable_recall": ("hits", "executable_deprecate"),
    "p39_known_keep_error": ("wrong_p39_keep", "retrieved_p39_keep"),
    "unlabeled_write_share": ("unlabeled", "emitted"),
    "all_known_keep_error": ("wrong_keep", "retrieved_keep"),
}


def counts_for(row, method):
    pick = row[method]
    op = pick["operation"] if pick else None
    p39 = bool(pick and pick["ids"][1] == "P39")
    return {"executable_deprecate": row["executable_deprecate"], "retrieved_keep": row["retrieved_keep"],
            "retrieved_p39_keep": row["retrieved_p39_keep"], "emitted": int(pick is not None),
            "hits": int(op == "deprecate"), "hits_p39": int(p39 and op == "deprecate"),
            "wrong_keep": int(op == "exists"), "wrong_p39_keep": int(p39 and op == "exists"),
            "unlabeled": int(op == "unlabeled")}


def totals(rows, method):
    counts = Counter()
    for row in rows:
        counts.update(counts_for(row, method))
    return dict(counts)


def interval(values):
    return np.nanpercentile(values, [2.5, 97.5]).tolist()


def cluster_intervals(records, config):
    by_page = defaultdict(list)
    for r in records:
        by_page[r["page"]].append(r)
    columns = list(counts_for(records[0], "baseline"))
    matrix = []
    for page in sorted(by_page):
        sides = [totals(by_page[page], method) for method in METHODS]
        matrix.append([[counts[k] for k in columns] for counts in sides])
    matrix = np.asarray(matrix, dtype=np.int64)
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = []
    for start in range(0, config["bootstrap_draws"], 250):
        n = min(250, config["bootstrap_draws"] - start)
        ix = rng.integers(len(matrix), size=(n, len(matrix)))
        draws.append(matrix[ix].sum(axis=1))
    draws = np.concatenate(draws)
    observed = matrix.sum(axis=0)
    results = {}
    for name, (num, den) in MEASURES.items():
        ni, di = columns.index(num), columns.index(den)
        frac = np.divide(draws[:, :, ni], draws[:, :, di], where=draws[:, :, di] != 0,
                         out=np.full(draws[:, :, ni].shape, np.nan))
        point = observed[:, ni] / observed[:, di]
        block = {}
        for j, method in enumerate(METHODS):
            block[method] = {"numerator": int(observed[j, ni]), "denominator": int(observed[j, di]),
                             "rate": float(point[j]), "rate_95": interval(frac[:, j]),
                             "count_95": interval(draws[:, j, ni])}
        block["challenger_minus_baseline"] = {
            "rate": float(point[1] - point[0]), "rate_95": interval(frac[:, 1] - frac[:, 0]),
            "count": int(observed[1, ni] - observed[0, ni]), "count_95": interval(draws[:, 1, ni] - draws[:, 0, ni]),
        }
        results[name] = block
    return {"unit": "anchor article", "articles": len(matrix), "draws": len(draws),
            "seed": config["bootstrap_seed"], "interval": "paired percentile", "measures": results}


def change_audit(records):
    eligible = [r for r in records if r["eligible"]]
    changed = [r for r in eligible if r["changed"]]
    statuses = ("deprecate", "exists", "unlabeled")
    transitions = {old: {new: 0 for new in statuses} for old in statuses}
    for r in changed:
        transitions[r["baseline"]["operation"]][r["challenger"]["operation"]] += 1
    errors = [r for r in records if r["baseline"] and r["baseline"]["ids"][1] == "P39"
              and r["baseline"]["operation"] == "exists"]
    recoverable = [r for r in errors if any(c["operation"] == "deprecate" for c in r["scored_roles"])]
    fixed = [r for r in recoverable if r["challenger"]["operation"] == "deprecate"]
    return {
        "baseline_p39_writes": sum(bool(r["baseline"] and r["baseline"]["ids"][1] == "P39") for r in records),
        "eligible_passages": len(eligible), "eligible_articles": len({r["page"] for r in eligible}),
        "eligible_candidates": sum(len(r["scored_roles"]) for r in eligible),
        "changes": len(changed), "changed_articles": len({r["page"] for r in changed}),
        "changed_only_matrix": transitions, "eligible_baseline_operations": dict(Counter(r["baseline"]["operation"] for r in eligible)),
        "known_p39_errors": len(errors), "errors_with_another_role": sum(r["eligible"] for r in errors),
        "errors_with_confirmed_alternative": len(recoverable),
        "articles_with_confirmed_alternative": len({r["page"] for r in recoverable}),
        "confirmed_corrections": len(fixed), "corrected_articles": len({r["page"] for r in fixed}),
        "distinct_corrected_target_triples": len({tuple(r["challenger"]["ids"]) for r in fixed}),
        "recoverable_case_ids": [r["id"] for r in recoverable], "corrected_case_ids": [r["id"] for r in fixed],
        "correct_to_known_wrong": transitions["deprecate"]["exists"],
        "correct_to_unlabeled": transitions["deprecate"]["unlabeled"],
        "wrong_to_unlabeled": transitions["exists"]["unlabeled"],
    }
