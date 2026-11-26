"""Run the fixed-opportunity EMERGE text baseline and optional P39 comparison."""

import argparse
import json
from pathlib import Path
import sys

from inputs import (REPO, REF, check_file, data_root, load_cohort, model_encoder, records_digest,
                    reference, save_json, save_rows)
from metrics import change_audit, cluster_intervals, totals
from text_model import baseline

sys.path.insert(0, str(REPO / "src"))
from benchmarks.wrappers.prediction import Prediction

YEARS = (2020, 2022)
TEXT_NAME = "role-selection/text"


def opportunities(cohort, picks):
    result = {}
    for row, links, pick in zip(cohort.rows, cohort.links, picks, strict=True):
        if pick is None:
            continue
        incumbent = row["candidates"][pick]["ids"]
        if incumbent[1] != "P39":
            continue
        roles = [c for c in row["candidates"] if c["ids"][:2] == incumbent[:2]]
        if len(roles) > 1:
            key = (cohort.year, row["id"])
            result[key] = {"year": cohort.year, "id": row["id"], "page": row["page"],
                           "source": row["source"], "revision": row["revision"], "passage": row["passage"],
                           "links": links, "incumbent": incumbent, "candidates": roles}
    return result


def choices_from_file(path, tasks):
    choices = {}
    with path.open(encoding="utf-8") as source:
        for line, value in enumerate(source, 1):
            rec = json.loads(value)
            if (not isinstance(rec, dict) or type(rec.get("year")) is not int or
                    not isinstance(rec.get("id"), str) or not isinstance(rec.get("triple"), list)):
                raise ValueError(f"Expected year, id and triple on submission line {line}")
            key = (rec["year"], rec["id"])
            if key not in tasks or key in choices:
                raise ValueError(f"Unexpected or duplicate opportunity on line {line}: {key}")
            if rec["triple"] not in [c["ids"] for c in tasks[key]["candidates"]]:
                raise ValueError(f"Choice outside this person's frozen P39 candidates on line {line}: {key}")
            choices[key] = rec["triple"]
    if missing := tasks.keys() - choices.keys():
        raise ValueError(f"Missing {len(missing)} choices, first: {sorted(missing)[0]}")
    return choices


def bert_choices(tasks, args, release, config):
    from role_model import PairEncoder, Passage, RoleScorer, score_bags, torch
    from safetensors.torch import load_file
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but this PyTorch installation has no available CUDA device")
    torch.set_num_threads(4)
    enc_dir = model_encoder(args.encoder_dir, release, args.offline)
    encoder = PairEncoder(config, enc_dir)
    model = RoleScorer(config, enc_dir)
    model.load_state_dict(load_file(str(args.weights_dir / "role.safetensors")))
    model.to(device)
    bags, keys = [], []
    for key, task in tasks.items():
        visible = Passage(task["passage"], task["page"], task["links"], task["year"])
        roles = task["candidates"]
        bags.extend(encoder.person_roles(visible, roles, roles))
        keys.extend((key, tuple(c["ids"])) for c in roles)
    scores = dict(zip(keys, score_bags(model, bags, config["batch_pairs"], details=True), strict=True))
    choices = {}
    for key, task in tasks.items():
        old = tuple(task["incumbent"])
        best = max((tuple(c["ids"]) for c in task["candidates"]), key=lambda t: scores[key, t][0])
        gain = scores[key, best][0] - scores[key, old][0]
        choices[key] = list(best if gain > config["switch_logit_margin"] else old)
    print(f"P39 inference on {device} for {len(tasks)} opportunities", flush=True)
    return choices, scores


def decision_rows(cohort, base, folds, tasks, choices, scores):
    records = []
    for row, pick, fold in zip(cohort.rows, base, folds, strict=True):
        key = (cohort.year, row["id"])
        gold = {tuple(t): op for t, op in row["gold"]}
        candidates = {tuple(c["ids"]): c for c in row["candidates"]}

        def annotate(triple):
            c = candidates[tuple(triple)]
            return {"ids": c["ids"], "labels": c["labels"], "operation": gold.get(tuple(triple), "unlabeled")}

        before = annotate(row["candidates"][pick]["ids"]) if pick is not None else None
        after = annotate(choices[key]) if choices is not None and key in tasks else before
        roles = []
        if key in tasks:
            for c in tasks[key]["candidates"]:
                role = annotate(c["ids"])
                if scores:
                    logit, evidence = scores[key, tuple(c["ids"])]
                    role.update(role_logit=logit, evidence_windows=evidence)
                roles.append(role)
        keeps = [t for t in candidates if gold.get(t) == "exists"]
        rec = {"year": cohort.year, "id": row["id"], "page": row["page"], "revision": row["revision"],
               "source": row["source"], "baseline_calibration_fold": fold, "baseline": before,
               "eligible": key in tasks, "executable_deprecate": sum(op == "deprecate" for op in gold.values()),
               "retrieved_keep": len(keeps), "retrieved_p39_keep": sum(t[1] == "P39" for t in keeps),
               "scored_roles": roles}
        if choices is not None:
            rec.update(challenger=after, changed=before != after)
            if ((before is None) != (after is None) or rec["changed"] and
                    (key not in tasks or before["ids"][:2] != after["ids"][:2] or after["ids"][1] != "P39")):
                raise ValueError(f"Write opportunity or person changed for {key}")
        records.append(rec)
    return records


def wrapper_record(record, methods):
    predictions = {}
    for side, name in methods.items():
        pick = record[side]
        prediction = Prediction()
        if pick:
            prediction.add_predicted_triple(pick["labels"][1], pick["ids"], pick["labels"], operation="DEPRECATE")
        predictions[name] = {"model": name, "model_type": "operation-tagged",
                             "predicted_triples": prediction.predicted_triples}
    return {"hash_id": record["id"], "predictions": predictions}


def run(args):
    release, config = reference()
    data_dir = data_root(args.data_dir, YEARS, release)
    paired = not args.text_only
    official_weights = args.weights_dir.resolve() == REF.resolve()
    if official_weights:
        check_file(REF / "text.npz", release["sha256"]["text.npz"])
        if paired and args.choices is None:
            check_file(REF / "role.safetensors", release["sha256"]["role.safetensors"])
    methods = {"baseline": TEXT_NAME}
    if paired:
        methods["challenger"] = "role-selection/external" if args.choices else "role-selection/bert"
    mode = "external" if args.choices else "paired" if paired else "text"
    dest = (args.output / mode).resolve()
    tasks, cohorts = {}, {}
    for year in YEARS:
        cohort = load_cohort(year, data_dir, args.cache_dir, release, args.offline)
        picks, folds = baseline(cohort, args.weights_dir / "text.npz", release)
        tasks.update(opportunities(cohort, picks))
        cohorts[year] = (cohort, picks, folds)
    choices, scores = None, {}
    if args.choices:
        choices = choices_from_file(args.choices, tasks)
    elif paired:
        choices, scores = bert_choices(tasks, args, release, config)
    summary = {"dataset_revision": release["dataset_revision"], "original_policy_lock_sha256":
               release["original_policy_lock_sha256"], "methods": methods, "years": {}}
    all_records = []
    for year, (cohort, picks, folds) in cohorts.items():
        recs = decision_rows(cohort, picks, folds, tasks, choices, scores)
        all_records.extend(recs)
        spec = release["years"][str(year)]
        result = {"passages": len(recs), "articles": len({r["page"] for r in recs}),
                  "eligible": sum(r["eligible"] for r in recs), "counts": {}, "decision_sha256": {}}
        for side in methods:
            result["counts"][side] = counts = totals(recs, side)
            signature = records_digest([r["id"], r[side]["ids"] if r[side] else None] for r in recs)
            result["decision_sha256"][side] = signature
            expected = "baseline" if side == "baseline" else "role_model"
            if official_weights and (side == "baseline" or not args.choices):
                if signature != spec["decision_sha256"][expected] or counts != spec["expected_counts"][expected]:
                    raise ValueError(f"{year} {side} differs from the original decisions or counts")
            print(f"{year} {methods[side]}: {counts['hits']}/{counts['executable_deprecate']} hits, "
                  f"{counts['wrong_p39_keep']}/{counts['retrieved_p39_keep']} known P39 errors, "
                  f"{counts['unlabeled']}/{counts['emitted']} unlabeled writes", flush=True)
        if paired:
            result.update(changes=change_audit(recs), paired_bootstrap=cluster_intervals(recs, config))
        summary["years"][str(year)] = result

    save_rows(dest / "opportunities.jsonl", tasks.values())
    save_rows(dest / "decisions.jsonl", all_records)
    if paired:
        save_rows(dest / "changes.jsonl", (r for r in all_records if r["changed"]))
        save_rows(dest / "choices.jsonl", ({"year": k[0], "id": k[1], "triple": t} for k, t in choices.items()))
    save_rows(dest / "upstream" / "predictions.jsonl", (wrapper_record(r, methods) for r in all_records))
    save_json(dest / "merge_config.json", {
        "input_main_dataset_path": str((data_dir / "evaluation_set").resolve()),
        "output_merged_dataset_path": str(dest / "merged"),
        "input_other_datasets_paths": [
            {"dataset_title": name, "input_dataset_path": str(dest / "upstream"), "merge_predictions": True,
             "predictions_model_name": name, "predictions_model_type": "tool"} for name in methods.values()]})
    save_json(dest / "metrics.json", summary)
    print(f"Results written to {dest}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--text-only", action="store_true", help="Run the baseline and export label-free opportunities")
    mode.add_argument("--choices", type=Path, help="Evaluate one supplied triple per eligible passage")
    p.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA if available, otherwise CPU")
    p.add_argument("--data-dir", type=Path, help="Root containing evaluation_set and kg_snapshots")
    p.add_argument("--cache-dir", type=Path, default=REPO / "output/role_selection/cache")
    p.add_argument("--encoder-dir", type=Path, default=REPO / "data/.cache/role_selection/bert_tiny")
    p.add_argument("--weights-dir", type=Path, default=REF, help="Directory containing text.npz and role.safetensors")
    p.add_argument("--output", type=Path, default=REPO / "output/role_selection")
    p.add_argument("--offline", action="store_true", help="Use verified local inputs only")
    run(p.parse_args())


if __name__ == "__main__":
    main()
