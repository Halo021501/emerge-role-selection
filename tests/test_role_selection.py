"""Keep user-provided picks paired and preserve DEPRECATE through upstream merge."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/role_selection"))
from run import choices_from_file, wrapper_record
from inputs import reference
from merge.s03_merge_predictions import merge_predictions_into_main_dataset
from evaluation.preloader.preloader import EvaluationPreLoader


# An actual 2022 opportunity in the release.
KEY = (2022, "WlBFgtzLss")
VP = ["Q7351526", "P39", "Q7925068"]
MEP = ["Q7351526", "P39", "Q27169"]
TASKS = {KEY: {"candidates": [{"ids": VP}, {"ids": MEP}]}}
PICK = {"year": KEY[0], "id": KEY[1], "triple": VP}


def test_release_fingerprints_and_eligible_pick(tmp_path):
    release, _ = reference()
    assert release["years"]["2022"]["passages"] == 496
    p = tmp_path / "choices.jsonl"
    p.write_text(json.dumps(PICK) + "\n", encoding="utf-8")
    assert choices_from_file(p, TASKS) == {KEY: VP}


@pytest.mark.parametrize("entries", [[], [PICK, PICK], [dict(PICK, year=2020)],
                                     [dict(PICK, triple=["Q7351526", "P39", "Q1"])]])
def test_external_model_must_fill_each_slot_with_a_locked_role(tmp_path, entries):
    p = tmp_path / "choices.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in entries), encoding="utf-8")
    with pytest.raises(ValueError):
        choices_from_file(p, TASKS)


def test_merge_and_route_both_a_write_and_an_abstention(tmp_path):
    name = "role-selection/bert"
    pick = {"ids": VP, "labels": ["Roberta Metsola", "position held", "Vice President of the European Parliament"]}
    records = [wrapper_record({"id": KEY[1], "challenger": pick}, {"challenger": name}),
               wrapper_record({"id": "abstention", "challenger": None}, {"challenger": name})]
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    main = [{"hash_id": r["hash_id"], "predictions": {}} for r in records]
    merge_predictions_into_main_dataset(main, {r["hash_id"]: i for i, r in enumerate(main)},
                                        {"input_dataset_path": str(tmp_path), "predictions_model_name": name,
                                         "predictions_model_type": "tool"}, {})
    for row, expected in zip(main, [{tuple(VP)}, set()], strict=True):
        entry = row["predictions"][name]
        assert entry["model_type"] == "operation-tagged"
        closed = {name: defaultdict(set)}
        EvaluationPreLoader.process_kg_prompt_predictions(name, entry, {name: defaultdict(set)}, closed, {})
        assert closed[name]["d-triples"] == expected
