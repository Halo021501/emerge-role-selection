"""Refit the two released scorers on the locked 2019 passages."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from inputs import REPO, REF, data_root, load_cohort, model_encoder, reference, save_json, sha256, staging
from text_model import fit_text


def fit(args):
    from role_model import PairEncoder, Passage, fit_epochs, torch
    from safetensors.torch import save_file
    if args.output.resolve() == REF.resolve():
        raise ValueError("Choose an output directory for the refit distinct from the released weights")
    release, config = reference()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = load_cohort(2019, data_root(args.data_dir, (2019,), release), args.cache_dir, release, args.offline)
    path = model_encoder(args.encoder_dir, release, args.offline, pretrained=True)
    encoder = PairEncoder(config, path)
    bags, labels = [], []
    for row, links in zip(train.rows, train.links, strict=True):
        gold = {tuple(t): op for t, op in row["gold"]}
        groups = defaultdict(list)
        for c in row["candidates"]:
            if c["ids"][1] == "P39":
                groups[c["ids"][0]].append(c)
        visible = Passage(row["passage"], row["page"], links, 2019)
        for roles in groups.values():
            known = [c for c in roles if tuple(c["ids"]) in gold]
            if known:
                bags.extend(encoder.person_roles(visible, roles, known))
                labels.extend(int(gold[tuple(c["ids"])] == "deprecate") for c in known)
    if Counter(labels) != {0: 114, 1: 46}:
        raise ValueError("2019 explicit P39 supervision changed")
    model, losses = fit_epochs(bags, labels, config, release["selected_epoch"], device, path)
    with staging(args.output / "role.safetensors") as temp:
        save_file({k: t.cpu().contiguous() for k, t in model.state_dict().items()}, str(temp))
    with staging(args.output / "text.npz") as temp, temp.open("wb") as stream:
        fit_text(train.rows, stream)
    info = {"seed": config["seed"], "epoch": release["selected_epoch"], "torch": torch.__version__,
            "device": device, "dataset_revision": release["dataset_revision"], "loss": losses,
            "role_supervision": {"deprecate": 46, "exists": 114},
            "sha256": {f: sha256(args.output / f) for f in ("role.safetensors", "text.npz")}}
    save_json(args.output / "training.json", info)
    print(f"Refit both scorers from 2019 in {args.output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--cache-dir", type=Path, default=REPO / "output/role_selection/cache")
    p.add_argument("--encoder-dir", type=Path, default=REPO / "data/.cache/role_selection/bert_tiny")
    p.add_argument("--output", type=Path, default=REPO / "output/role_selection/refit")
    p.add_argument("--offline", action="store_true")
    fit(p.parse_args())
