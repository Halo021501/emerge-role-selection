"""Sentence-level, entity-pair-conditioned scoring for a single old P39 triple."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import re
import unicodedata

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from retrieval import STOP

WORD = re.compile(r"[^\W_]+", re.UNICODE)
BREAK = re.compile(r"(?<=[.!?])\s+|[\n;*]+|={2,}")
HEADER = re.compile(r"^Excerpt from the Wikipedia page describing [^\n]*?:\s*")
YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
DATE_PARTS = r"(?:[\s,./-]+(?:on|in|the|[a-zA-Z]{3,9}|\d{1,2}(?:st|nd|rd|th)?)){0,3}[\s,./-]*$"
END_DATE = re.compile(r"(?:\b(?:to|through|until|till|ending|ended)\b|[–—-])" + DATE_PARTS, re.I)
START_DATE = re.compile(r"\b(?:from|since|beginning|starting|began)\b" + DATE_PARTS, re.I)
FEATURES = (
    "article_subject", "subject_coverage", "subject_link", "role_coverage", "role_exact",
    "role_link", "joint_coverage", "person_role_proximity", "year_current", "year_previous",
    "year_older", "year_future", "end_current", "end_previous", "end_older", "start_current",
    "start_previous", "start_older",
)


@lru_cache(maxsize=32000)
def normalize(word):
    return "".join(c for c in unicodedata.normalize("NFKD", word.casefold())
                   if not unicodedata.combining(c))


def terms(text):
    return tuple(normalize(w) for w in WORD.findall(text))


@dataclass(frozen=True)
class Passage:
    text: str
    page: str
    links: tuple   # (absolute start, end, Wikidata ID, surface form)
    year: int     # Old snapshot calendar year


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    text: str
    words: tuple
    offsets: tuple


def segments(passage, config):
    text = passage.text
    header = HEADER.match(text)
    start = header.end() if header else 0
    spans = []
    for boundary in BREAK.finditer(text, start):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(text)))
    result = []
    for left, right in spans:
        tokens = list(WORD.finditer(text, left, right))
        for first in range(0, len(tokens), config["window_stride"]):
            group = tokens[first:first + config["window_words"]]
            a = tokens[first].start() if first else left
            last = first + len(group) == len(tokens)
            b = right if last else group[-1].end()
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            result.append(Segment(a, b, text[a:b], tuple(normalize(m[0]) for m in group),
                                  tuple((m.start(), m.end()) for m in group)))
            if last:
                break
    if not result:
        result.append(Segment(0, len(text), text, (), ()))
    return result


@dataclass(frozen=True)
class Alignment:
    coverage: float = 0.0
    exact: bool = False
    linked: bool = False
    middle: float = 0.0


def align(seg, name, qid, links, token_weights):
    surface_forms = {terms(name)}
    for start, end, target, surface in links:
        if qid != target:
            continue
        if seg.start <= start < end <= seg.end:
            return Alignment(1.0, True, True, (start + end) / 2)
        surface_forms.add(terms(surface))
    best = Alignment()
    for form in sorted(surface_forms):
        if not form:
            continue
        useful = set(form) - STOP or set(form)
        total = sum(token_weights.get(w, 1.0) for w in sorted(useful))
        for i, w in enumerate(seg.words):
            if w == form[0] and seg.words[i:i + len(form)] == form:
                return Alignment(1.0, True, False, (seg.offsets[i][0] + seg.offsets[i + len(form) - 1][1]) / 2)
            if w not in useful:
                continue
            # A scattered country name elsewhere in the sentence is a weaker match.
            here = seg.words[i:i + len(form) + 3]
            hits = useful.intersection(here)
            coverage = sum(token_weights.get(t, 1.0) for t in sorted(hits)) / total
            if coverage > best.coverage:
                last = max(k for k, t in enumerate(here) if t in hits)
                best = Alignment(coverage, False, False, (seg.offsets[i][0] + seg.offsets[i + last][1]) / 2)
    return best


def date_features(seg, role, year):
    # Distance ties the calendar signal to this office mention. The network fits
    # coefficients for current, previous, earlier and future years from 2019 labels.
    values = np.zeros(10, dtype=np.float32)
    for match in YEAR.finditer(seg.text):
        delta = int(match[0]) - year
        bucket = 0 if delta == 0 else 1 if delta == -1 else 2 if delta < 0 else 3
        distance = abs(seg.start + match.start() - role.middle)
        weight = role.coverage * math.exp(-distance / 100.0)
        values[bucket] = max(values[bucket], weight)
        if bucket < 3:
            prefix = seg.text[max(0, match.start() - 48):match.start()]
            for offset, pattern in ((4, END_DATE), (7, START_DATE)):
                if pattern.search(prefix):
                    values[offset + bucket] = max(values[offset + bucket], weight)
    return values.tolist()


@dataclass
class Window:
    ids: list
    groups: list
    context_start: int
    features: list
    prior: float
    seg: Segment


class PairEncoder:
    def __init__(self, config, encoder_dir):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)

    @lru_cache(maxsize=12000)
    def wordpieces(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                     max_length=self.config["max_tokens"])

    def pair(self, passage, parts, candidate, token_weights):
        person, _, office = candidate["labels"]
        subject, _, obj = candidate["ids"]
        person_ids = self.wordpieces(person)[:24] or [self.tokenizer.unk_token_id]
        office_ids = self.wordpieces(office)[:36] or [self.tokenizer.unk_token_id]
        sep = self.tokenizer.sep_token_id
        query = [self.tokenizer.cls_token_id, *person_ids, sep, *office_ids, sep]
        qgroups = [0, *([1] * len(person_ids)), 0, *([2] * len(office_ids)), 0]
        matches = []
        for seg in parts:
            who = align(seg, person, subject, passage.links, {})
            role = align(seg, office, obj, passage.links, token_weights)
            proximity = who.coverage * role.coverage * math.exp(-abs(who.middle - role.middle) / 100)
            features = [float(subject == passage.page), who.coverage, float(who.linked),
                        role.coverage, float(role.exact), float(role.linked),
                        who.coverage * role.coverage, proximity, *date_features(seg, role, passage.year)]
            # A relevance prior only locates the entity pair. It contains no operation or date signal.
            prior = 2.5 * role.coverage + float(role.linked) + 0.75 * who.coverage
            matches.append((prior, seg, features))
        ranked = sorted(matches, key=lambda item: (-item[0], item[1].start))
        windows = []
        for prior, seg, features in ranked[:self.config["max_windows"]]:
            content = self.wordpieces(seg.text)[:self.config["max_tokens"] - len(query) - 1]
            content = content or [self.tokenizer.unk_token_id]
            windows.append(Window([*query, *content, sep], [*qgroups, *([3] * len(content)), 0],
                                  len(query), features, prior, seg))
        return windows

    def person_roles(self, passage, roles, selected):
        # IDF within the old role inventory downweights words shared by every office.
        df = Counter(term for role in roles for term in set(terms(role["labels"][2])))
        weights = {w: 1 + math.log((1 + len(roles)) / (1 + count)) for w, count in df.items()}
        parts = segments(passage, self.config)
        return [self.pair(passage, parts, c, weights) for c in selected]


def seed_run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class RoleScorer(nn.Module):
    def __init__(self, config, encoder_dir, pretrained=False):
        super().__init__()
        if pretrained:
            self.bert = AutoModel.from_pretrained(encoder_dir, local_files_only=True)
        else:
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(encoder_dir, local_files_only=True))
        size = self.bert.config.hidden_size
        self.register_buffer("center", torch.zeros(len(FEATURES)))
        self.register_buffer("scale", torch.ones(len(FEATURES)))
        self.relevance = nn.Linear(size + len(FEATURES), 1)
        self.support = nn.Sequential(nn.Dropout(config["dropout"]),
                                     nn.Linear(size * 2 + len(FEATURES), 64), nn.GELU(),
                                     nn.Dropout(config["dropout"]), nn.Linear(64, 1))
        self.direct = nn.Linear(len(FEATURES), 1, bias=False)
        nn.init.zeros_(self.relevance.weight)
        nn.init.zeros_(self.relevance.bias)
        nn.init.zeros_(self.direct.weight)

    def fit_scale(self, bags):
        features = np.asarray([w.features for bag in bags for w in bag], np.float32)
        self.center.copy_(torch.from_numpy(features.mean(axis=0)))
        self.scale.copy_(torch.from_numpy(np.maximum(features.std(axis=0), 0.25)))

    def forward(self, bags):
        windows = [w for bag in bags for w in bag]
        width = max(len(w.ids) for w in windows)
        ids = np.zeros((len(windows), width), dtype=np.int64)
        types = np.zeros_like(ids)
        groups = np.zeros_like(ids)
        for i, w in enumerate(windows):
            end = len(w.ids)
            ids[i, :end] = w.ids
            groups[i, :end] = w.groups
            types[i, w.context_start:end] = 1
        device = self.center.device
        ids = torch.as_tensor(ids, device=device)
        groups = torch.as_tensor(groups, device=device)
        states = self.bert(input_ids=ids, attention_mask=ids.ne(0),
                           token_type_ids=torch.as_tensor(types, device=device), output_attentions=True)
        att = states.attentions[-1].mean(dim=1)

        def from_entity(number):
            mask = (groups == number).to(att.dtype)
            return torch.bmm((mask / mask.sum(1, keepdim=True)).unsqueeze(1), att).squeeze(1)

        joint = from_entity(1) * from_entity(2) * (groups == 3)
        joint = joint / joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
        context = torch.bmm(joint.unsqueeze(1), states.last_hidden_state).squeeze(1)
        raw = torch.tensor([w.features for w in windows], device=device, dtype=context.dtype)
        features = (raw - self.center) / self.scale
        scores = self.support(torch.cat((states.last_hidden_state[:, 0], context, features), dim=1))
        scores = (scores + self.direct(features)).flatten()
        priors = torch.tensor([w.prior for w in windows], device=device, dtype=context.dtype)
        relevance = priors + self.relevance(torch.cat((context, features), dim=1)).flatten()
        sizes = [len(b) for b in bags]
        weights = [r.softmax(0) for r in relevance.split(sizes)]
        logits = torch.stack([(s * a).sum() for s, a in zip(scores.split(sizes), weights, strict=True)])
        return logits, scores, torch.cat(weights)


def fit_epochs(bags, labels, config, epochs, device, encoder_dir):
    seed_run(config["seed"])
    model = RoleScorer(config, encoder_dir, pretrained=True)
    model.fit_scale(bags)
    model.to(device)
    heads = [p for name, p in model.named_parameters() if not name.startswith("bert.")]
    optimizer = torch.optim.AdamW([{"params": model.bert.parameters(), "lr": config["encoder_lr"]},
                                   {"params": heads, "lr": config["head_lr"]}],
                                  weight_decay=config["weight_decay"])
    labels = np.asarray(labels, dtype=np.float32)
    positive_weight = torch.tensor(float((labels == 0).sum() / labels.sum()), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    rng = np.random.default_rng(config["seed"])
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(bags))
        losses = []
        for start in range(0, len(order), config["batch_pairs"]):
            batch = order[start:start + config["batch_pairs"]]
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model([bags[i] for i in batch])
            loss = criterion(logits, torch.as_tensor(labels[batch], device=device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()) * len(batch))
        history.append(sum(losses) / len(bags))
    return model, history


@torch.inference_mode()
def score_bags(model, bags, batch_size, details=False):
    model.eval()
    result = []
    for start in range(0, len(bags), batch_size):
        batch = bags[start:start + batch_size]
        logits, sentences, weights = model(batch)
        sentences = sentences.cpu().tolist()
        weights = weights.cpu().tolist()
        first = 0
        for bag, logit in zip(batch, logits.cpu().tolist(), strict=True):
            if not math.isfinite(logit):
                raise ValueError("Non-finite role score")
            evidence = []
            if details:
                for i, w in enumerate(bag, first):
                    evidence.append({"start": w.seg.start, "end": w.seg.end, "text": w.seg.text,
                                     "attention": weights[i], "sentence_logit": sentences[i],
                                     "role_coverage": w.features[3], "role_link": bool(w.features[5])})
                evidence.sort(key=lambda x: -x["attention"])
            result.append((logit, evidence))
            first += len(bag)
    return result
