# Which office does the passage support deprecating?

This experiment compares two decisions about the same person in the same passage. A text baseline decides whether to emit a deprecation and selects one old triple. If the selected relation is P39 (position held) and several of that person's offices appear among the frozen candidates, a second model can choose the office. Every method has the same write opportunities.

## Reproduce

From the root of this fork, install the experiment dependencies in a Python 3.10 or 3.11 environment. The reference run used Python 3.10, PyTorch 2.1.1 with CUDA 12.1 and Transformers 4.35.2. A CPU can run both released models.

~~~sh
pip install -r experiments/role_selection/requirements.txt
~~~

Run both models and the paired evaluation with one command.

~~~sh
python experiments/role_selection/run.py
~~~

The first run reuses complete official inputs in the repository's data directory. On a cold clone it downloads only the pinned 2020 and 2022 evaluation shards and their compressed old graphs into output/role_selection/public-data, plus the BERT tokenizer/config into a model cache. A cold run transfers about 1.1 GB. Retrieval streams the graphs and caches the verified candidate rows. The default run executes both trained scorers, then checks a digest of every chosen triple and abstention against the original experiment.

Use the following options for an existing data installation. The data directory contains evaluation_set/ and kg_snapshots/. The encoder directory contains Google's config.json and vocab.txt. The device defaults to CUDA when available.

~~~sh
python experiments/role_selection/run.py --data-dir PATH_TO_EMERGE_DATA --encoder-dir PATH_TO_BERT_TINY --offline --device cpu
~~~

## Results

| Cohort | Method | Writes | Correct / executable deprecations | Known P39 errors / retrieved P39 keeps | Unlabeled writes |
| --- | --- | ---: | ---: | ---: | ---: |
| 2020 development, 488 passages in 349 articles | Text | 379 | 42/224 | 15/89 | 296 |
| 2020 development | Text + P39 reranker | 379 | 45/224 | 12/89 | 296 |
| 2022 independent, 496 passages in 367 articles | Text | 427 | 29/209 | 4/18 | 359 |
| 2022 independent | Text + P39 reranker | 427 | 33/209 | 2/18 | 357 |

The recall gain was 1.34 percentage points in 2020 (article-paired 95% interval 0.00 to 3.72) and 1.91 in 2022 (0.00 to 4.47). A known wrong office became a confirmed deprecation in three passages from two articles in 2020 and two passages from two articles in 2022. Two further 2022 hits were previously unlabeled outputs in passages about the same office held by David Sassoli. Of the 24 eligible 2022 passages, three baseline selections were known keeps and 21 were unlabeled. This cohort had zero confirmed correct incumbent choices at risk of replacement. The reported rates use this fixed candidate and one-write protocol. The upstream benchmark has its own full-test evaluation protocol. [All 984 paired decisions](reference/decisions.jsonl) · [Change counts and article bootstrap intervals](reference/metrics.json)

The run writes to output/role_selection/paired/. The decisions file includes both choices, evaluation labels and the scored sentence windows. The changes file contains just the substitutions. The metrics file gives a full keep/deprecate/unlabeled transition table for substitutions and 10,000 paired article-bootstrap draws summarized as percentile intervals. A missing operation label stays unlabeled.

## Evaluate your own role choices

Run the text model alone to generate the fixed opportunities.

~~~sh
python experiments/role_selection/run.py --text-only
~~~

Read output/role_selection/text/opportunities.jsonl. There are 19 eligible passages in 2020 and 24 in 2022. Each row contains the year, passage ID, article, original passage, hyperlinks, incumbent triple and candidate offices in frozen order. The file exposes the inputs available at decision time. Return exactly one candidate triple per row in a JSONL file. Returning the incumbent keeps the baseline choice. For example, this is the submission format for one released opportunity.

~~~json
{"year":2022,"id":"WlBFgtzLss","triple":["Q7351526","P39","Q7925068"]}
~~~

~~~sh
python experiments/role_selection/run.py --choices my_choices.jsonl
~~~

The evaluator checks that every eligible passage has exactly one choice from its candidate list. It fills the other write slots directly from the baseline and saves the comparison under output/role_selection/external/. The included BERT run produces a complete example in output/role_selection/paired/choices.jsonl.

## Data, supervision and model

- The cohort file specifies each passage ID, a zero-based index into that year's sorted source shards and a one-based source line. Retrieval starts from the original hyperlinks and article entity, considers their old-snapshot outgoing edges and keeps at most 32. Executable gold and retrieved candidates are computed separately. Raw shards, old graphs, reconstructed candidate lists, the released parameters and original decisions have SHA-256 fingerprints. [Cohort membership](reference/cohorts.json) · [Release manifest](reference/release.json)
- The text model uses TF–IDF passage and triple features, numeric retrieval features and class-balanced logistic regression. Its 2019 training set contains 712 passages. The 2020 text policy uses five article-held-out calibrations for P39 and other relations. The independent policy uses the thresholds fitted on all of 2020.
- BERT Tiny matches each person–office query with up to eight original sentence windows. Joint attention from the two query entities pools passage tokens. Hyperlinks, role mentions and nearby years relative to the old snapshot support sentence selection and scoring. The challenger must beat the incumbent by more than ln(1.5) to change the office.
- The reranker used 46 explicitly deprecated and 114 explicitly kept P39 examples from 2019. Unlabeled candidates do not enter its classification loss. Six insufficient-evidence judgments had no independent operation label and were excluded from that loss. The original four-fold, article-grouped 2019 selection compared epochs 2, 4, 6 and 8, with mean average precision 0.5174, 0.4938, 0.5221 and 0.5054. The released refit uses six epochs and seed 73. [Configuration](config.json) · [Reranker](role_model.py)
- The model and policy were locked before opening 2022. Four of the 500 official passages were excluded by article overlap with prior cohorts. The released 2022 results document that original independent test. The intervals condition on the fitted model and resample articles.

To refit both estimators at the selected settings from the pinned 2019 corpus, old graph and original BERT weights, run the following commands. The second command scores the refitted models.

~~~sh
python experiments/role_selection/fit.py
python experiments/role_selection/run.py --weights-dir output/role_selection/refit --output output/role_selection/refitted-evaluation
~~~

## Upstream prediction format

Each run writes an upstream/predictions.jsonl containing canonical triple IDs, labels and the DEPRECATE operation tag, including empty predictions for abstentions. Use the generated merge configuration with EMERGE's existing command from the repository root.

~~~sh
python src/merge/s03_merge_predictions.py --config_file output/role_selection/paired/merge_config.json
~~~

The fork's evaluator routes the operation-tagged schema through the existing operation-aware prediction handler. The canonical names are role-selection/text and role-selection/bert. Rows outside the two cohorts receive no predictions from these models. [Upstream benchmarking interface](../../src/benchmarks/README.md)

## Attribution

The passages, hyperlinks, operation judgments and snapshots come from Klim Zaporojets and colleagues' EMERGE release. Cohort membership and result extracts in reference/ are derived from that release and shared under its CC BY-SA 4.0 terms. BERT initialization and vocabulary come from Google's two-layer, 128-hidden-unit uncased BERT. This fork supplies the trained deprecation scorers, evaluation and reproduction code. [EMERGE data](https://huggingface.co/datasets/klimzaporojets/emerge-benchmark/tree/6733e1cc7e3c2a882fc9464fa4bfc002eecd1bd1) · [EMERGE data license](../../LICENSE-DATA) · [Google BERT Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f) · [ATLOP localized context pooling](https://arxiv.org/abs/2010.11304)
