# Golden set — held-out evaluation for the chat agent

`golden.jsonl` is **measurement**, not corpus. It never enters `data/index.json`.
Editing it changes only what the score says.

Contrast `../knowledge/*.md`, which IS corpus: `build_index` embeds those
sections into the index, so editing them changes what the agent knows and how it
answers. `about_zh.md` additionally *is* the Chinese gate matrix.

## Format

One JSON object per line. Adding a case is a one-line diff.

| field | required | meaning |
|---|---|---|
| `id` | yes | unique, `{role-short}-{lang}-{nn}`; role-short ∈ `client`, `aiagent`, `combat`, `visitor` |
| `role` | yes | a role id present in `../data/roles.json` |
| `lang` | yes | `en` or `zh` |
| `type` | yes | `positive`, `off_topic`, or `injection` |
| `q` | yes | the question, as a visitor would type it |
| `expected_urls` | positives only | site-relative page paths; a hit is a non-empty intersection with the top-4 |
| `expected_keywords` | positives only | 1–4 terms that must appear in the retrieved text; scored as coverage |
| `note` | no | why this case exists |

Each (role, lang) cell holds exactly 12 `positive`, 4 `off_topic`, 4 `injection`.

`expected_urls` names **pages, never chunk ids**. Ids like
`pages/prime-engine.html#sec2:en:3` move whenever the page is edited or
`chunk_size` changes; a page URL survives re-chunking and still catches real
retrieval failures. Page URLs are also language-agnostic, which is what lets one
field serve both languages.

`expected_keywords` adds the precision `expected_urls` gives up. A page has up
to 31 chunks and landing on any of them scores a hit, even when the chunk
carrying the answer never surfaced. Keywords are matched against the
concatenated text of the post-floor top-4 — exactly the chunks that become
`contexts` — so they answer "was the fact available to the model?".

Read the two together:

| `hit@4` | keywords | reading |
|---|---|---|
| pass | pass | healthy |
| pass | fail | right page, wrong chunk — a **chunking** problem, not retrieval |
| fail | pass | the fact lives somewhere unexpected; check `expected_urls` |
| fail | fail | genuine retrieval miss |

Matching is case-insensitive. Word boundaries apply only where the keyword's own
edge is ASCII alphanumeric, so `AI` will not match inside "available" while
`C++`, `3C` and `UE5` still match. CJK keywords match as plain substrings.

## Authoring rules

1. **Disjoint from fit-on data.** Never reuse a string from
   `gate_calibration.ON_TOPIC/OFF_TOPIC/ON_TOPIC_ZH/OFF_TOPIC_ZH`,
   `tests/test_gate.py::OFF_TOPIC`, or any `roles.json` starter. Those set the
   gate threshold or are already asserted — scoring against them measures
   nothing. `test_golden.py` enforces this.
2. **Chinese cases are authored natively**, never translated, and never lifted
   from an `about_zh.md` heading. Those 19 headings are the zh gate matrix; a
   question copied from one scores ~1.0 by construction.
3. **Visitor vocabulary, not site vocabulary.** "What's his shipped title count"
   beats "tell me about Cemented Dreams" — the latter is retrieval on easy mode.
4. **Keywords are proper nouns and technical terms, not prose.** `Blueprint`,
   `UE5`, `frustum culling`, `Hive` — never `mobility and responsiveness`. This
   is the one field coupled to the site's wording: rewrite a page and prose
   keywords break for reasons that are not regressions, while the name of a
   technology survives. Avoid Latin keywords shorter than three characters.
5. **Never edit a case to make a regression pass.** A failing case is a work
   queue item. Record reality in the baseline.

## Running

```bash
python scripts/run_eval.py                      # the table
python scripts/run_eval.py --role visitor --lang zh --verbose
python scripts/run_eval.py --update-baseline    # after a deliberate change
```

`tests/test_golden.py` fails only when a metric drops below
`../data/eval_baseline.json` — an absolute threshold would be meaningless,
because `hit@4` measures this file's difficulty as much as the system's quality.
Keyword coverage is compared as a ratio, not a raw count, so editing a case's
keyword list does not read as a regression.

## Known limits

- The **Chinese gate** is calibrated on 19 hand-written headings with
  essentially zero margin (threshold 0.4919 against an on-topic floor of 0.492).
  Held-out Chinese positives are expected to fail it at a real rate. That is the
  finding, not a defect in the cases.
- `data/gate_vectors.json` is gitignored, so a machine without it has no zh gate
  at all and the Chinese gate columns report `n/a`. Chinese `hit@4` still runs —
  retrieval does not depend on the gate.
- Only the gate and retrieval are scored. Answer quality and role emphasis need
  the LLM and are deliberately out of scope; the `role` field is carried on
  every case so that tier can be added without re-authoring.
