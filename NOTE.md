# NOTE: RFQ parser

**Approach.** A rules-first hybrid, standard library only (no model, ~1 ms per message). One pipeline, one small extractor per field, so every wrong answer traces to one rule:
normalise (Unicode, case, noise, `1,50,000`, `20k`, Devanagari) → typo-fix (explicit map + conservative `difflib` match) → Indian number words (`do hazaar`, `1 lakh 20 hazaar`) → GSM/micron (unit-bound regexes; masks those numbers) → category → quantity → location.
- **Category:** 59 weighted phrase rules summed per class (e.g. `bopp … bag` 12, `ad star` 12, `jumbo` 9, `bopp film` beats lone `bopp`), not one keyword.
- **Quantity:** every number becomes a candidate (explicit `pcs/bags/rolls` > number next to a product noun > bare number; weight units only as a fallback). Number+kg next to bag/sack/pouch, `capacity`, `SWL`, `x`, `each`, or a small kg beside another figure is bag capacity, never the order.
- **Location:** 670-name gazetteer (cities, states, countries) + typo matching + grammar cues (`deliver to X`, `X mein`, `location: X`) so unseen towns still work; `from X` is penalised.

**Score.** `my_tests.jsonl`: 66/66 exact match (all six fields). Caveat: I wrote those alongside the parser, so this proves little. `extra_unseen_tests.jsonl` (34 messages written *before* the first run) scored 33/34 first time. About 50 further ad-hoc messages exposed 5 more bugs, all fixed (e.g. `1 lakh 20 hazaar`, `Ramgarh` fuzzy-matched to `Raigarh`). The official `score.py` was not in my files, so I used my own `eval_local.py`.

**Three messages it still gets wrong**
1. `bopp bags 5000 pcs 70 gsm kolar` → location `None`. An unlisted town with no cue word and no capital letter is indistinguishable from noise; I chose a missed place over invented ones.
2. `please share price of 1 tonne jumbo bag` → quantity `None`. "1 tonne" is bag capacity, and no order quantity exists. The spec gives no value for a missing quantity, so I return `None`.
3. `need shrink film. 2 rolls of 20 kg each, 25 mic` → `2 pcs`; the buyer may mean 40 kg. The rule prefers explicit counts over weights.

**With 1,000 labelled examples.** Hold out 200 and measure per field, then tune weights and the typo map from the errors. Add a small char n-gram classifier for category (rule scores as features) for low-margin cases, and a token tagger for place names in place of the cue heuristics. Rules stay as the first pass: explainable, and they can't invent a number.

**In production.** Log every parse with a per-field confidence (category margin, quantity runner-up score). Sample 50 messages a week for human labelling and track per-field accuracy. Track the share of `None` locations and the rate of buyers editing the parsed fields. Alert when `quantity is None` or the category margin is small, and route those to a human or a fallback.
