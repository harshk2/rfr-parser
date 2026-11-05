# rfq-parser

Rules-first parser that turns messy Hinglish buyer messages for packaging
(BOPP bags, woven sacks, FIBC, film, pouches, boxes, cartons) into structured fields.

    parse_rfq("bopp bags chahiye 40 hazaar, 65 gsm, Nagpur")
    # {'category': 'bopp_bag', 'quantity': 40000, 'unit': 'pcs',
    #  'gsm': 65, 'micron': None, 'location': 'Nagpur'}

## Run
    pip install -r requirements.txt     # no third-party dependencies
    python score.py my_tests.jsonl      # official scorer (add it to this folder)
    python eval_local.py my_tests.jsonl # my equivalent harness, if score.py is absent
    python parse.py "your message"      # shows the result plus a short trace

Requires Python 3.10+.

## Files
- `parse.py`: the parser (`parse_rfq`)
- `my_tests.jsonl`: 66 labelled test messages
- `extra_unseen_tests.jsonl`: 34 extra messages written before first run
- `eval_local.py`: local scoring script (per-field and exact-match)
- `NOTE.md`: approach, score, known failures, next steps

## Note on scoring
The official `score.py` was not included with the task, so results in
`NOTE.md` come from `eval_local.py`.