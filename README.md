# Korean speech-level evaluation

Benchmark data, saved model responses, and the Python code for calling model APIs
in multi-turn Korean conversations.

## Contents

- `data/dataset.json`: 250 conversations, 50 per condition, with 3,500 response
  slots, Korean prompts, and expected response properties.
- `results/*.jsonl`: 17,500 saved responses and their recorded overall-turn
  verdicts, joined by `dialogue_id` and `turn_id`.
- `eval/`: conversation runner, dataset validation, and provider API adapters.
- `configs/`: model identifiers and generation parameters used for the results.

The conditions are MAINTAIN, SWITCH-EXPLICIT, SWITCH-RELATIONAL, IMPLICIT, and
REFERENT. Conversations contain 6, 12, or 20 response slots. Speech-level codes
are HAPSYO (하십시오체), HAEYO (해요체), HAE (해체), and HAERA (해라체).

| Model | Provider | Responses |
|---|---|---:|
| `claude-sonnet-5-5` | anthropic | 3,500 |
| `gemini-3.8-flash` | google | 3,500 |
| `kimi-k3.1-pro` | kimi | 3,500 |
| `gpt-6.1-sol` | openai | 3,500 |
| `gpt-6.1-terra` | openai | 3,500 |

## Data format

The dataset contains a `conversations` array. Each conversation supplies a
`system_prompt`, ordered `turns`, and the context needed to interpret its expected
properties. Each turn supplies a `user` message and `expected` criteria. Expected
criteria and other annotations are never sent to the model. Each conversation
starts with fresh history; later turns include that model's actual earlier replies.

Result records preserve the original response text in `raw_response`, its UTF-8
SHA-256 in `response_sha256`, the model's finish reason, and whether an output limit
was reached. `verdict` is the final PASS/FAIL overall-turn judgment. Every verdict was
verified by three human reviewers (two native Korean speakers and the author),
and the final decision is their consensus, so `human_reviewed` is true for all
17,500 records.
All 3,500 response slots per model are included, including empty or limited outputs.

The published dataset preserves the final prompts and expected properties.
The provider adapters and conversation-history logic are retained; dataset loading
uses this publication's standalone schema.

## Use the API runner

Python 3.11+ on macOS or Linux is required. The runner uses the Python standard
library and can be run directly from the repository root.

```bash
python -m eval.dataset
cp .env.example .env
```

Fill the relevant provider key in `.env`, or set the corresponding environment
variable. Inspect a configuration without sending requests:

```bash
python -m eval --config configs/gpt-6.1-terra.json --dry-run
```

Start with a one-request cap:

```bash
python -m eval --config configs/gpt-6.1-terra.json --max-requests 1
```

Resume the same output directory to complete the benchmark:

```bash
python -m eval --config configs/gpt-6.1-terra.json --resume --full-run
```

Select another model by changing the configuration filename. Credentials require
access to the configured model. `--output` chooses a new output directory.
`--max-dialogues` selects a prefix of the conversations. Empty or limited outputs
pause execution; `--continue-after-incomplete` permits continuing with those saved
outputs after inspection. New response files are written under `outputs/` and
remain local. The repository uses an explicit file allowlist for Git tracking.

## License

Code: [MIT](LICENSE). Benchmark data: [CC BY 4.0](data/LICENSE).
