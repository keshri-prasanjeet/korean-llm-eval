# korean-llm-eval

A small, human-verified benchmark targeting a Korean-specific linguistic
phenomenon, with evaluation code for large language models.

> Work in progress. The specific phenomenon is still being chosen; this README
> will be updated with the research question, dataset description, and
> citation once the preprint is available.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Dataset (`raw/` drafts, `final/` verified JSONL) |
| `eval/` | Evaluation code: model adapters, prompt templates, metrics, CLI |
| `results/` | Raw model outputs and aggregated tables |
| `analysis/` | Figures and error analysis |
| `paper/` | LaTeX source (ACL template) |
| `notes/` | Research log, literature notes, proposal, design decisions |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
cp .env.example .env   # then fill in the API keys you need
```

## License

- **Code** is released under the [MIT License](LICENSE).
- **Dataset** (everything under `data/`) is released under
  [CC BY 4.0](data/LICENSE).
