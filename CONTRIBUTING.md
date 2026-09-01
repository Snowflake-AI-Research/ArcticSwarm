# Contributing to ArcticSwarm

Thanks for your interest in improving ArcticSwarm! This guide covers how to set up
a dev environment, run the checks, and submit changes.

## Development setup

ArcticSwarm targets **Python 3.11+** (validated on 3.12) and needs **Java 11+** only
for PDF reading. See [ENVIRONMENT.md](ENVIRONMENT.md) for the full setup (credentials,
optional caches).

```bash
git clone <your-fork-url> && cd arcticswarm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest + ruff
```

Then copy the settings template and add any keys you have:

```bash
cp config_files.template.json config_files.json   # git-ignored; never commit real keys
```

## Before you open a PR

```bash
pytest                 # run the test suite (offline; no API keys needed)
ruff check .           # lint
ruff format .          # autoformat (optional)
```

- **Keep tests green.** Add tests for behavior changes; the suite lives under
  `arcticswarm/**/tests/`.
- **Match the surrounding style** — comment density, naming, and idioms. The codebase
  favors small, focused functions and explicit config over magic.
- **Don't commit secrets, datasets, caches, or run outputs.** These are git-ignored
  (`config_files.json`, `arcticswarm/eval/data/*.csv`, `cache/`, `results/`). Datasets
  are regenerated locally — see [DATASETS.md](DATASETS.md).

## Scope notes

- New benchmarks/datasets: add a loader under `arcticswarm/eval/data/external/` and a
  subset spec under `arcticswarm/eval/data/subset_specs/`; never commit question/answer
  text. See the **Custom evaluation** section of the README for the unified-CSV contract.
- New retrieval backends: implement against the `web.corpus_backend` interface (see
  `arcticswarm/tools/corpus_retriever.py`).

## Reporting bugs / requesting features

Open an issue using the templates in `.github/ISSUE_TEMPLATE/`.

## License

By contributing, you agree that your contributions will be licensed under the
project's [Apache-2.0 License](LICENSE).
