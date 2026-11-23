# Contributing

MachineEar is a small offline audit. Changes should keep the default path free of network access, paid services, and private recordings.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
python -m ruff check src tests scripts
machineear demo --output demo-out
```

The showcase report must keep the counts in `src/machineear/fixtures/oracle.json`. If you change the generator, run `python scripts/generate_fixtures.py`, review the new counts, and update the oracle and `docs/oracle.md` in the same change.

## Fixtures

Default audio is generated in this repository. Do not add workplace recordings, speech, or datasets whose terms are non-commercial or unclear. An optional benchmark import, if one is added later, must be off by default and must record the license before any file is committed.

## Secrets

Do not commit tokens, keys, or `.env` files. The CI workflow has no secrets.
