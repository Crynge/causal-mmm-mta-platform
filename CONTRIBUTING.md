# Contributing

Thanks for contributing to `causal-mmm-mta-platform`.

## Recommended workflow

1. Fork the repository.
2. Create a focused branch.
3. Add or update tests for the behavior you change.
4. Run:

```bash
pip install -r requirements-core.txt
python -m pytest tests -q
```

5. Open a pull request with a clear verification note.

## High-value contributions

- MMM numerical stability improvements
- clearer optional dependency boundaries
- attribution method validation
- causal discovery benchmarks
- better docs and example notebooks

## Standards

- Prefer deterministic tests.
- Keep research claims aligned with what the code actually implements.
- Document heavyweight optional dependencies instead of assuming they exist.
