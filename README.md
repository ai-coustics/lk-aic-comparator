# lk-aic-comparator

Compare output from:

1. direct `aic-sdk` processing
2. `livekit-plugins-ai-coustics` (built locally from `plugins-ai-coustics-internal`)

## 1) Build local ai-coustics plugin package

```bash
./scripts/build_local_ai_coustics_package.sh
```

This creates:

`~/dev/projects/plugins-ai-coustics-internal/target/packages/python`

## 2) Override dependency in noise-canceller-fork

`~/dev/projects/noise-canceller-fork/pyproject.toml` should contain:

```toml
[tool.uv.sources]
livekit-plugins-ai-coustics = { path = "../plugins-ai-coustics-internal/target/packages/python", editable = true }
```

Then sync:

```bash
cd ~/dev/projects/noise-canceller-fork
uv sync
```

## 3) Run comparator

Create project environment:

```bash
uv sync
uv run lk-aic-compare --help
```

Example:

```bash
AIC_SDK_LICENSE=... \
uv run lk-aic-compare ~/dev/projects/noise-canceller-fork/noisy-sample.wav \
  --model quail_l --write-outputs
```

If `mismatching_samples` is `0` (or within your tolerance), both processors match.

Compatibility entrypoint still works:

```bash
uv run python compare_processors.py ./gym_raw.wav --model quail_l
```
