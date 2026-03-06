# lk-aic-comparator

Compare output from:

1. direct `aic-sdk` processing
2. public `livekit-plugins-ai-coustics` from PyPI

## 1) Install project dependencies

```bash
uv sync
```

## 2) Set required environment variables

Create a `.env` from the template:

```bash
cp .env.example .env
```

Then fill in values in `.env`.

The CLI loads `.env` automatically (or use `--env-file <path>`).

Required keys:

- `AIC_SDK_LICENSE`
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

You can still export via shell instead of `.env` if preferred.

Example shell export:

```bash
export AIC_SDK_LICENSE=...
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=...
export LIVEKIT_API_SECRET=...
```

## 3) Run comparator

```bash
uv run lk-aic-compare --help
```

Example:

```bash
uv run lk-aic-compare ~/dev/projects/noise-canceller-fork/noisy-sample.wav \
  --model quail_l \
  --noise-canceller-path ~/dev/projects/noise-canceller \
  --write-outputs
```

If `mismatching_samples` is `0` (or within your tolerance), both processors match.
