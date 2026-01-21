# dato-tune

Estimates difficulty of Lantern Questions using Item Response Theory.

## Setup

1. Make sure you have installed:

- [uv](https://docs.astral.sh/uv/)
- [SnowSQL](https://docs.snowflake.com/en/user-guide/snowsql-install-config.html)

2. Install dependencies using uv:
```bash
uv sync
```

3. Create configuration file:
```bash
cp config.ini.example config.ini
```

## Usage

### Fetch Data from Snowflake

Fetch student response data for a specific curriculum and date range:

```bash
uv run item_estimation/main.py fetch \
  --curriculum-id 15 \
  --outfile lantern_responses.csv \
  --begin-date 2025-10-01 \
  --end-date 2025-12-31
```

**Note:** A web browser will open automatically for Snowflake authentication. You'll need appropriate Snowflake access permissions.

### Run Item Difficulty Estimation

Estimate item difficulties using the fetched data:

```bash
uv run item_estimation/main.py infer \
  --curriculum-id 15 \
  --infile lantern_responses.csv \
  --outfile difficulties.csv
```
Results are saved to the specified output file, and further results are saved
to `result_folder` as defined in `config.ini`.
