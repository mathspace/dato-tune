# dato-tune

Estimates difficulty of Lantern Questions using Maximum Likelihood Estimation.

## Setup

1. Make sure you have installed:

- [uv](https://docs.astral.sh/uv/) -- Must be at least version 0.9
- [SnowSQL](https://docs.snowflake.com/en/user-guide/snowsql-install-config.html)

2. Configure SnowSQL with your username in `~/.snowsql/config`:
```
[connections]
username = "<your_username>"
```

3. Install dependencies using uv:
```bash
uv sync
```

4. Create configuration file:
```bash
cp config.ini.example config.ini
```

## Usage

### Fetch Data from Snowflake

Fetch student response data for a specific curriculum. You can use either date range mode or windowed mode.

#### Date Range Mode

Fetch data for a specific date range:

```bash
uv run item_estimation/main.py fetch \
  --region us \
  --curriculum-id 15 \
  --outfile lantern_responses.csv \
  --begin-date 2025-10-01 \
  --end-date 2025-12-31
```

#### Windowed Mode

We want to maintain data localised in time. Student ability is expected to change over time, so we can't expect a single student's data ranging over long period of time to reliably estimate their abilty. However, we don't want to ignore a significant amount of usable data by only considering a small period per student.

We use a 'windowed' approach where each students activity is chunked into periods, and for the purposes of estimation each student-window is a unique agent with distinct topic-abilities.

Fetch data using sliding x-month windows with x-month stride (accepts /\dm/ or /\dy/ as window-size arg):

```bash
uv run item_estimation/main.py fetch \
  --region us \
  --curriculum-id 15 \
  --outfile lantern_responses.csv \
  --window-size 12m
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
