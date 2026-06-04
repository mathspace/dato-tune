# dato-tune

Estimates difficulty of Lantern Questions using Maximum Likelihood Estimation.

## Setup

1. Make sure you have installed:

- [uv](https://docs.astral.sh/uv/) -- Must be at least version 0.9

2. Configure `[snowflake] username` in `config.ini`. The fetch path tries
`externalbrowser` first and falls back to `username_password_mfa` if browser auth
fails and `SNOWFLAKE_PASSWORD` is set in the environment.

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

Fetch student response data for a specific curriculum. There are two fetch subcommands: `fetch-lantern` and `fetch-mathspace`.

**Note:** A web browser will open automatically for Snowflake authentication. You'll need appropriate Snowflake access permissions.

#### Lantern — Date Range Mode

Fetch Lantern data for a specific date range:

```bash
uv run item_estimation/main.py fetch-lantern \
  --region us \
  --curriculum-id 15 \
  --outfile lantern_responses.csv \
  --begin-date 2025-10-01 \
  --end-date 2025-12-31
```

#### Lantern — Windowed Mode

We want to maintain data localised in time. Student ability is expected to change over time, so we can't expect a single student's data ranging over long period of time to reliably estimate their ability. However, we don't want to ignore a significant amount of usable data by only considering a small period per student.

We use a 'windowed' approach where each student's activity is chunked into periods, and for the purposes of estimation each student-window is a unique agent with distinct topic-abilities.

Fetch data using sliding x-month windows with x-month stride (accepts `<n>m` or `<n>y` as window-size):

```bash
uv run item_estimation/main.py fetch-lantern \
  --region us \
  --curriculum-id 15 \
  --outfile lantern_responses.csv \
  --window-size 12m
```

#### Mathspace — Windowed Mode

```bash
uv run item_estimation/main.py fetch-mathspace \
  --region us \
  --curriculum-id 15 \
  --outfile mathspace_responses.csv \
  --window-size 12m
```

### Run Item Difficulty Estimation

Estimate item difficulties from Lantern and/or Mathspace response files. At least one of `--lantern-infile` or `--mathspace-infile` must be provided; both can be supplied together and will be concatenated before inference.

```bash
uv run item_estimation/main.py infer \
  --curriculum-id 15 \
  --lantern-infile lantern_responses.csv \
  --mathspace-infile mathspace_responses.csv \
  --outfile-suffix my_run
```

Results are saved to `result_folder/<outfile-suffix>/` as defined in `config.ini`.

To deterministically sample students from the Mathspace data before inference, set
`[inference] student_sample_rate` in `config.ini` to a value between `0.0` and
`1.0`. A value of `1.0` leaves the dataset unchanged. Lantern data is never sampled.

### Run Propagation Parameter Estimation

Estimate propagation weighting parameters using response data:

```bash
uv run propagation_parameters_estimation/main.py estimate \
  --curriculum-id 15 \
  --infile lantern_responses.csv \
  --skill-links-infile skill_links.csv \
  --outfile propagation_parameters.csv
```
