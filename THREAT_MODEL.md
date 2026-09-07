# dato-tune threat model

## Overview

Public Python analytics CLI fetches student-response records through SnowSQL external-browser authentication or consumes CSV/stdin, computes local item/mastery estimates and writes CSV/plots. It does not call an LLM or publish results back to a production application (item_estimation/main.py:47; item_estimation/main.py:64; item_estimation/fetch.py:132; item_estimation/run_inference.py:345).

| Component / resource | Source |
| --- | --- |
| Snowflake fetch | item_estimation/fetch.py:122; item_estimation/fetch.py:130; item_estimation/fetch.py:132; item_estimation/main.py:61 |
| Inference outputs | config.ini.example:3; config.ini.example:6; item_estimation/main.py:146; item_estimation/run_inference.py:225; item_estimation/run_inference.py:231; item_estimation/run_inference.py:348 |
| Pickle helper | config.ini.example:2; item_estimation/load_data.py:49; item_estimation/load_data.py:230 |

| Deployment or workflow | Resource or capability | Configuration and precedence | Safe effective value or location | Readers, writers, or recipients | Enforcing control | Evidence or unknowns |
| --- | --- | --- | --- | --- | --- | --- |
| CLI fetch | Snowflake fetch | --region -> account mapping -> snowsql PATH executable + externalbrowser | US oua13326; AU pn30490.ap-southeast-2; warehouse reporting/database data_science/schema public | Snowflake and authenticated local operator; temporary CSV file then selected outfile/stdout | External-browser authentication; Snowflake configured role; subprocess argument vector | item_estimation/fetch.py:122; item_estimation/fetch.py:130; item_estimation/fetch.py:132; item_estimation/main.py:61 |
| CLI infer default example config | Inference outputs | cwd/config.ini interpolation + --outfile-suffix default - -> output functions | ./result/-/estimated_mastery.csv and estimated_item.csv plus inference plots; separately ./difficulties_-.csv; ./run.log | Local operator/filesystem/log readers | Host permissions; no repository encryption/access layer | config.ini.example:3; config.ini.example:6; item_estimation/main.py:146; item_estimation/run_inference.py:225; item_estimation/run_inference.py:231; item_estimation/run_inference.py:348 |
| conditional direct DataLoader use | Pickle helper | common data_folder -> knowledge_graph -> pickle.load | Default ./data/skill_topics.p | Python process receives deserialized object | Requires trusted pickle provenance; not established as current CLI call | config.ini.example:2; item_estimation/load_data.py:49; item_estimation/load_data.py:230 |

## Threat Model, Trust Boundaries, and Assumptions

### Protected assets

- Source-defined student IDs, answers/results and timestamps; calibrated item/mastery integrity; operator Snowflake authority and exported files (item_estimation/fetch.py:25; item_estimation/run_inference.py:231).

### Security objectives

- Protect response-level exports/mastery data with operator filesystem and Snowflake controls; preserve region/query selection and output provenance.
- Do not treat all data as aggregate/anonymized merely because the result includes item statistics; student mastery CSV preserves identifiers.

### Actors and capabilities

- Input CSV author can affect statistical output/resource use; operator controls config, PATH and output suffix. Untrusted pickle writer matters only when the helper is actually used. No source-backed anonymous network listener.

### Trust boundaries

- CLI region chooses one of two hardcoded Snowflake accounts; snowsql executable from PATH receives a constructed argument vector, externalbrowser authentication, reporting warehouse/data_science/public and query. No shell=True path; Snowflake role privileges remain externally configured (item_estimation/fetch.py:122; item_estimation/fetch.py:132; item_estimation/fetch.py:161).
- CSV reads enter pandas and local inference; config.ini in cwd supplies interpolated common/inference settings and output locations. Trusted configuration and chosen paths have local operator filesystem authority (item_estimation/main.py:99; item_estimation/main.py:67; item_estimation/run_inference.py:348).
- Outputs are not all under configured result root: mastery/item CSV and plots use result_folder/suffix, but difficulties_{suffix}.csv is written in cwd. Fetch defaults stdout and temporary SnowSQL CSV is host-temp (item_estimation/main.py:114; item_estimation/fetch.py:130; item_estimation/run_inference.py:225; item_estimation/run_inference.py:231).
- A DataLoader knowledge_graph helper unpickles data_folder/skill_topics.p; no call to that helper is established from current main infer path. Treat as conditional library use, not reachable CLI deserialization by assertion (item_estimation/load_data.py:49; item_estimation/load_data.py:228; item_estimation/main.py:64).

### Assumptions and open questions

- Public repository model uses only its own source and generic caller duties; no private organization policy is imported.
- Default config result root ./result is not an exclusive export boundary, and infer --outfile is accepted but not consumed in main branch (config.ini.example:3; item_estimation/main.py:145; item_estimation/main.py:166).
- Runnable packaging/import correctness and actual Snowflake role/schema permissions are unverified; this architecture does not claim successful execution.
- Offline architecture mapping of the supplied revision; not completed vulnerability-audit coverage. No application execution or deployment verification.

## Attack Surface, Mitigations, and Attacker Stories

These are threat hypotheses, not validated vulnerabilities. Priority reflects plausible impact; deployment and attacker prerequisites must be established before assigning a finding severity.

| Priority | Scenario and capability gain | Prerequisites | Impact | Existing controls | Mitigation | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Operator-generated data extracts escape intended file/log access controls. | Other users/processes or publication tooling can read local exports/stdout; data contains student response/mastery records. | Disclosure of identifiable educational response data. | Snowflake authentication and OS file permissions; temporary CSV uses NamedTemporaryFile. | Protect all derived exports and stdout/log destinations, including cwd difficulties file outside result root. | item_estimation/fetch.py:25; item_estimation/fetch.py:130; item_estimation/run_inference.py:225; item_estimation/run_inference.py:231 |
| P2 | Adversarial or incorrect CSV silently changes calibration results. | Untrusted producer supplies inference input and operator later relies on results. | Integrity loss in local analytics; downstream product effect requires separate import/publication workflow. | Curriculum filter and preprocessing; configured observation/iteration limits. | Validate provenance, required fields and result plausibility before downstream use; maintain source/result linkage. | item_estimation/main.py:67; item_estimation/load_data.py:60; item_estimation/run_inference.py:345 |
| P2 | Executable resolution or trusted pickle helper turns file control into local process execution. | Attacker can replace snowsql on operator PATH, or a caller actually invokes knowledge_graph with attacker-written pickle. | Code runs with operator authority, potentially reaching local data/authentication context. | Fetch checks executable availability and uses argument vector; current main infer path does not call pickle helper. | Protect PATH/executable provenance; only load trusted pickles or use data-only serialization when helper is required. | item_estimation/fetch.py:125; item_estimation/fetch.py:161; item_estimation/load_data.py:49; item_estimation/load_data.py:230 |
| P3 | Configuration/output suffix sends files to unexpected locations or overwrites local outputs. | Operator or workflow provides unintended config/suffix; paths have OS write permission. | Local data loss or output confidentiality drift. | Explicit CLI/config inputs; result directory created under resolved path; no remote path parameter. | Treat config and suffix as trusted filesystem inputs; verify exact outputs, including default suffix -, before automation. | item_estimation/main.py:99; item_estimation/main.py:146; item_estimation/run_inference.py:225; item_estimation/run_inference.py:348 |

## Severity Calibration (Critical, High, Medium, Low)

| Level | Example | Counterexample or limiting prerequisite |
| --- | --- | --- |
| Critical | Requires demonstrated broad privileged compromise through an actual execution path and sufficiently powerful operator account. | Unused pickle helper and shell-free SnowSQL invocation do not establish anonymous remote code execution. |
| High | Proven unauthorized disclosure of sensitive student-response/mastery exports or local execution crossing a real lower-trust input boundary. | Attacker already owning the operator account has no new privilege gain from choosing paths. |
| Medium | Material calibration corruption, local output loss or excessive analytics resource use from plausible untrusted input. | A downstream production impact needs evidence of a result ingestion workflow absent here. |
| Low | Recoverable parse/config errors or misleading unused --outfile behavior without material loss. | The existence of local student IDs should not be described as anonymous aggregate data. |

Confidence in source-established architecture is separate from confidence in live deployment or exploitability. This model incorporates an independent source architecture pass and direct reconciliation of material consumers. No application execution or live service/security configuration validation was performed. Revisit the model when the documented entry points, data recipients, permissions or deployment paths change.

Repository: github.com/mathspace/dato-tune
Version: a73b97c684c97506049768f4bae56fb85f23e1a7
