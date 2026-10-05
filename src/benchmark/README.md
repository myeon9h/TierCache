# SQStream generation

SQL and Cypher workloads use one set of commands in `scripts/`. Language-specific template extraction and DB sampling live in `backends/sql/` and `backends/cypher/`; paths, workload names, distribution sampling and JSON I/O are shared. This package does not load TierCache models or require a GPU.

Use the TierCache environment from Section 1, or install only generation dependencies in a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/benchmark/requirements.txt
```

## Inputs

Prepare target DBs using Section 1 of the repository README. Benchmark question/query inputs and schema metadata are included under `data/source/`, separate from the processed encoder/filler training archive. Database files and dumps are not included in this directory.

| Source | Default raw input | Required record fields |
| --- | --- | --- |
| EHRSQL | `src/benchmark/data/source/EHRSQL/{db}/valid.json` | `question`, `query` |
| ScienceBenchmark | `src/benchmark/data/source/ScienceBenchmark/{db}/dev.json` | `question`, `query`, optional `db_id` |
| BIRD | `src/benchmark/data/source/BIRD/{db}/dev.json` | `question`, `SQL` or `query`, optional `evidence` |
| CypherBench | `src/benchmark/data/source/CypherBench/raw/test.json` | `graph` or `db`, `nl_question` or `question`, `gold_cypher` or `cypher` |

The bundled files preserve the inputs from the SQL and Cypher generation repositories. See [source provenance and licenses](data/source/README.md). EHRSQL DB access is described separately in TierCache's EHRSQL setup section.

SQL source config `schema_root` points to the bundled per-DB `tables.json` metadata. DB connections in `db_config.yaml` still point to the databases prepared in Section 1.

The default CypherBench config uses the bundled preprocessed records (`masked_cypher`, `masked_question`, `literal_slots`, and sampling queries), matching the original CypherStream config. An existing preprocessed path takes priority over `raw`; remove the `preprocessed` mapping to rebuild masking and sampling metadata from the raw input without contacting Neo4j.

## Configure and check

From the TierCache repository root:

```bash
cp src/benchmark/configs/benchmark_sources.example.yaml src/benchmark/configs/benchmark_sources.yaml
cp src/benchmark/configs/db_config.example.yaml src/benchmark/configs/db_config.yaml
```

Edit these copies and retain the sources/DBs you need. All relative paths resolve from the TierCache root, even when scripts are invoked from another directory. Personal config copies and downloaded/generated data are ignored by Git.

`db_config.yaml` uses one mapping keyed by the original database name. SQL entries specify `type: sqlite` with `path`, or `type: postgresql` with `url` and optional `schema`. Cypher entries specify `type: neo4j`, `uri`, `username`, `password`. Cypher reads also accept `host` plus `port`. Use the same target database service for generation and evaluation; the example ports are not automatically provisioned.

```bash
python src/benchmark/scripts/check_setup.py --benchmark BIRD --check-connections
python src/benchmark/scripts/check_setup.py --benchmark CypherBench --check-connections
```

Without `--benchmark`, all configured sources are checked. Default split is Dev for SQL and Test for Cypher. `--skip-db-check` checks only source paths; `--check-connections` additionally performs a read-only `SELECT 1` or `RETURN 1`.

## Build pools

```bash
python src/benchmark/scripts/build_structural_pairs.py --benchmark BIRD --split Dev
python src/benchmark/scripts/build_structural_pairs.py --benchmark CypherBench --split Test
```

Replace BIRD with EHRSQL or ScienceBenchmark for their SQL sources. The default pool root is `data/benchmark/structural_pairs_pool/`, containing `template_pool/{Source}.json` and `literals/{Source}.json`. Existing other splits and DB entries are retained. Train/Dev SQL splits must be configured under `splits`. Cypher supports Train/Dev/Test keys in `raw` or `preprocessed`; `dbs` may be a flat list or a train/test mapping. `--input` can override the Cypher source path. `--bird-split` remains an alias for BIRD's `--split`.

## Generate workloads

```bash
python src/benchmark/scripts/generate_workload.py \
  --benchmark_type BIRD --target_db STUDENT --num_queries 1000

python src/benchmark/scripts/generate_workload.py \
  --benchmark_type CypherBench --target_db MOVIE --num_queries 1000
```

Default outputs are `data/SQStream/BIRD-STUDENT.json` and `data/SQStream/CypherBench-MOVIE.json`. Paper names and original source DB names are both accepted for `--target_db`:

| Source | Workload → database |
| --- | --- |
| EHRSQL | EICU → eicu; MIMIC → mimic_iii |
| ScienceBenchmark | CORDIS → cordis; ONCOMX → oncomx |
| BIRD | CODE → codebase_community; STUDENT → student_club |
| CypherBench | COMPANY → company; ACCIDENT → flight_accident; MOVIE → movie; NBA → nba |

For other DBs, supply their original name and an explicit `--output_file`. Existing files are not overwritten. Use `--pool-root`, `--sources-config` (building) and `--db_config_file` (generation) to select custom paths.

Distribution options are `query_len` (question character lengths weighted by length^-s, default s=2), `rank` (template ranks weighted by rank^-alpha, default alpha=1), and `uniform`. Default seed is 42. DB literal sampling is the default; `--sampling_mode hybrid` additionally permits cached/example values, but still validates generated queries against the live target DB. SQL validation requires at least one row; Cypher validation accepts valid empty results, matching the respective original pipelines. SQL connections are read-only; Neo4j sessions request read access.

Cypher defaults: `--sample_pool_size 100`, `--connection_timeout 2`, `--query_timeout 10`. It first samples correlated literals using the structural sampling query, then tries individual slots. Sampling preserves the original CypherStream behavior and string escaping. SQL uses the original column sampler and SQL template pipeline.

The output envelope is `{"config": {...}, "queries": [...]}`. Gold queries use `sql` or `cypher` according to language; statistics are printed at generation time. Generation fails without saving a partial file when it cannot produce the requested query count. DB sampling is not guaranteed to reproduce the released files byte-for-byte, even with the same seed; download the released SQStream files for the published evaluation inputs.
