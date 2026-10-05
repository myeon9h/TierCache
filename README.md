# TierCache: A Multi-Granularity Semantic Caching Framework for Structured Query Generation
Accepted at COLM 2026: [Paper](https://openreview.net/pdf/83a198968c97c1500828c9529794757cfa98e8e3.pdf)

![Overview of TierCache](TierCache.png)

## 1. Setup 

```bash
conda create -n tiercache python=3.9
conda activate tiercache
pip install -r requirements.txt
```

### Storage 

Install `SQLite` as Cache DB and `Milvus` as Vector DB for TierCache.

```bash
# SQLite
sudo apt update
sudo apt install sqlite3

# Milvus (requires Docker)
mkdir vectordb
wget https://github.com/milvus-io/milvus/releases/download/v2.5.10/milvus-standalone-docker-compose.yml \
     -O docker-compose.yml
sudo docker compose -f vectordb/docker-compose.yml up -d
```

### Models

<!-- HYUNJUN-REVIEW START: 2026-10-03 | Added Google Drive model download link. -->
Download the trained models for TierCache: [models.tar.gz](https://drive.google.com/file/d/1jfJVeHEi5rye_C9sJ40SvQK7u9JlNwjk/view). Then, extract the archive:

```bash
# Semantic encoder, Structure-aware encoder, Slot fillers
tar -zxvf models.tar.gz
```
<!-- HYUNJUN-REVIEW END: model download link -->
To train your own models, see '4. Lightweight Model Training' below.

### Databases

<!-- HYUNJUN-REVIEW START: 2026-10-03 | Modified database archive instructions; EHRSQL is excluded from the public archive. -->
Download the database assets: [databases.tar.gz](https://drive.google.com/file/d/1QrcmzoE-tr27QGJ32Y2bK5RppTUnMx-L/view). Then, extract the archive from the repository root:

```bash
# ScienceBenchmark, BIRD, CypherBench; EHRSQL is provided separately
tar -zxvf databases.tar.gz
```

The archive leaves `data/databases/EHRSQL/` empty. Prepare the EHRSQL databases separately as described below before running the EHRSQL examples or workloads.
<!-- HYUNJUN-REVIEW END: database archive instructions -->

<!-- HYUNJUN-REVIEW START: 2026-10-03 | Official EHRSQL DB download, placement and schema export instructions. -->
#### SQLite for EHRSQL

Follow the [official EHRSQL database instructions](https://github.com/glee4810/EHRSQL#database) to download the preprocessed `eicu.sqlite` and `mimic_iii.sqlite` files. Rename and place them at the following paths:

```text
data/databases/EHRSQL/eicu/db.sqlite
data/databases/EHRSQL/mimic_iii/db.sqlite
```

For example, run the following from the repository root, setting `ehrsql_download_dir` to the directory containing the downloaded files:

```bash
# Create directories for the EHRSQL databases.
mkdir -p ./data/databases/EHRSQL/eicu ./data/databases/EHRSQL/mimic_iii

# Set the download directory and copy each database as db.sqlite.
ehrsql_download_dir="/path/to/downloads"
cp "$ehrsql_download_dir/eicu.sqlite" ./data/databases/EHRSQL/eicu/db.sqlite
cp "$ehrsql_download_dir/mimic_iii.sqlite" ./data/databases/EHRSQL/mimic_iii/db.sqlite

# Export schema descriptions from the downloaded databases.
sqlite3 ./data/databases/EHRSQL/eicu/db.sqlite '.schema' \
  > ./data/databases/EHRSQL/eicu/schema.txt
sqlite3 ./data/databases/EHRSQL/mimic_iii/db.sqlite '.schema' \
  > ./data/databases/EHRSQL/mimic_iii/schema.txt
```

<!-- HYUNJUN-REVIEW END: EHRSQL setup instructions -->

#### PostgreSQL for ScienceBenchmark

```bash
# 1. Install PostgreSQL
sudo sh -c 'echo "deb https://apt-archive.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# 2. Open the PostgreSQL shell
sudo -u postgres psql

# 3. Create the database role and databases
CREATE ROLE test WITH LOGIN PASSWORD 'test1234';

CREATE DATABASE cordis OWNER test;
GRANT ALL PRIVILEGES ON DATABASE cordis TO test;

CREATE DATABASE oncomx OWNER test;
GRANT ALL PRIVILEGES ON DATABASE oncomx TO test;

\c cordis
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\c oncomx
CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER DATABASE cordis SET search_path TO unics_cordis, public;
ALTER DATABASE oncomx SET search_path TO oncomx_v1_0_25, public;
\q

# 4. Extract and restore the database dumps
# CORDIS
mkdir ./data/databases/ScienceBenchmark/cordis/dumpdir
tar -xf ./data/databases/ScienceBenchmark/cordis/cordis.sql.gz -C ./data/databases/ScienceBenchmark/cordis/dumpdir
pg_restore -F d \
  --dbname="postgresql://test:test1234@localhost:5432/cordis" \
  --no-owner --role=test \
  --clean \
  --jobs=4 \
  dumpdir

# ONCOMX
mkdir ./data/databases/ScienceBenchmark/oncomx/dumpdir
tar -xf ./data/databases/ScienceBenchmark/oncomx/cordis.sql.gz -C ./data/databases/ScienceBenchmark/oncomx/dumpdir
pg_restore -F d \
  --dbname="postgresql://test:test1234@localhost:5432/oncomx" \
  --no-owner --role=test \
  --clean \
  --jobs=4 \
  dumpdir
```

#### Neo4j for CypherBench

<!-- HYUNJUN-REVIEW START: 2026-10-03 | Official CypherBench graph deployment and TierCache connection settings. -->
`databases.tar.gz` contains only the CypherBench schemas. Download and deploy the graphs using the [original CypherBench setup](https://github.com/megagonlabs/cypherbench#1-installation).

Install [Docker](https://docs.docker.com/engine/install/) with the `docker-compose` command and [Git LFS](https://git-lfs.com/) first. CypherBench recommends at least 64 GB RAM for deploying all seven test graphs.

```bash
# Run from the TierCache root in a separate terminal.
cd ..
conda create -n cypherbench python=3.11
conda activate cypherbench
git clone https://github.com/megagonlabs/cypherbench.git
cd cypherbench
pip install -e .

# Download graphs
git lfs install
git clone https://huggingface.co/datasets/megagonlabs/cypherbench benchmark

# Deploy the seven test graphs.
cd docker
bash start_neo4j_test.sh
cd ..

# Initial loading typically takes at least 10 minutes.
python scripts/print_db_status.py
```

Wait until the test graphs are ready, then run TierCache in the `tiercache` environment. The [default deployment](https://github.com/megagonlabs/cypherbench/blob/main/docker/docker-compose-test.yml) uses the following endpoints, with username `neo4j` and password `cypherbench`:

| Workload | Graph | `--target_db_path` |
| --- | --- | --- |
| `COMPANY` | `company` | `bolt://localhost:15062` |
| `ACCIDENT` | `flight_accident` | `bolt://localhost:15064` |
| `MOVIE` | `movie` | `bolt://localhost:15066` |
| `NBA` | `nba` | `bolt://localhost:15067` |

Use `--target_db_dialect cypher`, `--structured_query_type cypher`, and `data/databases/CypherBench/{graph}/schema.json` as `--schema_description_path` (see Section 3).

```bash
# Stop the test graphs from the CypherBench root when finished.
cd docker
bash stop_neo4j_test.sh
```
<!-- HYUNJUN-REVIEW END: CypherBench Neo4j setup -->

## 2. TierCache Configuration

### Config format

Example configuration files are provided in `./configs/cache`.

```json
{
  "cache_db_path": "",
  "vector_db_host": "",
  "vector_db_port": "",

  "t1": {
    "cache_on": true,
    "encoder_path": "",
    "max_nlq_length": 0,
    "num_candidates": 0,
    "similarity_threshold": 0.0,
    "cache_size": 0,
    "eviction_policy": "",
    "vector_db": {
      "collection_name": "",
      "vector_dim": 0,
      "metric_type": "",
      "index_type": ""
    }
  },

  "t2": {
    "cache_on": true,
    "encoder_path": "",
    "filler_path": "",
    "max_nlq_length": 0,
    "max_sq_length": 0,
    "max_sq_template_length": 0,
    "num_candidates": 0,
    "similarity_threshold": 0.0,
    "cache_size": 0,
    "eviction_policy": "",
    "vector_db": {
      "collection_name": "",
      "vector_dim": 0,
      "metric_type": "",
      "index_type": ""
    }
  },

  "text2sq_method": {
    "method": "",
    "model_path": "",
    "dialect": "",
    "max_model_length": 0,
    "vllm_memory_utilization": 0.0
  }
}
```

Here, NLQ denotes a natural language query, and SQ denotes a structured query. The values above are placeholders; use the example configuration files as a starting point.

- `cache_db_path`: path to the local database that stores cache data.
- `vector_db_host`: host address of the vector database server.
- `vector_db_port`: port number of the vector database server.
- `t1` and `t2` share the following fields:
	- `cache_on`: enables or disables the cache tier.
  - `encoder_path`: path to the model used to encode NLQs (the semantic encoder for `t1` and the structure-aware encoder for `t2`).
  - `max_nlq_length`: maximum NLQ length.
  - `num_candidates`: number of candidates retrieved by vector search.
  - `similarity_threshold`: minimum similarity score required for a cache hit.
  - `cache_size`: maximum number of entries stored in the cache tier.
  - `eviction_policy`: policy used to evict entries when the cache tier is full (supported: `LFU`).
  - `vector_db`
    - `collection_name`: name of the vector collection.
    - `vector_dim`: dimensionality of the embedding vectors.
    - `metric_type`: similarity metric used for vector search.
    - `index_type`: index type used in the vector database.
- `t2` has the following additional fields:
  - `filler_path`: path to the slot filler model used to fill SQ templates.
  - `max_sq_length`: maximum SQ length.
  - `max_sq_template_length`: maximum SQ template length.
- `text2sq_method`
  - `method`: generation method or framework (supported: `OmniSQL-vLLM` or `T5` for Text-to-SQL; `Qwen-vLLM` for Text-to-Cypher).
  - `model_path`: local path or Hugging Face model ID of the generation model (recommended: `seeklhy/OmniSQL-32B` for `OmniSQL-vLLM`; `Qwen/Qwen2.5-72B-Instruct` for `Qwen-vLLM`).
  - `dialect`: target query language or dialect (supported: `sqlite` or `postgres` for Text-to-SQL; `cypher` for Text-to-Cypher).
  - `max_model_length`: maximum input length supported by `method`.
  - `vllm_memory_utilization`: fraction of GPU memory allocated to `vLLM`.

## 3. Execution Examples

### Running TierCache on a simple NLQ sequence

<!-- HYUNJUN-REVIEW START: 2026-10-03 | Corrected target_db_path to ./data/databases/... and schema_description_path to ./data/databases/.../schema.txt. -->
```bash
CUDA_VISIBLE_DEVICES=0 python src/run_example.py \
	--config_path "./configs/cache/example.json" \
	--target_db_path "./data/databases/EHRSQL/eicu/db.sqlite" \
	--target_db_dialect "sqlite" \
	--schema_description_path "./data/databases/EHRSQL/eicu/schema.txt" \
 	--structured_query_type "sql"
```
<!-- HYUNJUN-REVIEW END: EHRSQL example path corrections -->

- `config_path`: path to the configuration file.
- `target_db_path`: path to the target database file or the database connection URL.
- `target_db_dialect`: query language or dialect used by the target database.
- `structured_query_type`: type of structured query (`sql` or `cypher`).
- `schema_description_path`: path to a schema file in one of the formats shown below. See the schema files under `./data/databases` for detailed examples.

  For Text-to-SQL, provide SQL table definitions in a text file (e.g., `schema.txt`):

  ```sql
  CREATE TABLE diagnosis
  (
      diagnosisid INT NOT NULL PRIMARY KEY,
      patientunitstayid INT NOT NULL,
      diagnosisname VARCHAR(200) NOT NULL,
      diagnosistime TIMESTAMP(0) NOT NULL,
      icd9code VARCHAR(100),
      FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
  );
  ```

  For Text-to-Cypher, provide entity and relation definitions in a JSON file (e.g., `schema.json`). 

  ```json
  {
    "entities": [
      {
        "label": "FlightAccident",
        "description": null,
        "properties": {
          "number_of_survivors": "int",
          "number_of_injuries": "int",
          "number_of_deaths": "int",
          "date": "date",
          "location": "str",
          "flight_number": "str"
        }
      },
    ],
    "relations": [
      {
        "label": "operatedBy",
        "subj_label": "FlightAccident",
        "obj_label": "Operator",
        "properties": {}
      },
    ]
  }
  ```

### Running TierCache on SQStream workloads

<!-- HYUNJUN-REVIEW START: 2026-10-03 | Added SQStream download link; corrected execution examples. -->
Download the workloads: [sqstream.tar.gz](https://drive.google.com/file/d/1iVhp_sgfdaV8eHEpT1kvqejXGjKdOTx1/view). Extract the archive from the repository root:

```bash
tar -zxvf sqstream.tar.gz
```

```bash
# EHRSQL
CUDA_VISIBLE_DEVICES=0 python src/run_benchmark.py \
	--workload_path "./data/SQStream/EHRSQL-EICU.json" \
	--config_path "./configs/cache/SQStream-EHRSQL-EICU.json" \
	--target_db_path "./data/databases/EHRSQL/eicu/db.sqlite" \
	--target_db_dialect "sqlite" \
	--schema_description_path "./data/databases/EHRSQL/eicu/schema.txt" \
	--structured_query_type "sql"

# ScienceBenchmark
CUDA_VISIBLE_DEVICES=0 python src/run_benchmark.py \
	--workload_path "./data/SQStream/ScienceBenchmark-CORDIS.json" \
	--config_path "./configs/cache/SQStream-ScienceBenchmark-CORDIS.json" \
	--target_db_path "postgresql://test:test1234@localhost:5432/cordis" \
	--target_db_dialect "postgres" \
	--schema_description_path "./data/databases/ScienceBenchmark/cordis/schema.txt" \
	--structured_query_type "sql"

# BIRD
CUDA_VISIBLE_DEVICES=0 python src/run_benchmark.py \
	--workload_path "./data/SQStream/BIRD-CODE.json" \
	--config_path "./configs/cache/SQStream-BIRD-CODE.json" \
	--target_db_path "./data/databases/BIRD/codebase_community/db.sqlite" \
	--target_db_dialect "sqlite" \
	--schema_description_path "./data/databases/BIRD/codebase_community/schema.txt" \
	--structured_query_type "sql"

# CypherBench
CUDA_VISIBLE_DEVICES=0 python src/run_benchmark.py \
	--workload_path "./data/SQStream/CypherBench-MOVIE.json" \
	--config_path "./configs/cache/SQStream-CypherBench-MOVIE.json" \
	--target_db_path "bolt://localhost:15066" \
	--target_db_dialect "cypher" \
	--schema_description_path "./data/databases/CypherBench/movie/schema.json" \
	--structured_query_type "cypher"
```
<!-- HYUNJUN-REVIEW END: SQStream execution example corrections -->

Target database names for each SQStream workload:

| Workload | Target Database |
| --- | --- |
| `EICU` | `eicu` |
| `MIMIC` | `mimic_iii` |
| `CORDIS` | `cordis` |
| `ONCOMX` | `oncomx` |
| `CODE` | `codebase_community` |
| `STUDENT` | `student_club` |
| `COMPANY` | `company` |
| `ACCIDENT` | `flight_accident` |
| `MOVIE` | `movie` |
| `NBA` | `nba` |

## 4. Lightweight Model Training

<!-- HYUNJUN-REVIEW START: 2026-10-03 | Added Google Drive training dataset download link. -->
Download the train/dev datasets: [datasets.tar.gz](https://drive.google.com/file/d/1SEsB6_IVMIYCQ1GD8iSmm5VLlOSwpuPv/view). Then, extract the archive:

```bash
tar -zxvf datasets.tar.gz
# The datasets will be extracted into the 'data' directory.
```
<!-- HYUNJUN-REVIEW END: training dataset download link -->

To adjust the training settings (e.g., `epochs`, `learning_rate`), edit the configuration files under `./configs/models`. The examples below specify the configuration file paths for each model.

### Structure-aware encoder
```bash
# For Text-to-SQL
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/models/encoder/train_sql.py \
	--config_path "./configs/models/encoder/sql/config1.json"

# For Text-to-Cypher
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/models/encoder/train_cypher.py \
	--config_path "./configs/models/encoder/cypher/config1.json"
```

### Slot filler
```bash
# For Text-to-SQL
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/models/filler/train.py \
	--config_path "./configs/models/filler/sql/sqlite_config1.json"

# For Text-to-Cypher
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/models/filler/train.py \
	--config_path "./configs/models/filler/cypher/config1.json"
```

## 5. SQStream (Optional)

<!-- HYUNJUN-REVIEW START: 2026-10-05 | Optional SQStream generation using BIRD/STUDENT as a single example. -->
Use the released SQStream workloads for evaluation in Section 3. To regenerate a workload, follow the example below for **BIRD / STUDENT (`student_club`)**.

### 1) Prepare source data and configs

Prepare the databases in Section 1. Benchmark question/query files are included under `src/benchmark/data/source/`.

```bash
cp src/benchmark/configs/benchmark_sources.example.yaml src/benchmark/configs/benchmark_sources.yaml
cp src/benchmark/configs/db_config.example.yaml src/benchmark/configs/db_config.yaml
```

Keep only `BIRD` with `dbs: [student_club]` in `benchmark_sources.yaml` and the `student_club` entry in `db_config.yaml`. Set its DB path to `data/databases/BIRD/student_club/db.sqlite`. Relative paths resolve from the repository root.

### 2) Validate setup

```bash
python src/benchmark/scripts/check_setup.py --benchmark BIRD --check-connections
```

### 3) Build structural pairs pools

```bash
python src/benchmark/scripts/build_structural_pairs.py --benchmark BIRD --split Dev
```

### 4) Generate workloads

```bash
python src/benchmark/scripts/generate_workload.py \
  --benchmark_type BIRD \
  --split Dev \
  --target_db STUDENT \
  --distribution_type query_len \
  --num_queries 1000
```

The output is saved to `data/SQStream/BIRD-STUDENT.json`. If the file already exists, select a new path with `--output_file`. See [the generation guide](src/benchmark/README.md) for other workloads.
<!-- HYUNJUN-REVIEW END: optional BIRD/STUDENT SQStream generation -->
