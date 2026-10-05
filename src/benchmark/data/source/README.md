# SQStream benchmark inputs

These JSON files are the research inputs used by the SQL and Cypher generation repositories. They were copied without changing their contents; `manifest.json` records the origin, size, record count and SHA-256 of each file. Database files and dumps are excluded.

| Source | Included files | Default SQStream input |
| --- | --- | --- |
| EHRSQL | `eicu/` and `mimic_iii/`: `train.json`, `valid.json`, `test.json`, `tables.json` | `valid.json` |
| ScienceBenchmark | `cordis/` and `oncomx/`: `seed.json`, `synth.json`, `dev.json`, `tables.json` | `dev.json` |
| BIRD | `student_club/` and `codebase_community/`: `dev.json`, `tables.json` | `dev.json` |
| CypherBench | `raw/` and `preprocessed/`: `train.json`, `test.json` | `preprocessed/test.json` |

Copy `../../configs/benchmark_sources.example.yaml` to `../../configs/benchmark_sources.yaml` as described in the generation guide. The default config selects SQL Dev and Cypher Test. Other included splits require corresponding config entries.

## Sources and licenses

The original authors retain their rights to these datasets. The TierCache code license does not replace the dataset licenses.

- **EHRSQL:** Gyubok Lee et al., *EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records* (NeurIPS 2022). [Original repository](https://github.com/glee4810/EHRSQL); [CC BY 4.0](EHRSQL/LICENSE). The local research snapshot is retained rather than replaced with a newer upstream release.
- **BIRD:** Jinyang Li et al., *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs* (NeurIPS 2023). [Original release and license announcement](https://bird-bench.github.io/); [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The included question files are the two DB subsets used for SQStream.
- **ScienceBenchmark:** Yi Zhang et al., *ScienceBenchmark: A Complex Real-World Benchmark for Evaluating Natural Language to SQL Systems* (PVLDB 2024). [Dataset repository](https://github.com/ckosten/sciencebenchmark_dataset). No explicit dataset license is stated in the upstream repository.
- **CypherBench:** Yanlin Feng, Simone Papicchio and Sajjadur Rahman, *CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era* (ACL 2025). [Original dataset](https://huggingface.co/datasets/megagonlabs/cypherbench); [Apache 2.0](CypherBench/LICENSE). The local `raw/` files retain the question/query records used by CypherStream; `preprocessed/` contains the CypherStream masking and sampling annotations derived from those records.
