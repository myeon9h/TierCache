"""Paper workload names and the unchanged names of their source databases."""

WORKLOAD_DATABASES = {
    "EHRSQL": {"EICU": "eicu", "MIMIC": "mimic_iii"},
    "ScienceBenchmark": {"CORDIS": "cordis", "ONCOMX": "oncomx"},
    "BIRD": {"CODE": "codebase_community", "STUDENT": "student_club"},
    "CypherBench": {
        "COMPANY": "company", "ACCIDENT": "flight_accident",
        "MOVIE": "movie", "NBA": "nba",
    },
}
BENCHMARKS = tuple(WORKLOAD_DATABASES)


def database_name(benchmark, value):
    return WORKLOAD_DATABASES[benchmark].get(value.upper(), value)


def workload_name(benchmark, database):
    for name, db in WORKLOAD_DATABASES[benchmark].items():
        if database == db:
            return name
    raise ValueError(f"No paper workload name for {benchmark}/{database}; specify --output_file")
