import argparse
from tiercache.tiercache import TierCache

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--target_db_path", required=True)
    parser.add_argument("--target_db_dialect", required=True)
    parser.add_argument("--schema_description_path", required=True)
    parser.add_argument("--structured_query_type", required=True)
    args = parser.parse_args()

    # NLQ examples (EHRSQL-EICU)
    nlqs = [
        "count the number of patients who were diagnosed with respiratory alkalosis - therapeutic hyperventilation.",
        "count the number of patients who were diagnosed with respiratory alkalosis - therapeutic hyperventilation.",
        "how many times have patient 002-76990 been to the icu?",
        "how many times have patient 016-5834 been to the icu?",
    ]

    cache = TierCache(
        config_path = args.config_path,
        target_db_path = args.target_db_path,
        target_db_dialect = args.target_db_dialect,
        schema_description_path = args.schema_description_path,
        structured_query_type = args.structured_query_type
    )

    for q in nlqs:
        """
        output format: {
            "NLQ": input natural language query,
            "SQ": structured query (e.g., SQL),
            "serving_path": SQ serving path (T1 hit, T2 hit, or miss),
            "execution_result": SQ execution result,
            "error": type of error if it fails during SQ completion, generation or execution; otherwise, "No Errors"
        }
        """
        output = cache.run(nlq = q)
        print(output)

    cache.close()