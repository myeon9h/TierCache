import argparse
from benchmark.workload import load_workload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--workload_path", required=True)
    parser.add_argument("--target_db_path", required=True)
    parser.add_argument("--target_db_dialect", required=True)
    parser.add_argument("--schema_description_path", required=True)
    parser.add_argument("--structured_query_type", required=True)
    args = parser.parse_args()

    workload = load_workload(args.workload_path, args.structured_query_type)

    from tiercache.tiercache import TierCache

    cache = TierCache(
        config_path = args.config_path,
        target_db_path = args.target_db_path,
        target_db_dialect = args.target_db_dialect,
        schema_description_path = args.schema_description_path,
        structured_query_type = args.structured_query_type
    )
    
    # For evaluation
    if args.structured_query_type == "sql":
        if args.target_db_dialect == "sqlite":
            from tiercache.execution.sql_executor import SQLiteExecutor
            sq_executor = SQLiteExecutor(db_url = args.target_db_path)
        elif args.target_db_dialect == "postgres":
            from tiercache.execution.sql_executor import PostgresExecutor
            sq_executor = PostgresExecutor(db_url = args.target_db_path)
        else:
            raise ValueError(f"Unsupported dialect for SQL: {args.target_db_dialect}")
    
    elif args.structured_query_type == "cypher":
        if args.target_db_dialect == "cypher":
            from tiercache.execution.cypher_executor import Neo4jExecutor
            sq_executor = Neo4jExecutor(db_url = args.target_db_path)
        else:
            raise ValueError(f"Unsupported dialect for Cypher: {args.target_db_dialect}")
    
    else:
        raise ValueError(f"Unsupported structured query type: {args.structured_query_type}")

    try:
        correct = 0
        num_queries = 0
        for w in workload["queries"]:
            question = w["question"].strip()
            evidence = w.get("evidence","")

            """
            output format: {
                "NLQ": input natural language query,
                "SQ": structured query (SQL or Cypher),
                "serving_path": SQ serving path (T1 hit, T2 hit, or miss),
                "execution_result": SQ execution result,
                "error": type of error if it fails during SQ completion, generation or execution; otherwise, "No Errors"
            }
            """
            output = cache.run(nlq = question, evidence = evidence)

            # Evaluation (Execution Accuracy; EX)
            gold_sq = w[args.structured_query_type].strip()
            gold_result = sq_executor.execute(gold_sq)

            if gold_result is not None and gold_result == output["execution_result"]:
                correct += 1

            num_queries += 1
            if num_queries % 10 == 0:
                print(f"Processed {num_queries} queries. Current accuracy: {100*correct/num_queries:.2f}%")

        print(f"Completed {num_queries} queries. Execution accuracy: {100 * correct / num_queries:.2f}%")
    finally:
        cache.close()
        sq_executor.close()
