from pymilvus import utility, Collection, connections
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_db_path", required=True)
    parser.add_argument("--t1_vector_db_collection_name", required=True)
    parser.add_argument("--t2_vector_db_collection_name", required=True)
    parser.add_argument("--vector_db_host", required=True)
    parser.add_argument("--vector_db_port", required=True)
    args = parser.parse_args()

    if os.path.exists(args.cache_db_path):
        os.remove(args.cache_db_path)
    else:
        print(f"Cache DB file {args.cache_db_path} does not exist or failed to remove.")

    connections.connect(host=args.vector_db_host, port=args.vector_db_port, alias="default")
    try:
        Collection(name=args.t1_vector_db_collection_name).drop()
    except:
        print(f"Collection {args.t1_vector_db_collection_name} does not exist or failed to drop.")

    try:
        Collection(name=args.t2_vector_db_collection_name).drop()
    except:
        print(f"Collection {args.t2_vector_db_collection_name} does not exist or failed to drop.")

    connections.disconnect(alias="default")