import os
import sqlite3
from typing import Any, Dict, List, Optional, Literal

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

class CacheDB:
    def __init__(self):
        pass

class VectorDB:
    def __init__(self):
        pass

class SQLiteCacheDB(CacheDB):
    def __init__(
        self, 
        cache_db_path: str, 
        t1_commit_interval: Optional[int] = 500, 
        t2_commit_interval: Optional[int] = 500
    ):
        
        self.cache_db_path = cache_db_path
        self.DDL = {
            "t1": """
                CREATE TABLE IF NOT EXISTS t1_table (
                    id          INTEGER PRIMARY KEY,
                    nlq         TEXT NOT NULL,
                    sq          TEXT,
                    frequency   INTEGER NOT NULL
                );
            """.strip(),

            "t2": """
                CREATE TABLE IF NOT EXISTS t2_table (
                    id          INTEGER PRIMARY KEY,
                    nlq         TEXT NOT NULL,
                    sq_template TEXT,
                    frequency   INTEGER NOT NULL
                );
            """.strip(),
        }

        self.t1_num_updates = 0
        self.t1_commit_interval = t1_commit_interval

        self.t2_num_updates = 0
        self.t2_commit_interval = t2_commit_interval

        # SQLite setup
        if os.path.exists(cache_db_path):
            print("-- CacheDB found: ", self.cache_db_path)
            self._conn = sqlite3.connect(self.cache_db_path, isolation_level=None)
            self._conn.execute("PRAGMA foreign_keys = ON;")
        else:
            print("-- CacheDB not found")
            print("-- Init CacheDB: ", self.cache_db_path)
            os.makedirs(os.path.dirname(cache_db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.cache_db_path, isolation_level=None)

            # Create CacheDB
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute(self.DDL["t1"])
            self._conn.execute(self.DDL["t2"])
            self._conn.commit()

    def add(self, tier: Literal["t1", "t2"], nlq_id: int, nlq: str, sq_or_template: str) -> None:
        frequency = 1
        if tier == "t1":
            sql = "INSERT INTO t1_table (id, nlq, sq, frequency) VALUES (?,?,?,?)"
        elif tier == "t2":
            sql = "INSERT INTO t2_table (id, nlq, sq_template, frequency) VALUES (?,?,?,?)"
        else:
            raise ValueError(f"Invalid cache tier: {tier}")

        self._conn.execute(
            sql, (nlq_id, nlq.strip(), sq_or_template.strip(), frequency)
        )

        if tier == "t1":
            self.t1_num_updates += 1
            if self.t1_num_updates >= self.t1_commit_interval:
                self._conn.commit()
                self.t1_num_updates = 0
        elif tier == "t2":
            self.t2_num_updates += 1
            if self.t2_num_updates >= self.t2_commit_interval:
                self._conn.commit()
                self.t2_num_updates = 0

    def evict_by_lfu(self, tier: Literal["t1", "t2"], number_of_data: int) -> List[int]:
        if tier == "t1":
            vic_sql = "SELECT id FROM t1_table ORDER BY frequency ASC LIMIT ?"
            del_sql = "DELETE FROM t1_table WHERE id IN ("
        elif tier == "t2":
            vic_sql = "SELECT id FROM t2_table ORDER BY frequency ASC LIMIT ?"
            del_sql = "DELETE FROM t2_table WHERE id IN ("
        else:
            raise ValueError(f"Invalid cache tier: {tier}")

        cur = self._conn.execute(vic_sql, (number_of_data,))
        victim_ids = [c[0] for c in cur.fetchall()]

        # eviction
        for vid in victim_ids:
            del_sql += "?,"
        del_sql = (del_sql[:-1] + ")")
        cur = self._conn.execute(del_sql, tuple(victim_ids))

        if tier == "t1":
            self.t1_num_updates += len(victim_ids)
            if self.t1_num_updates >= self.t1_commit_interval:
                self._conn.commit()
                self.t1_num_updates = 0
        elif tier == "t2":
            self.t2_num_updates += len(victim_ids)
            if self.t2_num_updates >= self.t2_commit_interval:
                self._conn.commit()
                self.t2_num_updates = 0

        return victim_ids

    def update_frequency_by_id(self, tier: Literal["t1", "t2"], nlq_id: int) -> None:
        if tier == "t1":
            sql = "UPDATE t1_table SET frequency = frequency + 1 WHERE id = ?"
        elif tier == "t2":
            sql = "UPDATE t2_table SET frequency = frequency + 1 WHERE id = ?"
        else:
            raise ValueError(f"Invalid cache tier: {tier}")
        
        cur = self._conn.execute(sql, (nlq_id,))
        if cur.rowcount == 0:
            raise ValueError(f"Record not found for id={nlq_id} (tier={tier})")
        
        if tier == "t1":
            self.t1_num_updates += 1
            if self.t1_num_updates >= self.t1_commit_interval:
                self._conn.commit()
                self.t1_num_updates = 0
        elif tier == "t2":
            self.t2_num_updates += 1
            if self.t2_num_updates >= self.t2_commit_interval:
                self._conn.commit()
                self.t2_num_updates = 0

    # NLQ id list --> SQ list or SQ-template list
    def get_sq_contents_by_ids(self, tier: Literal["t1", "t2"], nlq_ids: List[int]) -> List[str]:
        if tier == "t1":
            sql = "SELECT sq FROM t1_table WHERE id = ?"
        elif tier == "t2":
            sql = "SELECT sq_template FROM t2_table WHERE id = ?"
        else:
            raise ValueError(f"Invalid cache tier: {tier}")
            
        sq_contents_list: List[str] = []
        for nid in nlq_ids:
            cur = self._conn.execute(sql, (nid,))
            res = cur.fetchone() # Tuple or None

            if res != None:
                sq_contents_list.append(res[0])
        
        return sq_contents_list
    
    def get_num_cached_data(self, tier: Literal["t1", "t2"]) -> int:
        if tier == "t1":
            sql = "SELECT COUNT(*) FROM t1_table"
        elif tier == "t2":
            sql = "SELECT COUNT(*) FROM t2_table"
        else:
            raise ValueError(f"Invalid cache tier: {tier}")
        
        cur = self._conn.execute(sql)
        num_cached_data = cur.fetchone()[0]
        return num_cached_data
    
    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

class MilvusVectorDB(VectorDB):
    def __init__(
        self,
        host: str,
        port: str,
        collection_name: str,
        dim: int,
        metric_type: str = "COSINE", # "COSINE", "IP" etc.
        index_type: str = "FLAT",    # "HNSW", "FLAT", "IVF_FLAT", "IVF_SQ8" etc.
        index_build_interval: int = 500,
        update_batch_size: int = 500,
        index_params: Optional[Dict[str, Any]] = {},
        search_params: Optional[Dict[str, Any]] = {"metric_type": "COSINE", "params": {}},
        alias: str = "default",
    ):
        self.collection_name = collection_name
        self.dim = dim
        self.metric_type = metric_type
        self.index_type = index_type
        self.index_params = index_params # or ({"M": 16, "efConstruction": 120} if index_type == "HNSW" else {"nlist": 1024})
        self.search_params = search_params # or ({"metric_type": self.metric_type, "params": {"ef": 10}} if self.index_type == "HNSW" else {"metric_type": self.metric_type})
        self.alias = alias

        self.num_updated_vectors = 0
        self.index_build_interval = index_build_interval
        self.update_batch_size = update_batch_size

        # Milvus setup
        if not connections.has_connection(alias=self.alias):
            connections.connect(host=host, port=port, alias=self.alias)

        self._collection = self._ensure_collection()
    
    def _ensure_collection(self) -> Collection:
        if not utility.has_collection(self.collection_name):
            print("-- VectorDB not found")
            print("-- Init VectorDB: ", self.collection_name)
            id_field = FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
            )
            emb_field = FieldSchema(
                name="embed", dtype=DataType.FLOAT_VECTOR, dim=self.dim
            )
            schema = CollectionSchema(
                fields=[id_field, emb_field], 
                description="NLQ embeddings"
            )

            col = Collection(name=self.collection_name, schema=schema)
            col.create_index(
                field_name="embed",
                index_params={
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": self.index_params,
                },
            )
        else:
            print("-- VectorDB found: ", self.collection_name)
            col = Collection(name=self.collection_name)

        col.load()
        return col

    def add_vectors(self, ids: List[int], embs: List[Any]) -> None:

        for i in range(0, len(ids), self.update_batch_size):
            id_batch = ids[i:i+self.update_batch_size]
            emb_batch = embs[i:i+self.update_batch_size]
            data = [id_batch, emb_batch]
            self._collection.insert(data)
        
        self.num_updated_vectors += len(ids)

        if self.num_updated_vectors >= self.index_build_interval:
            self._collection.flush()
            if self.index_type != "FLAT":
                self._index_rebuild()
            self._index_reload()
            self.num_updated_vectors = 0
        
    def remove_vectors(self, ids: List[int]) -> None:
        for i in range(0, len(ids), self.update_batch_size):
            id_batch = ids[i:i+self.update_batch_size]
            self._collection.delete(f"id in {id_batch}")

        self.num_updated_vectors += len(ids)

        if self.num_updated_vectors >= self.index_build_interval:
            self._collection.flush()
            self._collection.compact()
            self._collection.wait_for_compaction_completed()
            if self.index_type != "FLAT":
                self._index_rebuild()
            self._index_reload()
            self.num_updated_vectors = 0

    def _index_reload(self) -> None:
        self._collection.release()
        self._collection.load()

    def _index_rebuild(self) -> None:
        self._collection.drop_index(field_name="embed")
        self._collection.create_index(
            field_name="embed",
            index_params={
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": self.index_params,
            },
        )
        utility.wait_for_index_building_complete(self._collection.name)

    def search(self, query_emb: Any, top_k: int = 10) -> Any:
        if len(query_emb) != self.dim:
            raise ValueError(f"Embedding dim mismatch: got {len(query_emb)}, expected {self.dim}")
        
        search_results = self._collection.search(
            data=[query_emb],
            anns_field="embed",
            param=self.search_params,
            limit=top_k,
            output_fields=["id"], 
            consistency_level="Strong",
        )

        return search_results

    def get_num_cached_data(self) -> int:
        ids = self._collection.query(
            expr = "id >= 0",
            output_fields = ["id"],
            consistency_level="Strong"
        )
        return len(ids)

    def close(self, disconnect_flag: bool) -> None:
        self._collection.flush()
        self._collection.compact()
        self._collection.wait_for_compaction_completed()
        self._collection.release()
        if disconnect_flag:
            connections.disconnect(alias = self.alias)