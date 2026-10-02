from typing import List, Union
from neo4j import GraphDatabase
from neo4j.time import Date, DateTime
import math

class BaseCypherExecutor:
    def __init__(self):
        pass

class Neo4jExecutor(BaseCypherExecutor):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._driver = GraphDatabase.driver(db_url, auth=("neo4j", "cypherbench"))
        
    def execute(self, cypher: str) -> Union[List, None]:
        try:
            with self._driver.session() as session:
                rows = session.run(cypher)
                cypher_result = self.rows_to_tuples([dict(row) for row in rows])
        except:
            cypher_result = None
        return cypher_result
    
    def normalize_value(self, v):
        if isinstance(v, Date):
            return v.iso_format()
        if isinstance(v, DateTime):
            return v.iso_format()
        if isinstance(v, float):
            if math.isnan(v):
                return "NaN"
            return round(v, 8)
        if isinstance(v, list):
            return tuple(sorted(self.normalize_value(x) for x in v))
        if isinstance(v, dict):
            return tuple(sorted((k, self.normalize_value(val)) for k, val in v.items()))
        return v

    def rows_to_tuples(self, records):
        """
        records: List[Dict[str, Any]]
        key/alias 이름은 무시하고 value tuple만 반환.
        Neo4j driver가 반환한 dict insertion order를 사용.
        """
        if not records:
            return []

        keys = list(records[0].keys())
        result = []
        for row in records:
            # CypherBench처럼 모든 row가 같은 key set이라고 가정/검증
            if set(row.keys()) != set(keys):
                raise ValueError("Inconsistent result columns across rows")
            result.append(tuple(self.normalize_value(row[k]) for k in keys))
        return result

    def close(self):
        self._driver.close()
    