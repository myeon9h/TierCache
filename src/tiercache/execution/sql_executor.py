import sqlite3
import psycopg
from typing import List, Union

class BaseSQLExecutor:
    def __init__(self):
        pass

class SQLiteExecutor(BaseSQLExecutor):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._conn = sqlite3.connect(self.db_url, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        
    def execute(self, sq: str) -> Union[List, None]:
        try:
            exec_result = self._conn.execute(sq).fetchall()
        except:
            exec_result = None
        return exec_result
    
    def close(self) -> None:
        self._conn.close()
    
class PostgresExecutor:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._conn = psycopg.connect(self.db_url)
        
    def execute(self, sq: str) -> Union[List, None]:
        try:
            exec_result = self._conn.execute(sq).fetchall()
        except:
            self._conn.rollback()
            exec_result = None
        return exec_result

    def close(self) -> None:
        self._conn.close()