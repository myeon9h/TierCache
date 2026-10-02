import json, re, textwrap
from .text2sql_method import OmniSQLvLLM, T5
from .text2cypher_baselines import QwenvLLM
from ..utils.cypherbench import Nl2CypherSample, PropertyGraphSchema, DataType
from typing import Tuple
from pathlib import Path

# OmniSQL official repo prompt (https://github.com/RUCKBReasoning/OmniSQL)
OMNISQL_INPUT_PROMPT_TEMPLATE = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.'''

T5_INPUT_PROMPT_TEMPLATE = '''You are an expert SQL generator.
Database schema:
{db_schema}

Question:
{question}

Write only the SQL query:
'''

# CypherBench official repo prompt (https://github.com/megagonlabs/cypherbench)
NL2CYPHER_PROMPT_DEFAULT = """Translate the question to Cypher query based on the schema of a Neo4j knowledge graph.
- Output the Cypher query in a single line, without any additional output or explanation. Do not wrap the query with any formatting like ```.
- Perform graph pattern matching in the `MATCH` clause if possible.
- Avoid listing the same entity multiple times in the results. However, if multiple distinct entities share the same name, their names should be repeated as separate entries.
- Do not return node objects. Instead, return entity names or properties.

Graph Schema:
{db_schema}

Question: {question}
Cypher: """

class SQLAdapter:
    def __init__(self):
        """
        Text-to-SQL by OmniSQL, T5
        """
        pass

class OmniSQLAdapter(SQLAdapter):
    def __init__(
        self,
        schema_description_path: str,
        model_path: str = "seeklhy/OmniSQL-7B",
        max_model_length: int = 8192,
        vllm_memory_utilization: float = 0.9
    ):
        self.model_path = model_path

        # Init Model & Tokenizer
        self.omnisql = OmniSQLvLLM(
            model_path = self.model_path, 
            max_model_length = max_model_length,
            vllm_memory_utilization = vllm_memory_utilization
        )
        self.input_prompt_templ = OMNISQL_INPUT_PROMPT_TEMPLATE
        self.target_db_schema = self._read_schema_description(schema_description_path)

    def text2sq(self, query: str, evidence: str) -> str:
        input_prompt = self._make_prompt(
            query = query, 
            evidence = evidence
        )
        outputs = self.omnisql.generate(input_prompt = input_prompt)
        for output in outputs:
            responses = [o.text.strip() for o in output.outputs]
        sql = self._post_process(ans = responses[0])
        return sql

    def _read_schema_description(self, schema_description_path: str) -> str:
        if not Path(schema_description_path).suffix == ".txt":
            raise ValueError("Schema description file must be a .txt file.")
        with open(schema_description_path, "r", encoding="utf-8") as f:
            schema_description = f.read().strip()
        return schema_description

    def _make_prompt(self, query: str, evidence: str) -> str:
        prompt = self.input_prompt_templ.format(
            db_schema = self.target_db_schema, 
            question = (query + " " + evidence).strip()
        )
        return prompt
    
    def _post_process(self, ans: str) -> str:
        fence_re = re.compile(
            r"""
            (?P<fence>```|''')  
            [ \t]*
            (?:sql)?            
            [ \t]*
            (?:\r?\n)?          
            (?P<body>.*?)       
            \r?\n?
            (?P=fence)          
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

        m = fence_re.search(ans)
        sql = m.group("body") if m else ans

        sql = sql.strip(" \t\r\n`\"")

        sql = textwrap.dedent(sql)

        sql = re.sub(r"\s*\r?\n\s*", " ", sql) 
        sql = re.sub(r"[ \t\f\v]+", " ", sql) 
        sql = sql.strip()

        return sql
    
    def get_recent_input_output_token_length(self) -> Tuple[int, int]:
        return self.omnisql.get_recent_input_output_token_length()

class T5Adapter(SQLAdapter):
    def __init__(self, schema_description_path: str, model_path: str = "google/t5-v1_1-large"):
        self.model_path = model_path

        # Init Model & Tokenizer
        self.model = T5(model_path = self.model_path)
        self.input_prompt_templ = T5_INPUT_PROMPT_TEMPLATE
        self.target_db_schema = self._read_schema_description(schema_description_path)

    def text2sq(self, query: str, evidence: str) -> str:
        input_prompt = self._make_prompt(
            query = query, 
            evidence = evidence
        )
        output = self.model.generate(input_prompt = input_prompt)
        return self._post_process(output)

    def _read_schema_description(self, schema_description_path: str) -> str:
        if not Path(schema_description_path).suffix == ".txt":
            raise ValueError("Schema description file must be a .txt file.")
        with open(schema_description_path, "r") as f:
            schema_description = f.read().strip()
        return schema_description

    def _make_prompt(self, query: str, evidence: str) -> str:
        if evidence.strip() != "": # BIRD
            query = f"{query}\n\nEvidence:\n{evidence}"

        prompt = self.input_prompt_templ.format(
            db_schema = self.target_db_schema, 
            question = query.strip()
        )
        return prompt
    
    def _post_process(self, ans: str) -> str:
        fence_re = re.compile(
            r"""
            (?P<fence>```|''')  
            [ \t]*
            (?:sql)?            
            [ \t]*
            (?:\r?\n)?          
            (?P<body>.*?)       
            \r?\n?
            (?P=fence)          
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

        m = fence_re.search(ans)
        sql = m.group("body") if m else ans

        sql = sql.strip(" \t\r\n`\"")

        sql = textwrap.dedent(sql)

        sql = re.sub(r"\s*\r?\n\s*", " ", sql) 
        sql = re.sub(r"[ \t\f\v]+", " ", sql) 
        sql = sql.strip()

        return sql

class CypherAdapter:
    def __init__(self):
        """
        Text-to-Cypher by Qwen
        """
        pass

class QwenAdapter(CypherAdapter):
    def __init__(
        self, 
        schema_description_path: str, # json schema file path
        model_path: str = "Qwen/Qwen2.5-32B-Instruct", 
        max_model_length: int = 4096,
        vllm_memory_utilization: float = 0.9
    ):
        self.model_path = model_path

        # Init Model & Tokenizer
        self.qwen = QwenvLLM(
            model_path = self.model_path, 
            max_model_length = max_model_length,
            vllm_memory_utilization = vllm_memory_utilization
        )
        self.input_prompt_templ = NL2CYPHER_PROMPT_DEFAULT
        self.target_db_schema = self._load_schema_from_json(schema_description_path)

    def text2sq(self, query: str, evidence: str) -> str:
        input_prompt = self._make_prompt(
            query = query, 
            evidence = evidence
        )
        outputs = self.qwen.generate(input_prompt = input_prompt)
        for output in outputs:
            responses = [o.text.strip() for o in output.outputs]
        cypher = self._post_process(ans = responses[0])
        return cypher

    def _load_schema_from_json(self, schema_description_path: str) -> str:
        if not Path(schema_description_path).suffix == ".json":
            raise ValueError("Schema description file must be a .json file.")

        with open(schema_description_path, "r") as fin:
            schema = PropertyGraphSchema.from_json(
                json.load(fin),
                add_meta_properties={'name': DataType.STR}
            ).to_sorted()
        return schema.to_str(exclude_description=True)

    def _make_prompt(self, query: str, evidence: str) -> str:
        prompt = self.input_prompt_templ.format(
            db_schema = self.target_db_schema, 
            question = (query + " " + evidence).strip()
        )
        return prompt
    
    def _post_process(self, ans: str) -> str:
        cypher = ans.strip(" \t\r\n`\"")

        # 들여쓰기 공통 부분 제거
        cypher = textwrap.dedent(cypher)

        # 줄바꿈을 공백 하나로 축약 + 과도한 공백 정리
        cypher = re.sub(r"\s*\r?\n\s*", " ", cypher) # 개행 -> 단일 공백
        cypher = re.sub(r"[ \t\f\v]+", " ", cypher) # 탭/공백 뭉치 -> 단일 공백
        cypher = cypher.strip()

        return cypher