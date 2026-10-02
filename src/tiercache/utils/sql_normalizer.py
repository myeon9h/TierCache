
import re
from sqlglot import parse_one #, TokenType, expressions as exp
from typing import Dict, List, Tuple, Union
from ..utils.special_tokens import SPECIAL_MASK

class BaseNormalizer:
    def __init__(self):
        pass

class SQLGlotNormalizer(BaseNormalizer):
    def __init__(
        self, 
        read_dialect: str = "sqlite", 
        write_dialect: str = "sqlite" 
    ):
        
        self.read_dialect = read_dialect
        self.write_dialect = write_dialect

        self._NUM_PREFIX = "91919191" 
        self._SQ_STR_TAG = "__SQ_STR_LIT_{i}__" 
        self._DQ_STR_TAG = "__DQ_STR_LIT_{i}__" 
        self._NUM_TAG = self._NUM_PREFIX + "{i:06d}"

    def normalize(self, sql: str, type: str = "semi") -> Union[str, None]:
        if type == "semi":
            encoded_sql, mapping = self._encode_placeholders_for_parse(semi_templ=sql)
        else:
            encoded_sql = sql

        # SQL -> ASTree 
        try:
            astree = parse_one(sql=encoded_sql, dialect=self.read_dialect)
        except:
            # Invalid SQL or semi-template
            return None
        
        # ASTree -> SQL
        normalized_sql = astree.sql(dialect=self.write_dialect)

        if type == "semi":
            normalized_sql = self._decode_placeholders_from_sql(
                encoded_sql = normalized_sql, 
                mapping = mapping
            )
    
        return normalized_sql
    
    def _encode_placeholders_for_parse(self, semi_templ: str) -> Tuple[str, Dict[str, str]]:
        mapping: Dict[str, str] = {}
        count_str_literal = 0
        count_num_literal = 0

        SQ_STR_TAG = self._SQ_STR_TAG
        DQ_STR_TAG = self._DQ_STR_TAG
        NUM_TAG = self._NUM_TAG

        def repl_sq_str(m):
            nonlocal count_str_literal
            count_str_literal += 1
            tag = SQ_STR_TAG.format(i=count_str_literal)
            mapping[f"'{tag}'"] = "SQ_STRING"
            return f"'{tag}'"
        semi_sql = re.sub(rf"'\[{SPECIAL_MASK[1:-1]}\]'", repl_sq_str, semi_sql)

        def repl_dq_str(m):
            nonlocal count_str_literal
            count_str_literal += 1
            tag = DQ_STR_TAG.format(i=count_str_literal)
            mapping[f"'{tag}'"] = "DQ_STRING"
            return f"'{tag}'"
        semi_sql = re.sub(rf'"\[{SPECIAL_MASK[1:-1]}\]"', repl_dq_str, semi_sql)

        def repl_num(m):
            nonlocal count_num_literal
            count_num_literal += 1
            tag = NUM_TAG.format(i=count_num_literal)
            mapping[tag] = "NUMBER"
            return tag
        semi_sql = re.sub(rf"\[{SPECIAL_MASK[1:-1]}\]", repl_num, semi_sql)

        return semi_templ, mapping
    
    def _decode_placeholders_from_sql(self, encoded_sql: str, mapping: Dict[str, str]) -> str:

        NUM_PREFIX = self._NUM_PREFIX

        def restore_str(m):
            body = m.group(1)
            key = f"'{body}'"
            tag = mapping.get(key)
            if tag == "SQ_STRING":
                return f"'{SPECIAL_MASK}'"
            if tag == "DQ_STRING":
                return f'"{SPECIAL_MASK}"'
            return m.group(0)
        normalized_sql = re.sub(r"'(__DQ_STR_LIT_\d+__|__SQ_STR_LIT_\d+__)'", restore_str, normalized_sql)
        
        def restore_num(m):
            key = m.group(0)
            tag = mapping.get(key)
            if tag == "NUMBER":
                return f"{SPECIAL_MASK}"
            return m.group(0)
        normalized_sql = re.sub(rf"\b{re.escape(NUM_PREFIX)}\d{{6}}\b", restore_num, normalized_sql)
        
        return normalized_sql