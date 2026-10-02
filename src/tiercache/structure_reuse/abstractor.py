import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union, Optional, Callable, Any
from sqlglot import TokenType, Tokenizer, expressions as exp, parse_one
from ..utils.sql_normalizer import SQLGlotNormalizer
from ..utils.special_tokens import SPECIAL_MASK, SPECIAL_MASK_AGG, SPECIAL_MASK_OP

class BaseAbstractor:
    def __init__(self):
        pass

class SQLAbstractorPatternMatch(BaseAbstractor):
    def __init__(self, read_dialect: str = "sqlite", write_dialect: str = "sqlite"):
        self.read_dialect = read_dialect
        self.write_dialect = write_dialect
        self.normalizer = SQLGlotNormalizer(
            read_dialect = self.read_dialect, 
            write_dialect = self.write_dialect
        )

        self._UNARY_PRECEDER_CHARS = set("([=,+-*/%&|^~<>!?:;")
        self._UNARY_PRECEDER_WORDS = {
            "BETWEEN", "AND", "OR", "WHEN", "THEN", "ELSE",
            "IN", "IS", "LIKE", "ILIKE", "NOT",
            "CASE", "VALUES", "SET", "ON", "WHERE", "BY", "OVER"
        }

        target_agg_funcs = {"COUNT", "SUM", "AVG", "MIN", "MAX"}

        # target_op_keywords = [
        #     "IS NOT", "NOT BETWEEN", "NOT ILIKE", "NOT LIKE", "NOT IN",
        #     "BETWEEN", "ILIKE", "LIKE", "IN", "IS",
        #     ">=", "<=", "<>", "!=", ">", "<", "="
        # ]

        # AGG
        self._AGG_REGEX = re.compile(
            r'\b(' + '|'.join(target_agg_funcs) + r')\s*\(\s*(.*?)\s*\)',
            flags=re.IGNORECASE | re.DOTALL
        )

        # column or (table.column)
        self._IDENT_PART = r'(?:[A-Za-z_][A-Za-z0-9_]*|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]+\])'
        self._IDENT_FULL = rf'{self._IDENT_PART}(?:\.{self._IDENT_PART})*'
        self._IDENT_FULL_REGEX = re.compile(self._IDENT_FULL)

        # OP
        self._PATTERN_NOT_BEFORE_COL = re.compile(
            rf'\bNOT(\s+)({self._IDENT_FULL})(\s+)(IS|LIKE|ILIKE|IN|BETWEEN)\b',
            flags=re.IGNORECASE
        )
        self._PATTERN_COL_WITH_NOT_OP = re.compile(
            rf'({self._IDENT_FULL})(\s+)(IS\s+NOT|NOT\s+LIKE|NOT\s+ILIKE|NOT\s+IN|NOT\s+BETWEEN)\b',
            flags=re.IGNORECASE
        )
        self._PATTERN_COL_WITH_OP = re.compile(
            rf'({self._IDENT_FULL})(\s+)(IS|LIKE|ILIKE|IN|BETWEEN)\b',
            flags=re.IGNORECASE
        )
        self._PATTERN_COL_WITH_SYM = re.compile(
            rf'({self._IDENT_FULL})(\s*)(' + '|'.join(map(re.escape, [">=", "<=", "<>", "!=", ">", "<", "="])) + r')',
            flags=re.IGNORECASE
        )

    # sql -> semi-template, base template, meta data
    def extract_templs_and_meta_data(self, sql: str) -> Union[Tuple[str, str, Dict[str, Dict[str, List[str]]]], Tuple[None, None, None]]:
        semi_templ = self._extract_semi_templ(sql=sql)
        base_templ, meta_data = self._extract_base_templ_and_meta_data(semi_templ=semi_templ)

        return semi_templ, base_templ, meta_data
    
    # sql -> semi-template, base template, meta data
    def extract_semi_templ(self, sql: str) -> Union[Tuple[str, str, Dict[str, Dict[str, List[str]]]], Tuple[None, None, None]]:
        semi_templ = self._extract_semi_templ(sql=sql)
       
        return semi_templ
    
    def _extract_semi_templ(self, sql: str) -> str:

        templ_tokens: List[str] = []
        cur = 0 
        sql_length = len(sql)

        def _is_ident_char(c: str) -> bool:
            return c.isalnum() or c == "_"

        def _scan_number(start_pos: int) -> Tuple[int, Optional[str]]:
            pos = start_pos
            has_digit = False

            while (pos < sql_length) and sql[pos].isdigit():
                pos += 1
                has_digit = True

            if (pos < sql_length) and (sql[pos] == '.'):
                if ((pos + 1 < sql_length) and (sql[pos + 1].isdigit())) or has_digit:
                    pos += 1
                    while (pos < sql_length) and sql[pos].isdigit():
                        pos += 1
                    has_digit = True
                else:
                    if not has_digit:
                        return start_pos, None

            if not has_digit:
                return start_pos, None

            if (pos < sql_length) and (sql[pos] in 'eE'):
                pos2 = pos + 1
                if (pos2 < sql_length) and (sql[pos2] in '+-'):
                    pos2 += 1
                pos3 = pos2
                while (pos3 < sql_length) and sql[pos3].isdigit():
                    pos3 += 1
                if pos3 > pos2:
                    pos = pos3

            return pos, sql[start_pos:pos]

        def _prev_word_or_char(start_pos: int) -> Tuple[Optional[str], Optional[str]]:
            pos = start_pos - 1
            while (pos >= 0) and sql[pos].isspace():
                pos -= 1
            if pos < 0:
                return None, None
            ch = sql[pos]
            if sql[pos].isalnum() or (sql[pos] == '_'):
                pos2 = pos
                while (pos2 >= 0) and (sql[pos2].isalnum() or (sql[pos2] == '_')):
                    pos2 -= 1
                word = sql[pos2+1:pos+1].upper()
                return ch, word
            return ch, None

        UNARY_PRECEDER_CHARS = self._UNARY_PRECEDER_CHARS
        UNARY_PRECEDER_WORDS = self._UNARY_PRECEDER_WORDS

        def _is_unary_sign(start_pos: int) -> bool:
            if (start_pos >= sql_length) or (sql[start_pos] not in "+-"):
                return False
            pos = start_pos + 1
            while (pos < sql_length) and sql[pos].isspace():
                pos += 1
            if pos >= sql_length:
                return False
            if not (sql[pos].isdigit() or ((sql[pos] == '.') and (pos + 1 < sql_length) and sql[pos + 1].isdigit())):
                return False
            ch, word = _prev_word_or_char(start_pos)
            if ch is None or ch in UNARY_PRECEDER_CHARS or (word and word in UNARY_PRECEDER_WORDS):
                return True
            return False
        
        def _replace_signed_number_with_mask(start_pos: int) -> int:
            sign = sql[start_pos]
            pos = start_pos + 1
            while (pos < sql_length) and sql[pos].isspace():
                pos += 1
            end_pos, num_body = _scan_number(pos)
            if num_body is None:
                templ_tokens.append(sign)
                return start_pos + 1
            templ_tokens.append(f"{SPECIAL_MASK}")
            return end_pos

        def _replace_unsigned_number(start_pos: int) -> int:
            if (start_pos > 0) and _is_ident_char(sql[start_pos - 1]):
                templ_tokens.append(sql[start_pos])
                return start_pos + 1

            end_pos, num_body = _scan_number(start_pos)

            if num_body is None:
                templ_tokens.append(sql[start_pos])
                return start_pos + 1
            if end_pos < sql_length and _is_ident_char(sql[end_pos]):
                templ_tokens.append(sql[start_pos])
                return start_pos + 1

            templ_tokens.append(f"{SPECIAL_MASK}")
            return end_pos
        
        while cur < sql_length:
            ch = sql[cur]
            
            if ch == "'":
                cur += 1
                while cur < sql_length:
                    c = sql[cur]
                    if c == "'":
                        if (cur + 1 < sql_length) and (sql[cur + 1] == "'"):
                            cur += 2
                        else:
                            cur += 1
                            break
                    else:
                        cur += 1
                templ_tokens.append(f"'{SPECIAL_MASK}'")
                continue

            if ch == '"':
                cur += 1
                inner_chars = []
                while cur < sql_length:
                    c = sql[cur]
                    if c == '"':
                        if (cur + 1 < sql_length) and (sql[cur + 1] == '"'):
                            inner_chars.append('"')
                            cur += 2
                        else:
                            cur += 1
                            break
                    else:
                        inner_chars.append(c)
                        cur += 1
                inner = "".join(inner_chars)
                templ_tokens.append(f"\"{SPECIAL_MASK}\"")
                continue

            if (ch in "+-") and _is_unary_sign(cur):
                cur = _replace_signed_number_with_mask(cur)
                continue

            if ch.isdigit() or ((ch == '.') and (cur + 1 < sql_length) and sql[cur + 1].isdigit()):
                cur = _replace_unsigned_number(cur)
                continue

            templ_tokens.append(ch)
            cur += 1

        return "".join(templ_tokens)

    def _extract_base_templ_and_meta_data(self, semi_templ: str) -> Tuple[str, Dict]:
        meta_data: Dict[str, Dict[str, List[str]]] = {}
        base_templ = self._mask_aggs_and_collect_meta_data(semi_templ, meta_data)
        base_templ = self._mask_ops_and_collect_meta_data(base_templ, meta_data)
        return base_templ, meta_data

    def _mask_aggs_and_collect_meta_data(self, sql: str, meta_data: Dict[str, Dict[str, List[str]]]) -> str:
        templ_tokens = []
        pos = 0

        for m in self._AGG_REGEX.finditer(sql):
            func = m.group(1).upper()
            arg = m.group(2)
            templ_tokens.append(sql[pos:m.start()])

            if (func == "COUNT") and re.fullmatch(r'\s*\*\s*', arg or ""):
                templ_tokens.append(m.group(0))  
            else:
                colq = self._get_identifier(arg)
                if colq:
                    d = meta_data.setdefault(colq, {})
                    lst = d.setdefault(SPECIAL_MASK_AGG, [])
                    if func not in lst:
                        lst.append(func)
                templ_tokens.append(f"{SPECIAL_MASK_AGG}(" + arg + ")")
            pos = m.end()
        templ_tokens.append(sql[pos:])
        return "".join(templ_tokens)

    def _mask_ops_and_collect_meta_data(self, sql: str, meta_data: Dict[str, Dict[str, List[str]]]) -> str:

        # A) NOT col KW → col [OP]  (KW∈{IS,LIKE,ILIKE,IN,BETWEEN})
        def repl_a(m: re.Match) -> str:
            sp1, col, sp2, kw = m.group(1), m.group(2), m.group(3), m.group(4)
            kw_up = kw.upper()
            op_text = "IS NOT" if kw_up == "IS" else f"NOT {kw_up}"
            self._add_op_to_meta_data(meta_data, col, op_text)
            return f"{col}{sp2}{SPECIAL_MASK_OP}"

        op_masked_sql = self._sub_iter(sql, self._PATTERN_NOT_BEFORE_COL, repl_a)

        # B) col (IS NOT|NOT LIKE|...) → col [OP]
        def repl_b(m: re.Match) -> str:
            col, sp, op = m.group(1), m.group(2), m.group(3)
            self._add_op_to_meta_data(meta_data, col, op)
            return f"{col}{sp}{SPECIAL_MASK_OP}"

        op_masked_sql = self._sub_iter(op_masked_sql, self._PATTERN_COL_WITH_NOT_OP, repl_b)

        # C) col (IS|LIKE|ILIKE|IN|BETWEEN) → col [OP]
        def repl_c(m: re.Match) -> str:
            col, sp, kw = m.group(1), m.group(2), m.group(3)
            self._add_op_to_meta_data(meta_data, col, kw)
            return f"{col}{sp}{SPECIAL_MASK_OP}"

        op_masked_sql = self._sub_iter(op_masked_sql, self._PATTERN_COL_WITH_OP, repl_c)

        # D) col (>=|<=|<>|!=|>|<|=) → col[sp][OP]
        def repl_d(m: re.Match) -> str:
            col, sp, sym = m.group(1), m.group(2), m.group(3)
            self._add_op_to_meta_data(meta_data, col, sym)
            return f"{col}{sp}{SPECIAL_MASK_OP}"

        op_masked_sql = self._sub_iter(op_masked_sql, self._PATTERN_COL_WITH_SYM, repl_d)
        return op_masked_sql

    def _get_identifier(self, arg: str) -> Union[str, None]:
        a = re.sub(r'^\s*DISTINCT\s+', '', arg, flags=re.IGNORECASE)
        matches = list(self._IDENT_FULL_REGEX.finditer(a))
        if not matches:
            return None
        ident_full = matches[-1].group(0)
        norm = self._normalize_identifier(ident_full)
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', norm.split('.')[-1]):
            return norm
        return None
    
    def _normalize_identifier(self, ident: str) -> str:

        parts = [m.group(0) for m in re.finditer(self._IDENT_PART, ident)]
        parts = [self._strip_part_quotes(p) for p in parts]

        if not parts:
            return ident.strip()
        if len(parts) >= 2:
            return f"{parts[-2]}.{parts[-1]}"
        return parts[-1]
    
    def _strip_part_quotes(self, part: str) -> str:
        part = part.strip()
        if (len(part) >= 2) and ((part[0] == part[-1] == '"') or (part[0] == part[-1] == '`')):
            return part[1:-1].replace(part[0]*2, part[0])  # "" -> ", `` -> `
        if len(part) >= 2 and part[0] == '[' and part[-1] == ']':
            return part[1:-1]
        return part

    def _add_op_to_meta_data(self, meta_data: Dict[str, Dict[str, List[str]]], col_full: str, op_text: str):
        key = self._normalize_identifier(col_full)
        op_norm = " ".join(op_text.upper().split()) 
        d = meta_data.setdefault(key, {})
        lst = d.setdefault("[OP]", [])
        if op_norm not in lst:
            lst.append(op_norm)
    
    def _sub_iter(self, sql: str, pat: re.Pattern, repl_fn: Callable[[re.Match], str]) -> str:
        templ_tokens = []
        pos = 0
        for m in pat.finditer(sql):
            templ_tokens.append(sql[pos:m.start()])
            templ_tokens.append(repl_fn(m))
            pos = m.end()
        templ_tokens.append(sql[pos:])
        return "".join(templ_tokens)

    # sql -> semi-template, base template, meta data
    def extract_templs_and_meta_data_and_normalized_sql(self, sql: str) -> Union[Tuple[str, str, Dict[str, Dict[str, List[str]]], str], Tuple[None, None, None]]:
        normalized_sql = self.normalizer.normalize(sql=sql, type="sql")
        # Invalid SQL
        if normalized_sql is None:
            return None, None, None, None

        semi_templ = self._extract_semi_templ(sql=normalized_sql)
        base_templ, meta_data = self._extract_base_templ_and_meta_data(semi_templ=semi_templ)

        return semi_templ, base_templ, meta_data, normalized_sql

class SQLAbstractorWithVerification(BaseAbstractor):
    def __init__(self, read_dialect: str = "sqlite", write_dialect: str = "sqlite"):
        self.read_dialect = read_dialect
        self.write_dialect = write_dialect

    def _normalize_case(self, value: str, case_sensitive: bool) -> str:
        return value if case_sensitive else value.lower()

    def _find_column_context(self, column_node: exp.Column) -> Tuple[Optional[exp.Select], List[Tuple[str, Optional[str]]]]:
        """Find the nearest SELECT context and table candidates from its FROM clause."""
        current = column_node.parent
        select_node = None

        while current:
            if isinstance(current, exp.Select):
                select_node = current
                break
            current = current.parent

        if not select_node:
            return None, []

        tables: List[Tuple[str, Optional[str]]] = []
        from_node = select_node.find(exp.From)
        if from_node:
            for table in from_node.find_all(exp.Table):
                table_name = str(table.this)
                alias = str(table.alias) if getattr(table, "alias", None) else None
                tables.append((table_name, alias))

        return select_node, tables

    def resolve_column_to_table_column(
        self,
        column_node: exp.Column,
        tree: exp.Expression
    ) -> str:
        """Resolve alias.column or bare column using SQL context only (schema-free)."""

        col_name = str(column_node.this) if hasattr(column_node, "this") else column_node.sql()

        alias_to_table: Dict[str, str] = {}
        default_table: Optional[str] = None
        for node in tree.find_all(exp.Table):
            table_name = str(node.this)
            if getattr(node, "alias", None):
                alias_to_table[str(node.alias)] = table_name
            elif default_table is None:
                default_table = table_name

        if getattr(column_node, "table", None):
            alias = str(column_node.table)
            actual_table = alias_to_table.get(alias)
            if actual_table:
                return f"{actual_table}.{col_name}"
            return f"{alias}.{col_name}"

        _, context_tables = self._find_column_context(column_node)
        if context_tables:
            return f"{context_tables[0][0]}.{col_name}"

        if default_table:
            return f"{default_table}.{col_name}"

        return column_node.sql()

    def _get_literal_type(self, literal_node: exp.Literal, original_sql: str) -> str:
        """Infer literal type from AST value + original SQL quoting context."""
        value_str = str(literal_node.this)

        if f"'{value_str}'" in original_sql or f'"{value_str}"' in original_sql:
            return "str"

        lowered = value_str.lower()
        if lowered in {"true", "false"}:
            return "bool"

        try:
            if "." in value_str:
                float(value_str)
                return "float"
            int(value_str)
            return "int"
        except ValueError:
            return "str"

    def _collect_target_literals(self, sql: str, case_sensitive: bool) -> Set[str]:
        target_literals: Set[str] = set()
        tok_stream = Tokenizer(dialect=self.read_dialect).tokenize(sql)

        for tok in tok_stream:
            if tok.token_type in {TokenType.STRING, TokenType.NUMBER, TokenType.BOOLEAN}:
                literal_value = tok.text.strip("'\"")
                target_literals.add(self._normalize_case(literal_value, case_sensitive))

        return target_literals

    def extract_evidence_literals(self, evidence_text: str, case_sensitive: bool) -> Set[str]:
        """
        Extract literal-like values from evidence text.
        - numbers (int/float)
        - quoted strings ('...' or "...")
        """
        number_pattern = r"\b\d+(?:\.\d+)?\b"
        string_pattern = r"'([^']*)'|\"([^\"]*)\""

        numbers = re.findall(number_pattern, evidence_text or "")
        strings = re.findall(string_pattern, evidence_text or "")

        result: Set[str] = set()

        for num in numbers:
            result.add(self._normalize_case(num, case_sensitive))

        for match in strings:
            value = match[0] if match[0] else match[1]
            value = value.strip()
            if value:
                result.add(self._normalize_case(value, case_sensitive))

        return result

    def extract_literals_with_columns(
        self,
        tree: exp.Expression,
        target_literals: Set[str],
        original_sql: str,
        case_sensitive: bool,
    ) -> List[Dict[str, Any]]:
        """Extract literals used in comparison / IN expressions with their linked column."""
        extracted_data: List[Dict[str, Any]] = []
        seen_values: Set[str] = set()

        def maybe_add(column_node: exp.Column, literal_node: exp.Literal) -> None:
            literal_raw = str(literal_node.this)
            literal_norm = self._normalize_case(literal_raw, case_sensitive)
            if literal_norm not in target_literals:
                return
            if literal_norm in seen_values:
                return

            resolved_column = self.resolve_column_to_table_column(column_node, tree)
            extracted_data.append(
                {
                    "column_name": resolved_column,
                    "value": literal_node.this,
                    "original_type": self._get_literal_type(literal_node, original_sql),
                }
            )
            seen_values.add(literal_norm)

        for comp in tree.find_all((exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ)):
            column_node, literal_node = None, None
            if isinstance(comp.left, exp.Column) and isinstance(comp.right, exp.Literal):
                column_node, literal_node = comp.left, comp.right
            elif isinstance(comp.right, exp.Column) and isinstance(comp.left, exp.Literal):
                column_node, literal_node = comp.right, comp.left

            if column_node and literal_node:
                maybe_add(column_node, literal_node)

        for in_expr in tree.find_all(exp.In):
            if isinstance(in_expr.this, exp.Column):
                column_node = in_expr.this
                for literal_node in in_expr.expressions:
                    if isinstance(literal_node, exp.Literal):
                        maybe_add(column_node, literal_node)

        return extracted_data

    def to_literal_kv(self, literals: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Convert internal literal list into key-value format.
        Example:
        [{"value": "A"}, {"value": "B"}] -> {"0": "A", "1": "B"}
        """
        return {str(i): str(lit.get("value", "")) for i, lit in enumerate(literals)}

    def create_indexed_template(self, text: str, literals: List[Dict[str, Any]], method_prefix: str, case_sensitive: bool) -> str:
        """Mask literal values in NLQ text using [mX_i] tokens."""
        template = text

        indexed_literals = sorted(
            enumerate(literals),
            key=lambda item: len(str(item[1]["value"])),
            reverse=True,
        )

        for i, item in indexed_literals:
            original_value = str(item["value"])
            mask = f"[{method_prefix}_{i}]"
            pattern = r"\b" + re.escape(original_value) + r"\b"
            flags = 0 if case_sensitive else re.IGNORECASE
            template = re.sub(pattern, mask, template, flags=flags)

        return template

    def create_indexed_sql_template(
        self,
        tree: exp.Expression,
        literals: List[Dict[str, Any]],
        method_prefix: str,
        case_sensitive: bool,
    ) -> str:
        """Mask literal values in SQL while preserving quote style for strings."""
        result_sql = tree.sql(dialect=self.write_dialect)

        sorted_literals = sorted(
            enumerate(literals),
            key=lambda item: len(str(item[1]["value"])),
            reverse=True,
        )

        for i, literal_info in sorted_literals:
            original_value = str(literal_info["value"])
            original_type = literal_info.get("original_type", "str")
            mask = f"[{method_prefix}_{i}]"

            if original_type in {"int", "float", "bool"}:
                pattern = r"\b" + re.escape(original_value) + r"\b"
                flags = 0 if case_sensitive else re.IGNORECASE
                result_sql = re.sub(pattern, mask, result_sql, flags=flags)
                continue

            if case_sensitive:
                result_sql = result_sql.replace(f"'{original_value}'", f"'{mask}'")
                result_sql = result_sql.replace(f'"{original_value}"', f'"{mask}"')
            else:
                single_quoted = re.escape(f"'{original_value}'")
                double_quoted = re.escape(f'"{original_value}"')
                result_sql = re.sub(single_quoted, f"'{mask}'", result_sql, flags=re.IGNORECASE)
                result_sql = re.sub(double_quoted, f'"{mask}"', result_sql, flags=re.IGNORECASE)

        return result_sql

    def extract_template_tokens(self, template: str) -> Set[str]:
        return set(re.findall(r"\[m2_\d+\]", template))

    def restore_missing_tokens(self, sql_template: str, literals: List[Dict[str, Any]], missing_tokens: Set[str]) -> str:
        """Restore [m2_i] tokens to original values when NLQ does not contain those masks."""
        result = sql_template

        for i, lit in enumerate(literals):
            token = f"[m2_{i}]"
            if token not in missing_tokens:
                continue

            original_value = str(lit["value"])
            original_type = lit.get("original_type", "str")

            if original_type in {"int", "float", "bool"}:
                result = result.replace(token, original_value)
            else:
                result = result.replace(f"'{token}'", f"'{original_value}'")
                result = result.replace(f'"{token}"', f'"{original_value}"')

        return result

    def validate_sql_template(
        self,
        question_template: str,
        sql_template: str,
        literals: List[Dict[str, Any]],
        evidence_template: Optional[str] = None,
    ) -> str:
        """
        Keep SQL masks only when they are grounded in NLQ (and optional evidence).
        - non-BIRD: NLQ only
        - BIRD(with evidence): NLQ + evidence
        """
        question_tokens = self.extract_template_tokens(question_template)
        allowed_tokens = set(question_tokens)
        if evidence_template:
            allowed_tokens |= self.extract_template_tokens(evidence_template)

        sql_tokens = self.extract_template_tokens(sql_template)
        missing_tokens = sql_tokens - allowed_tokens

        if not missing_tokens:
            return sql_template

        return self.restore_missing_tokens(sql_template, literals, missing_tokens)

    def extract_sq_templ_and_literals(
        self,
        nlq: str,
        sq: str,
        evidence: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """
        Unified preprocess + validation for one example.

        Input:
        - nlq
        - sq
        - evidence (optional, used for BIRD preprocess/validation)

        Output:
        - sq_template
        - literals
        """

        tree = parse_one(sq, dialect=self.read_dialect)

        target_literals = self._collect_target_literals(sq, case_sensitive)
        if evidence != "":
            target_literals.update(self.extract_evidence_literals(evidence, case_sensitive))

        literals = self.extract_literals_with_columns(
            tree=tree,
            target_literals=target_literals,
            original_sql=sq,
            case_sensitive=case_sensitive,
        )

        question_template = self.create_indexed_template(nlq, literals, "m2", case_sensitive)
        evidence_template = (
            self.create_indexed_template(evidence, literals, "m2", case_sensitive)
            if evidence != ""
            else None
        )
        sq_template = self.create_indexed_sql_template(tree, literals, "m2", case_sensitive)
        fixed_sq_template = self.validate_sql_template(
            question_template,
            sq_template,
            literals,
            evidence_template=evidence_template,
        )

        # [m2_i], literal dict --> [LITERAL], literal list
        return self.convert_template_and_literals(fixed_sq_template, self.to_literal_kv(literals))
    
    def convert_template_and_literals(
        self,
        semi_template: str,
        literals: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        """
        Convert:
            semi_template with [m2_i] placeholders
            + literals dict {"i": value}
        into:
            semi_template with [LITERAL]
            + literal list in exact appearance order (including duplicates)

        Example
        -------
        semi_template:
            "A = [m2_0] AND B = [m2_1] AND C = [m2_0]"
        literals:
            {"0": "foo", "1": "bar"}

        returns:
            (
                "A = [LITERAL] AND B = [LITERAL] AND C = [LITERAL]",
                ["foo", "bar", "foo"]
            )
        """
        if not isinstance(semi_template, str):
            raise TypeError("semi_template must be a string.")
        if not isinstance(literals, dict):
            raise TypeError("literals must be a dictionary.")

        appearance_order: List[str] = []

        def replacer(match: re.Match) -> str:
            idx = match.group(1)  # string, e.g. "0"
            if idx not in literals:
                raise KeyError(
                    f"Placeholder [m2_{idx}] found in semi_template, "
                    f"but literals does not contain key '{idx}'."
                )
            appearance_order.append(idx)
            return SPECIAL_MASK

        converted_template = re.compile(r"\[m2_(\d+)\]").sub(replacer, semi_template)
        converted_literals = [literals[idx] for idx in appearance_order]

        return converted_template, converted_literals

class CypherAbstractorWithVerification(BaseTemplExtractor):
    def __init__(self, read_dialect: str = "sqlite", write_dialect: str = "sqlite"):
        self.read_dialect = read_dialect
        self.write_dialect = write_dialect

    def is_inside(self, span: tuple[int, int], containers: list[tuple[int, int]]) -> bool:
        return any(start <= span[0] and span[1] <= end for start, end in containers)

    def is_limit_one(self, cypher: str, match: re.Match[str]) -> bool:
        if match.group(0) != "1":
            return False
        prefix = cypher[max(0, match.start() - 16) : match.start()]
        return bool(re.search(r"\bLIMIT\s*$", prefix, flags=re.IGNORECASE))

    def extract_cypher_literals(self, cypher: str) -> list[dict[str, Any]]:
        literals: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []

        date_pattern = re.compile(r"date\('(?P<value>\d{4}-\d{2}-\d{2})'\)")
        for match in date_pattern.finditer(cypher):
            literals.append(
                {
                    "value": match.group("value"),
                    "type": "date",
                    "cypher_span": match.span("value"),
                    "raw_span": match.span(),
                }
            )
            occupied.append(match.span())

        string_pattern = re.compile(r"'(?P<value>(?:\\'|[^'])*)'")
        for match in string_pattern.finditer(cypher):
            if self.is_inside(match.span(), occupied):
                continue
            literals.append(
                {
                    "value": match.group("value").replace("\\'", "'"),
                    "type": "string",
                    "cypher_span": match.span("value"),
                    "raw_span": match.span(),
                }
            )
            occupied.append(match.span())

        number_pattern = re.compile(r"(?<![\w.])-?\b\d+(?:\.\d+)?\b")
        for match in number_pattern.finditer(cypher):
            if self.is_inside(match.span(), occupied):
                continue
            if self.is_limit_one(cypher, match):
                continue
            value = match.group(0)
            literals.append(
                {
                    "value": value,
                    "type": "float" if "." in value else "number",
                    "cypher_span": match.span(),
                    "raw_span": match.span(),
                }
            )

        return sorted(literals, key=lambda item: item["cypher_span"])

    def spans_overlap(self, left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    def filter_available_spans(
        self,
        spans: list[tuple[int, int]],
        reserved: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        return [span for span in spans if not any(self.spans_overlap(span, taken) for taken in reserved)]

    def has_token_boundary(self, text: str, start: int, end: int, value: str) -> bool:
        if not value:
            return False
        if value[0].isalnum() and start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            return False
        if value[-1].isalnum() and end < len(text) and (text[end].isalnum() or text[end] == "_"):
            return False
        return True

    def find_exact_spans(self, text: str, value: str, *, ignore_case: bool = False) -> list[tuple[int, int]]:
        flags = re.IGNORECASE if ignore_case else 0
        spans: list[tuple[int, int]] = []
        for match in re.finditer(re.escape(value), text, flags):
            start, end = match.span()
            if self.has_token_boundary(text, start, end, value):
                spans.append((start, end))
        return spans

    def replace_spans(self, text: str, replacements: list[tuple[int, int, str]]) -> str:
        output = text
        for start, end, value in sorted(replacements, key=lambda item: item[0], reverse=True):
            output = output[:start] + value + output[end:]
        return output

    def extract_sq_templ_and_literals(
        self,
        nlq: str,
        sq: str,
        evidence: Optional[str] = None,
        case_sensitive: bool = True,
    ) -> dict[str, Any]:
    
        cypher = sq
        candidates = self.extract_cypher_literals(cypher)
        reserved_question_spans: list[tuple[int, int]] = []
        slots: list[dict[str, Any]] = []
        slot_by_value: dict[tuple[str, str], dict[str, Any]] = {}

        prioritized = sorted(candidates, key=lambda item: (-len(item["value"]), item["cypher_span"][0]))
        for candidate in prioritized:
            question_spans = self.filter_available_spans(
                self.find_exact_spans(nlq, candidate["value"], ignore_case=not case_sensitive),
                reserved_question_spans,
            )
            if not question_spans:
                continue

            key = (candidate["type"], candidate["value"])
            if key not in slot_by_value:
                slot = {
                    "value": candidate["value"],
                    "type": candidate["type"],
                    "question_spans": question_spans,
                    "cypher_spans": [],
                }
                slot_by_value[key] = slot
                slots.append(slot)
                reserved_question_spans.extend(question_spans)

            slot_by_value[key]["cypher_spans"].append(candidate["cypher_span"])

        slots = sorted(slots, key=lambda item: min(item["cypher_spans"]) if item["cypher_spans"] else (10**9, 10**9))

        cypher_replacements: list[tuple[int, int, str]] = []
        literals: dict[str, str] = {}

        for idx, slot in enumerate(slots):
            placeholder = f"[m2_{idx}]"
            literals[str(idx)] = str(slot["value"])
            for start, end in slot["cypher_spans"]:
                cypher_replacements.append((start, end, placeholder))

        # [m2_i], literal dict --> [LITERAL], literal list
        return self.convert_template_and_literals(self.replace_spans(cypher, cypher_replacements), literals)

    def convert_template_and_literals(
        self,
        semi_template: str,
        literals: Dict[str, Any],
    ) -> Tuple[str, List[str]]:

        if not isinstance(semi_template, str):
            raise TypeError("semi_template must be a string.")
        if not isinstance(literals, dict):
            raise TypeError("literals must be a dictionary.")

        appearance_order: List[str] = []

        def replacer(match: re.Match) -> str:
            idx = match.group(1) 
            if idx not in literals:
                raise KeyError(
                    f"Placeholder [m2_{idx}] found in semi_template, "
                    f"but literals does not contain key '{idx}'."
                )
            appearance_order.append(idx)
            return SPECIAL_MASK

        converted_template = PLACEHOLDER_PATTERN.sub(replacer, semi_template)
        converted_literals = [literals[idx] for idx in appearance_order]

        return converted_template, converted_literals