import torch, os, json
from datetime import datetime
from typing import Dict, List, Union, Tuple, Optional, TypedDict, Any
from sqlglot import parse

from .core.cache import CacheManager
from .core.storage import SQLiteCacheDB, MilvusVectorDB
from .core.encoder import SimpEncoder, StructEncoder
from .structure_reuse.filler import T5LiteralFiller
from .utils.flags import FILLING_ERROR_FLAG, EXECUTION_ERROR_FLAG, PARSING_ERROR_FLAG, ERROR_FLAGS
from .utils.special_tokens import SPECIAL_MASK

class RunResult(TypedDict):
    NLQ: str
    SQ: str
    serving_path: str
    execution_result: Optional[Any]
    error: str

class TierCache:
    def __init__(
        self, 
        config_path: str, 
        target_db_path: str, 
        schema_description_path: str,
        target_db_dialect: str,
        structured_query_type: str
    ):

        self.config = self._load_config(config_path = config_path)

        supported_dialects = {
            "sql": {"sqlite", "postgres"},
            "cypher": {"cypher"},
        }
        
        if structured_query_type not in supported_dialects:
            raise ValueError(
                f"Unsupported structured_query_type: {structured_query_type}"
            )

        if target_db_dialect not in supported_dialects[structured_query_type]:
            raise ValueError(
                f"Unsupported target_db_dialect "
                f"'{target_db_dialect}' for "
                f"'{structured_query_type}'"
            )

        self.structured_query_type = structured_query_type
        self.target_db_dialect = target_db_dialect

        num_devices = torch.cuda.device_count()
        if num_devices > 0:
            self.light_model_device = f"cuda:{num_devices-1}"
        else:
            self.light_model_device = "cpu"

        self._init_cache_component()
        self._init_structure_reuse_component()
        self._init_generation_component(schema_description_path)
        self._init_execution_component(target_db_path)
        
        print("Initialization done!")

    def run(self, nlq: str, evidence: Optional[str] = None) -> RunResult:
        original_nlq = nlq.strip()
        evidence = (evidence or "").strip()
        nlq = (original_nlq + " " + evidence).strip()

        serving_path = None
        error_flag = None

        # 1. Search Tier 1 cache (Semantic retrieval)
        if self.t1_on:
            candidate_sqs, candidate_nlq_ids, candidate_scores = self._t1_cache_search(nlq=nlq)
        else:
            candidate_sqs = []

        # 2-1. T1 cache hit
        if (self.t1_on) and (len(candidate_sqs) > 0):
            serving_path = "T1 hit"
            
            # Top-k candidate SQs -> Top-1 SQ & Frequency update
            sq = self._t1_cache_hit_handling(
                candidate_sqs = candidate_sqs,
                candidate_nlq_ids = candidate_nlq_ids
            )

            # SQ --> Exeuction result
            exec_result = self.sq_executor.execute(sq = sq)
            if exec_result is None:
                error_flag = EXECUTION_ERROR_FLAG

        # 2-2. T1 cache miss (or T1 cache off)
        else:
            # 3. Search Tier 2 cache (Structure-aware retrieval)
            if self.t2_on:
                candidate_sq_templs, candidate_nlq_ids, candidate_scores = self._t2_cache_search(nlq=nlq)
            else:
                candidate_sq_templs = []

            # 4-1. T2 cache hit
            if (self.t2_on) and (len(candidate_sq_templs) > 0):
                serving_path = "T2 hit"

                # Top-k candidate SQ templates -> Top-1 SQ template
                # & SQ template --> SQL by literal filling
                # & Frequency update
                sq = self._t2_cache_hit_handling(
                    nlq = nlq, 
                    candidate_sq_templs = candidate_sq_templs,
                    candidate_nlq_ids = candidate_nlq_ids
                )
                if SPECIAL_MASK in sq:
                    error_flag = FILLING_ERROR_FLAG

            # 4-2. T2 cache miss (or T2 cache off)
            else:
                serving_path = "miss"

                # Miss handling
                # NLQ -> SQL or Cypher by Text2SQ method
                generated_sq = self.text2sq_adapter.text2sq(
                    query = original_nlq, 
                    evidence = evidence
                )
                
                # Convert generated SQ into target DB dialect if needed 
                # This also serves as a validation step for the generated SQL's parsability, as unparsable SQL will lead to execution errors.
                if self.structured_query_type == "sql":
                    try:
                        parsed = parse(sql=generated_sq, read=self.text2sq_method_dialect)
                        sq = parsed[0].sql(dialect=self.target_db_dialect)
                    except Exception as e:
                        sq = generated_sq
                        error_flag = PARSING_ERROR_FLAG
                else:
                    sq = generated_sq
        
            # 5. SQ execution
            if error_flag is None:
                exec_result = self.sq_executor.execute(sq = sq)
                if exec_result is None:
                    error_flag = EXECUTION_ERROR_FLAG
            else:
                exec_result = None

            # 6. Write back if sq is executable
            if (serving_path == "miss") and (exec_result is not None):
                # T1 write back (NLQ, SQ)
                if self.t1_on:
                    try:
                        self.cache.write_back_t1_cache_miss(nlq = nlq, sq = sq)
                    except Exception as e:
                        print(f"T1 write-back failed: {e}")

                # T2 write back (NLQ, SQ template)
                if self.t2_on:
                    try:
                        # SQ -> SQ template for structure reuse
                        sq_templ, _ = self.abstractor.extract_sq_templ_and_literals(nlq = original_nlq, sq = sq, evidence = evidence)

                        # Write back if the abstracted template contains masked spans
                        if SPECIAL_MASK in sq_templ:
                            self.cache.write_back_t2_cache_miss(nlq = nlq, sq_templ = sq_templ)
                    except Exception as e:
                        print(f"T2 write-back failed: {e}")

        return {
            "NLQ": nlq,
            "SQ": sq,
            "serving_path": serving_path,
            "execution_result": exec_result,
            "error": ERROR_FLAGS[error_flag] if error_flag is not None else "No Errors"
        }
    
    # NLQ -> Top-k candidate SQs with NLQ ids and scores
    def _t1_cache_search(self, nlq: str) -> Tuple[List[str], List[int], List[float]]:
        return self.cache.search(
            tier = "t1",
            query = nlq, 
            top_k = self.t1_num_candidates,
            sim_threshold = self.t1_sim_threshold
        )
    
    # NLQ -> Top-k candidate SQ templates with NLQ ids and scores
    def _t2_cache_search(self, nlq: str) -> Tuple[List[str], List[int], List[float]]:
        return self.cache.search(
            tier = "t2",
            query = nlq, 
            top_k = self.t2_num_candidates,
            sim_threshold = self.t2_sim_threshold
        )
    
    # Top-k candidate SQs -> Top-1 SQ & Update frequency
    def _t1_cache_hit_handling(self, candidate_sqs: List[str], candidate_nlq_ids: List[int]) -> str:
        # 1. Select Top-1 
        # [Re-ranking can be applied here]
        top1_sq = candidate_sqs[0]
        top1_nlq_id = candidate_nlq_ids[0]

        # 2. Frequency update
        self.cache.freq_update_by_cache_hit(tier = "t1", nlq_id = top1_nlq_id)

        return top1_sq

    # Top-k candidate SQ templates -> Top 1 SQ template -> SQ & Update frequency
    def _t2_cache_hit_handling(self, nlq: str, candidate_sq_templs: List[str], candidate_nlq_ids: List[int]) -> str:
        # 1. Select Top-1 
        # [Re-ranking can be applied here]
        top1_sq_templ = candidate_sq_templs[0]
        top1_nlq_id = candidate_nlq_ids[0]

        # 2. Frequency update
        self.cache.freq_update_by_cache_hit(tier = "t2", nlq_id = top1_nlq_id)

        # 3. Literal filling
        sq = self.filler.literal_filling(nlq = nlq, sq_template = top1_sq_templ)

        return sq
    
    def close(self) -> None:
        self.cache.close()
        self.sq_executor.close()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    
    def _init_cache_component(self):
        print("< Init Cache Component>")

        if self.config["t1"]["cache_on"]:
            self.t1_on = True
            t1_config = self.config["t1"]
            t1_vector_db_config = t1_config["vector_db"]
            
            self.t1_sim_threshold = t1_config.get("similarity_threshold", 0.95)
            self.t1_num_candidates = t1_config.get("num_candidates", 1)
            t1_cache_size = t1_config.get("cache_size", 1000)
            t1_metric_type = t1_vector_db_config.get("metric_type", "COSINE")
            t1_eviction_policy = t1_config.get("eviction_policy", "LFU")
            print("- Tier 1 cache ON")
        else:
            self.t1_on = False
            self.t1_sim_threshold = None
            self.t1_num_candidates = None

            t1_cache_size = None
            t1_metric_type = None
            t1_eviction_policy = None
            print("- Tier 1 cache OFF")

        if self.config["t2"]["cache_on"]:
            self.t2_on = True
            t2_config = self.config["t2"]
            t2_vector_db_config = t2_config["vector_db"]

            self.t2_sim_threshold = t2_config.get("similarity_threshold", 0.9)
            self.t2_num_candidates = t2_config.get("num_candidates", 1)
            t2_cache_size = t2_config.get("cache_size", 1000)
            t2_metric_type = t2_vector_db_config.get("metric_type", "COSINE")
            t2_eviction_policy = t2_config.get("eviction_policy", "LFU")
            print("- Tier 2 cache ON")
        else:
            self.t2_on = False
            self.t2_sim_threshold = None
            self.t2_num_candidates = None

            t2_cache_size = None
            t2_metric_type = None
            t2_eviction_policy = None
            print("- Tier 2 cache OFF")

        if (not self.t1_on) and (not self.t2_on):
             raise ValueError("At least one of Tier 1 or Tier 2 cache must be enabled.")

         # Load semantic & structure-aware encoder models
        if self.t1_on:
            t1_encoder_path = t1_config["encoder_path"]
            print("- Loading semantic encoder from:", t1_encoder_path)
            t1_encoder = SimpEncoder(
                model_path = t1_encoder_path, 
                tokenizer_path = t1_encoder_path,
                max_length = t1_config["max_nlq_length"], 
                device = self.light_model_device
            )
        else:
            t1_encoder = None

        if self.t2_on:
            t2_encoder_path = t2_config["encoder_path"]
            print("- Loading structure-aware encoder from:", t2_encoder_path)

            if os.path.isfile(os.path.join(t2_encoder_path, "tokenizer.json")):
                tokenizer_path = t2_encoder_path
            else:
                tokenizer_path = "Salesforce/SFR-Embedding-Code-400M_R"
                print("-- Tokenizer (tokenizer.json) not found in ", t2_encoder_path)
                print("--- Using default tokenizer instead (This may lead to unexpected results): ", tokenizer_path)

            t2_encoder = StructEncoder(
                model_path = t2_encoder_path,
                tokenizer_path = tokenizer_path,
                max_length = t2_config["max_nlq_length"], 
                device = self.light_model_device
            )
        else:
            t2_encoder = None

        # Cache storages
        print("- Loading Cache DB")
        cache_db = SQLiteCacheDB(
            cache_db_path = self.config["cache_db_path"],
            t1_commit_interval= t1_config.get("cache_commit_interval", 500) if self.t1_on else None,
            t2_commit_interval= t2_config.get("cache_commit_interval", 500) if self.t2_on else None,
        )

        if self.t1_on:
            print("- Loading Vector DB (Tier 1)")
            t1_vector_db = MilvusVectorDB(
                host = self.config["vector_db_host"], 
                port = self.config["vector_db_port"], 
                collection_name = t1_vector_db_config["collection_name"],
                dim = t1_vector_db_config["vector_dim"],
                metric_type = t1_metric_type,
                index_type = t1_vector_db_config.get("index_type", "FLAT"),
                index_build_interval = t1_vector_db_config.get("index_build_interval", 500),
                update_batch_size = t1_vector_db_config.get("update_batch_size", 500),
                index_params = t1_vector_db_config.get("index_params", {}),
                search_params = t1_vector_db_config.get("search_params", {"metric_type": t1_metric_type, "params": {}}),
                alias = t1_vector_db_config.get("alias", "default"),
            )
        else:
            print("- Vector DB (Tier 1) OFF")
            t1_vector_db = None

        if self.t2_on:
            print("- Loading Vector DB (Tier 2)")
            t2_vector_db = MilvusVectorDB(
                host = self.config["vector_db_host"], 
                port = self.config["vector_db_port"], 
                collection_name = t2_vector_db_config["collection_name"],
                dim = t2_vector_db_config["vector_dim"],
                metric_type = t2_metric_type,
                index_type = t2_vector_db_config.get("index_type", "FLAT"),
                index_build_interval = t2_vector_db_config.get("index_build_interval", 500),
                update_batch_size = t2_vector_db_config.get("update_batch_size", 500),
                index_params = t2_vector_db_config.get("index_params", {}),
                search_params = t2_vector_db_config.get("search_params", {"metric_type": t2_metric_type, "params": {}}),
                alias = t2_vector_db_config.get("alias", "default"),
            )
        else:
            print("- Vector DB (Tier 2) OFF")
            t2_vector_db = None

        # Cache
        print("- Initiating Cache Mananger")
        self.cache = CacheManager(
            cache_db = cache_db, 
            t1_vector_db = t1_vector_db,
            t2_vector_db = t2_vector_db,
            t1_encoder = t1_encoder, 
            t2_encoder = t2_encoder,
            t1_cache_size = t1_cache_size,
            t2_cache_size = t2_cache_size,
            t1_metric_type = t1_metric_type,
            t2_metric_type = t2_metric_type,
            t1_eviction_policy = t1_eviction_policy,
            t2_eviction_policy = t2_eviction_policy,
        )
        print("- Done")
    
    def _init_structure_reuse_component(self):
        if not self.t2_on:
            print("\nSkip structure reuse component initialization (Tier 2 cache is OFF)")
            self.filler = None
            self.abstractor = None
            return
        
        print("\n< Init Structure Reuse Component>")
        t2_config = self.config["t2"]

        # Literal-filler
        filler_path = t2_config["filler_path"]
        print("- Loading Literal-filler from:", filler_path)
        self.filler = T5LiteralFiller(
            model_path = filler_path, 
            device = self.light_model_device,
            max_input_length = t2_config["max_nlq_length"] + t2_config["max_sq_template_length"],
            max_output_length = t2_config["max_sq_length"]
        )

        # Structured query abstractor
        if self.structured_query_type == "sql":
            print("- Initiating SQL Abstractor")
            from .structure_reuse.abstractor import SQLAbstractorWithVerification
            self.abstractor = SQLAbstractorWithVerification(
                read_dialect = self.target_db_dialect,
                write_dialect = self.target_db_dialect
            )

        elif self.structured_query_type == "cypher":
            print("- Initiating Cypher Abstractor")
            from .structure_reuse.abstractor import CypherAbstractorWithVerification
            self.abstractor = CypherAbstractorWithVerification(
                read_dialect = self.target_db_dialect,
                write_dialect = self.target_db_dialect
            )

        else:
            raise ValueError(f"Unsupported structured_query_type: {self.structured_query_type}")
        
        print("- Done")

    def _init_generation_component(self, schema_description_path: str):
        print("\n< Init Generation Component>")
        text2sq_method_config = self.config["text2sq_method"]
        self.text2sq_method_dialect = text2sq_method_config["dialect"]

        # Adapter for Text2SQL method
        method = text2sq_method_config["method"].lower()
        model_path = text2sq_method_config["model_path"]

        if (self.structured_query_type == "sql") and (method in ["omnisql-vllm", "t5"]):
            print(f"- Initiating Text2SQL method: {text2sq_method_config['method']} ({model_path})")
            if method == "omnisql-vllm":
                from .generation.adapter import OmniSQLAdapter
                self.text2sq_adapter = OmniSQLAdapter(
                    model_path = model_path,
                    schema_description_path = schema_description_path,
                    max_model_length = text2sq_method_config.get("max_model_length", 8192),
                    vllm_memory_utilization = text2sq_method_config.get("vllm_memory_utilization", 0.9)
                )
            elif method == "t5": 
                from .generation.adapter import T5Adapter
                self.text2sq_adapter = T5Adapter(
                    model_path = model_path, 
                    schema_description_path = schema_description_path
                )

        elif (self.structured_query_type == "cypher") and (method in ["qwen-vllm"]):
            print(f"- Initiating Text2Cypher method: {text2sq_method_config['method']} ({model_path})")
            from .generation.adapter import QwenAdapter
            self.text2sq_adapter = QwenAdapter(
                model_path = model_path, 
                schema_description_path = schema_description_path,
                max_model_length = text2sq_method_config.get("max_model_length", 4096),
                vllm_memory_utilization = text2sq_method_config.get("vllm_memory_utilization", 0.9)
            )

        else:
            raise ValueError(f"Unsupported Text2SQ method for {self.structured_query_type}: {text2sq_method_config['method']}")
        
        print("- Done")
    
    def _init_execution_component(self, target_db_path: str):
        print("\n< Init Execution Component>")

        if self.structured_query_type == "sql":
            # -- SQL executor
            print("- Initiating SQL executor")
            if self.target_db_dialect == "sqlite":
                from .execution.sql_executor import SQLiteExecutor
                self.sq_executor = SQLiteExecutor(db_url = target_db_path)
            elif self.target_db_dialect == "postgres":
                from .execution.sql_executor import PostgresExecutor
                self.sq_executor = PostgresExecutor(db_url = target_db_path)
            else:
                raise ValueError(f"Unsupported dialect for SQL: {self.target_db_dialect}")

        elif self.structured_query_type == "cypher":
            # -- Cypher executor
            print("- Initiating Cypher executor")
            from .execution.cypher_executor import Neo4jExecutor
            if self.target_db_dialect == "cypher":
                self.sq_executor = Neo4jExecutor(db_url = target_db_path)
            else:
                raise ValueError(f"Unsupported dialect for Cypher: {self.target_db_dialect}")

        else:
            raise ValueError(f"Unsupported structured_query_type: {self.structured_query_type}")
        
        print("- Done")