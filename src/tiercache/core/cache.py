from typing import List, Dict, Any, Union, Tuple, Literal, Optional
from .storage import CacheDB, VectorDB
from .encoder import BaseEncoder

class CacheManager:
    def __init__(
        self,
        cache_db: CacheDB,
        t1_vector_db: Optional[VectorDB],
        t2_vector_db: Optional[VectorDB],
        t1_encoder: Optional[BaseEncoder],
        t2_encoder: Optional[BaseEncoder],
        t1_cache_size: Optional[int] = 10000,
        t2_cache_size: Optional[int] = 10000,
        t1_metric_type: Optional[str] = "COSINE",
        t2_metric_type: Optional[str] = "COSINE",
        t1_eviction_policy: Optional[str] = "LFU",
        t2_eviction_policy: Optional[str] = "LFU"
    ):
        self.cache_db = cache_db
        self.t1_vector_db = t1_vector_db
        self.t2_vector_db = t2_vector_db
        self.t1_encoder = t1_encoder
        self.t2_encoder = t2_encoder
        self.t1_cache_size = t1_cache_size
        self.t2_cache_size = t2_cache_size
        self.t1_metric_type = t1_metric_type
        self.t2_metric_type = t2_metric_type
        self.t1_eviction_policy = t1_eviction_policy
        self.t2_eviction_policy = t2_eviction_policy

        self._check_storage_consistency()

        if self.t1_vector_db != None:
            self.t1_num_cached_data = self.cache_db.get_num_cached_data(tier="t1")
            self.available_t1_id = self.t1_num_cached_data if self.t1_num_cached_data < self.t1_cache_size else None

        if self.t2_vector_db != None:
            self.t2_num_cached_data = self.cache_db.get_num_cached_data(tier="t2")
            self.available_t2_id = self.t2_num_cached_data if self.t2_num_cached_data < self.t2_cache_size else None

        self._current_t1_embedding = None
        self._current_t2_embedding = None

    def search(
        self, 
        tier: Literal["t1", "t2"],
        query: str, 
        top_k: int = 1,
        sim_threshold: float = 0.9,
        search_method: str = "m2"
    ) -> Tuple[List[str], List[int], List[float]]:

        if tier == "t1":
            if self.t1_vector_db == None or self.t1_encoder == None:
                raise ValueError("T1 vector DB and encoder must be provided for T1 search")

            query_emb = self.t1_encoder.encode([query])[0]
            self._current_t1_embedding = query_emb

            search_results = self.t1_vector_db.search(
                query_emb = query_emb, 
                top_k = top_k
            )

            metric_type = self.t1_metric_type

        elif tier == "t2":
            if self.t2_vector_db == None or self.t2_encoder == None:
                raise ValueError("T2 vector DB and encoder must be provided for T2 search")
            
            query_emb = self.t2_encoder.encode([query])[0]
            self._current_t2_embedding = query_emb
            search_results = self.t2_vector_db.search(
                query_emb = query_emb, 
                top_k = top_k
            )

            metric_type = self.t2_metric_type

        else:
            raise ValueError(f"Invalid cache tier: {tier}")

        # Cache miss 1
        if not search_results[0]:
            return [], [], []
        
        candidate_ids = [c.id for c in search_results[0]]
        candidate_scores = [float(c.distance) for c in search_results[0]]

        # Cache miss 2
        if candidate_scores[0] < sim_threshold:
            return [], [], []

        # Cache hit
        # If m2, filter Top-k satisfying the threshold
        if top_k > 1 and search_method == "m2":
            higher_is_better = metric_type in {"IP", "COSINE"}
            def pass_threshold(d: float) -> bool:
                return (d >= sim_threshold) if higher_is_better else (d <= sim_threshold)

            filtered_candidate_ids = [candidate_ids[i] for i in range(len(candidate_ids)) if pass_threshold(candidate_scores[i])]
            candidate_ids = filtered_candidate_ids


        # NLQ id list --> SQ list or SQ-template list
        candidate_sq_contents = self.cache_db.get_sq_contents_by_ids(
            tier = tier, 
            nlq_ids = candidate_ids
        )

        return candidate_sq_contents, candidate_ids, candidate_scores[:len(candidate_ids)]
    
    # write back miss single NLQ
    def write_back_t1_cache_miss(self, nlq: str, sq: str) -> None:
        if self.check_cache_full(tier = "t1"):
            self.available_t1_id = self.t1_eviction(number_of_data = 1)[0]
            self.t1_num_cached_data -= 1

        self._cache_t1(
            nlq_id = self.available_t1_id, 
            nlq = nlq, 
            sq = sq
        )

        self.t1_num_cached_data += 1
        self.available_t1_id += 1

    # write back miss SQ template
    def write_back_t2_cache_miss(self, nlq: str, sq_templ: str) -> None:
        if self.check_cache_full(tier = "t2"):
            self.available_t2_id = self.t2_eviction(number_of_data = 1)[0]
            self.t2_num_cached_data -= 1

        self._cache_t2(
            nlq_id = self.available_t2_id, 
            nlq = nlq, 
            sq_templ = sq_templ
        )
        
        self.t2_num_cached_data += 1
        self.available_t2_id += 1 

    # Cache NLQ & SQ ==
    def _cache_t1(self, nlq_id: int, nlq: str, sq: str):
        # Cache DB write back
        self.cache_db.add(
            tier = "t1",
            nlq_id = nlq_id,
            nlq = nlq, 
            sq_or_template = sq
        )
        # Vector DB write back
        self.t1_vector_db.add_vectors(
            ids = [nlq_id],
            embs = [self._current_t1_embedding]
        )
    
    # Cache NLQ & SQ template ==
    def _cache_t2(self, nlq_id: int, nlq: str, sq_templ: str):
        # Cache DB write back
        self.cache_db.add(
            tier = "t2",
            nlq_id = nlq_id,
            nlq = nlq, 
            sq_or_template = sq_templ
        )
        # Vector DB write back
        self.t2_vector_db.add_vectors(
            ids = [nlq_id],
            embs = [self._current_t2_embedding]
        )

    # Update frequency of cached data when cache hit
    def freq_update_by_cache_hit(self, tier: Literal["t1", "t2"], nlq_id: int) -> None:
        self.cache_db.update_frequency_by_id(tier = tier, nlq_id = nlq_id)

    def _t1_eviction(self, number_of_data: int) -> List[int]:
        if self.t1_eviction_policy == "LFU":
            victim_ids = self.cache_db.evict_by_lfu(tier = "t1", number_of_data = number_of_data)
        else:
            raise NotImplementedError
        self.t1_vector_db.remove_vectors(ids = victim_ids)
        return victim_ids

    def _t2_eviction(self, number_of_data: int) -> List[int]:
        if self.t2_eviction_policy == "LFU":
            victim_ids = self.cache_db.evict_by_lfu(tier = "t2", number_of_data = number_of_data)
        else:
            raise NotImplementedError
        self.t2_vector_db.remove_vectors(ids = victim_ids)
        return victim_ids

    def close(self) -> None:
        self.cache_db.close()
        if (self.t1_vector_db != None) and (self.t2_vector_db != None):
            self.t1_vector_db.close(disconnect_flag=False)
            self.t2_vector_db.close(disconnect_flag=True)
        else:
            if self.t1_vector_db != None:
                self.t1_vector_db.close(disconnect_flag=True)
            else:
                self.t2_vector_db.close(disconnect_flag=True)

    def check_cache_full(self, tier: Literal["t1", "t2"]) -> bool:
        if tier == "t1":
            return self.t1_num_cached_data == self.t1_cache_size
        elif tier == "t2":
            return self.t2_num_cached_data == self.t2_cache_size
        else:
            raise ValueError(f"Invalid cache tier: {tier}")
    
    def _check_storage_consistency(self) -> None:
        if self.t1_vector_db != None:
            num_t1_cache_db_data = self.cache_db.get_num_cached_data(tier="t1")
            num_t1_vector_db_data = self.t1_vector_db.get_num_cached_data()
            if num_t1_cache_db_data != num_t1_vector_db_data:
                raise ValueError(f"Missmatch # of t1 cached data in Cache DB ({num_t1_cache_db_data} data) & VectorDB ({num_t1_vector_db_data} data)")

        if self.t2_vector_db != None:
            num_t2_cache_db_data = self.cache_db.get_num_cached_data(tier="t2")
            num_t2_vector_db_data = self.t2_vector_db.get_num_cached_data()
            if num_t2_cache_db_data != num_t2_vector_db_data:
                raise ValueError(f"Missmatch # of t2 cached data in Cache DB ({num_t2_cache_db_data} data) & VectorDB ({num_t2_vector_db_data} data)")
