import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384     # prefill中总的token数量, 不包括prefix cache的部分
    max_num_seqs: int = 512   # 并发处理的句子数量
    max_model_len: int = 4096    # 模型最长能处理的句子长度, 包含prompt和response
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False    # 减少 kernel launch 的气泡, 预先跑一遍知道有哪些 kernel launch, 后续只要回放就可以
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1      # 总共的block块, 后面会自动计算

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
