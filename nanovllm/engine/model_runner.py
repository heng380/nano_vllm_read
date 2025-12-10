import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)   # 多卡通信, world_size 为 tp 规模
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)   # 模型结构加载
        load_model(self.model, config.model)   # 初始化模型权重
        self.sampler = Sampler()   # 温度采样器
        self.warmup_model()    # 空跑一边
        self.allocate_kv_cache()    # 分配给 kv cache 的内存
        if not self.enforce_eager:     # 开启 cuda graph
            self.capture_cudagraph()    # 只对 decode 生效
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:     # 主进程
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)   
                dist.barrier()
            else:
                dist.barrier()     # 等待主进程到达barrier, 创建内存成功
                self.shm = SharedMemory(name="nanovllm") # 连接 共享cpu内存
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()   # 如果有它要做的任务的话
            self.call(method_name, *args)       # call函数
            if method_name == "exit":
                break

    def read_shm(self):         # 阻塞自己, 等待任务到来
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # clear 标志位
        return method_name, args

    def write_shm(self, method_name, *args):     # 分发任务
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):     # step中call run函数
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)   # 主进程给子进程分发任务, 子进程直接进入下面的方法执行
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)   # 计算双重约束下的最大并发 seq 数量, min(4, 512)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)    # 只跑一次 prefill
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]   # prefill 阶段最大值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 每个 gpu 只负责自己的那部分 head 的计算
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize  # 单个 block 需要的内存量, 8 个 head(考虑 tp), 一个 head 是 128 维
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes   # used 为模型没有计算的时候的静态内存消耗, 比如模型权重, peak-current 是指进程拉满并发的情况下, 所产生的额外内存开销
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)   # 把 kvcahe 的内存预留出来, 并把 shape 先设置好
        # config.num_kvcache_blocks, self.block_size slot_mapping 中记录的是这两个维度的
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):  # 绑定模型内存, 只有 attention 有 kv cache
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]  # attention 类中的 k_cache 和 v_cache, 每个是一个kv_blocks*256(size)*head*dim的张量
                layer_id += 1                               # 虽然逻辑上是一个四维数组, 但是在 kernel 计算的时候当做一维数组处理, 通过 block_table* block_size+offset 获取前两个维度的索引, 直接拿回了 head*dim 的元素

    def prepare_block_tables(self, seqs: list[Sequence]):   # 有prefix cache, 准备block tables
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables     # 固定形状的block table   [[1,2,3,-1,-1], [1,2,4,5,6]]

    def prepare_prefill(self, seqs: list[Sequence]):    # slot_mapping 只用于写入, 读取使用 block table 和
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])   # 将所有 seq 合并为一维向量, 去除缓存
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))   # 将所有 pos 合并为一维向量, 和input_ids一一对应
            seqlen_q = seqlen - seq.num_cached_tokens    
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)   # 边界感知
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:                     # 最后一个可能没满
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))    # 没有被cache的token
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 存在 prefix cache, q和k不完全一致
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):  # flash attention相关
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)        # 最后一个token
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)     
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)   # 设置flash atten相关的变量
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:  # 最大捕获到 512
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)    # decode 阶段形状为[batch_size, ], 只需要最后一个 token 就可以了
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]  # 找到最接近的图
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()   # 直接回放
            return self.model.compute_logits(graph_vars["outputs"][:bs])   # 直接拿出来

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)  
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)   # 实际计算, 返回词表维度的logits.
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None   # 进行温度采样.
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):    # 消除 cpu launch kernel 的时间消耗, 通过重放计算图去掉这部分消耗, cpu 一次性发射所有 kernel
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size   # 一个句子的最大 block 数量, 向上取整
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)    # seq 个数, 默认最大
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))   # 定义多个 batch size, 找最接近的
        self.graphs = {}   # 不同 bs 对应的 cudagraph 对象
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False,    # decode
                slot_mapping=slot_mapping[:bs], 
                context_lens=context_lens[:bs], 
                block_tables=block_tables[:bs]
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture, 捕获 kernel 的执行顺序和参数等计算图
            if self.graph_pool is None:
                self.graph_pool = graph.pool()    # 专用内存池用于kernel/activation等
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(    # 一组 placeholder, 用于预分配输入
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
