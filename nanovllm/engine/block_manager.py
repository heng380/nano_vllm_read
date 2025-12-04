from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:    # 一个block就是一个定长seq

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):  # num_blocks是通过utilization计算出给kv cache的显存之后,除以每个block需要的显存大小,算出来的最大block数量
        self.block_size = block_size     
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]     # 所有block
        self.hash_to_block_id: dict[int, int] = dict()     # 通过hash值来定位是否已经存在相同的block
        self.free_block_ids: deque[int] = deque(range(num_blocks))        # 双端队列
        self.used_block_ids: set[int] = set()     # set, 快速定位已经使用过的block

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):   # 依赖上一个hash的累计hash值
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:  # 从free的双端队列中获取头部的block id
        block = self.blocks[block_id]
        assert block.ref_count == 0   # 没有ref
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]   # 分配的block结构体

    def _deallocate_block(self, block_id: int) -> Block:  # 
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):    #只在prefill的时候调用一次, 给输入的sequence一次性分配所有block
        assert not seq.block_table     # 尚未分配block
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):    # num_blocks是token长度除以block_size + 1, 向上取整
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1   # 对满的block计算hash
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # cache miss
                cache_miss = True
            if cache_miss:    # 一个cache miss之后, 后面的就算一样也当作cache miss来处理
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:    # cache找到了
                seq.num_cached_tokens += self.block_size   # 计数总共的cached的token
                if block_id in self.used_block_ids:   # 找到对应的block, ref_count累加
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)    # 分配一块新的内存
            if h != -1:     # block是满的
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):    # 后面的block更容易是特有的, 所以先检查, 先释放
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
