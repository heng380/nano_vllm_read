import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)    # 同名传参覆盖
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):    # 初始化event对象
            event = ctx.Event()      # 进程同步, 主进程会通过event.set通知子进程, 子进程通过event.wait等待任务
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)    # 建立通信, 初始化模型, 建立共享内存池
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)      # 初始化
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):    # 传入string, 进行encoder
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()   # 第一次返回prefill为true, 然后把waiting中的seq全都返回回来, waiting空了, running满了; 第二次
        token_ids = self.model_runner.call("run", seqs, is_prefill)    # 多进程, 所以用call的方法, 批量计算了每个seq的下一个token是什么, 返回的list维度和len(seqs)相同
        self.scheduler.postprocess(seqs, token_ids)   # 每次生成一个token加入到seq中, 之后判断是否完成, 完成就销毁退出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]   # 结束的先收集到outputs中
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)   # 区分prefill和decode的两种token数量统计方式
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished() # return not self.waiting and not self.running

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):    # 复制多份一一对应
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):   # 一一绑定
            self.add_request(prompt, sp)    # 加入scheduler waiting队列中
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():   # waiting队列和running队列都空了的话, 是batch推理,先结束的并不会先返回
            t = perf_counter()
            output, num_tokens = self.step()   # key step 
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
