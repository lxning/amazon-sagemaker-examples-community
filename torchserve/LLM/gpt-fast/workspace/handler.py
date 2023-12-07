import json
import logging
import os
import time
from typing import Optional
from pathlib import Path
import subprocess
import shutil
import sys

sys.path.append('/home/model-server/gpt-fast')

import torch
from generate import _load_model, decode_one_token, encode_tokens, prefill, model_forward, multinomial_sample_one_no_sync
from sentencepiece import SentencePieceProcessor

from ts.handler_utils.timer import timed
from ts.protocol.otf_message_handler import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class GptHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.context = None
        self.prefill = prefill
        self.decode_one_token = decode_one_token
        self.initialized = False
        self.device = torch.device("cpu")
        self.prompt_length = 0
        self.speculate_k = 8
        self.draft_model = None
        self.is_speculative = False

    def initialize(self, ctx):
        self.context = ctx
        properties = ctx.system_properties
        if torch.cuda.is_available():
            self.device = 'cuda'

        quantization = ctx.model_yaml_config["handler"].get("quantization", "int8")
        model_dir = ctx.system_properties.get("model_dir")
 
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        checkpoint_root_path = f'{model_dir}/{model_name}/checkpoints'
        logger.info(f"checkpoint_root_path={checkpoint_root_path}")
        if not os.path.exists(checkpoint_root_path):
            os.environ["MODEL_REPO"] = model_name
            cmd = ["sh", "scripts/prepare.sh", model_name]
            run_script = subprocess.Popen(cmd, cwd="/home/model-server/gpt-fast")
            run_script.wait()

            draft_model_name = ctx.model_yaml_config["handler"].get("draft_model_name", None)
            if draft_model_name:
                os.environ["draft_model_name"] = draft_model_name
                cmd = ["sh", "scripts/prepare.sh", draft_model_name]
                run_script = subprocess.Popen(cmd, cwd="/home/model-server/gpt-fast")
                run_script.wait()
            shutil.move("/home/model-server/gpt-fast/checkpoints", checkpoint_root_path)

        logger.info("checkpoint_root_path already existed")
        checkpoint_path = Path(f'{model_dir}/{ctx.model_yaml_config["handler"]["converted_ckpt_dir"]}')
        assert checkpoint_path.is_file(), checkpoint_path

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        draft_checkpoint_path = ctx.model_yaml_config["handler"].get("draft_checkpoint_dir", None)
        draft_checkpoint_path = Path(f'{model_dir}/{draft_checkpoint_path}') if draft_checkpoint_path else None

        use_tp = False
        if "LOCAL_RANK" in os.environ:
            use_tp = True
            rank = int(os.getenv("LOCAL_RANK", 0))
            world_size = int(os.getenv("WORLD_SIZE", 0))
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
        
        #self.is_speculative = draft_checkpoint_path is not None

        logger.info("Loading model ...")
        t0 = time.time()
        self.model = _load_model(checkpoint_path, self.device, torch.bfloat16, use_tp)

        if self.is_speculative:
            self.speculate_k = ctx.model_yaml_config["handler"].get("speculate_k", 8)
            self.draft_model = _load_model(draft_checkpoint_path, self.device, torch.bfloat16, use_tp)

        torch.cuda.synchronize()
        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        self.tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

        torch.manual_seed(1234)
        if ctx.model_yaml_config["handler"]["compile"]:
            if self.is_speculative and use_tp:
                torch._inductor.config.triton.cudagraph_trees = False 

            if self.is_speculative:
                self.model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)
            self.decode_one_token = torch.compile(
                decode_one_token, mode="reduce-overhead", fullgraph=True
            )

            self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


        self.initialized = True

    @timed
    def preprocess(self, requests):
        assert(len(requests) == 1), "GPT fast is currently only supported with batch_size=1"

        req_data = requests[0]
        input_data = req_data.get("data") or req_data.get("body")
        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        input_data = json.loads(input_data)

        prompt = input_data["prompt"]

        encoded = encode_tokens(self.tokenizer, prompt, bos=True, device=self.device)

        self.prompt_length = encoded.size(0)

        return {
            "encoded": encoded,
            "max_new_tokens": input_data.get("max_new_tokens", 50),
            "temperature": input_data.get("temperature", 0.8),
            "top_k": input_data.get("top_k", 1),
        }

    @timed
    def inference(self, input_data):
        torch.cuda.synchronize()
        tokenizer = self.tokenizer
        period_id = tokenizer.encode(".")[0]

        done_generating = False
        def call_me(x):
            nonlocal done_generating, period_id, tokenizer
            if done_generating:
                return
            logger.info(f"period_id={period_id},type={type(period_id)}, x={x.tolist()}, type={type(x)}")
            text = self.tokenizer.decode([period_id]+x.tolist())[1:]
            rank=os.getenv("LOCAL_RANK", 0)
            logger.info(f"text={text}, rank={rank}")
            send_intermediate_predict_response(
                [text],
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )
            if x.item() == tokenizer.eos_id():
                done_generating = True

        logger.info(f'input_data={input_data["encoded"]}, max_new_tokens={input_data["max_new_tokens"]}') 
        y = self.generate(
            prompt=input_data["encoded"],
            max_new_tokens=input_data["max_new_tokens"],
            speculate_k=self.speculate_k,
            callback=call_me,
            temperature=input_data["temperature"],
            top_k=input_data["top_k"],
        )
        logger.info(f"Num tokens = {y.size(0) - self.prompt_length}")
        return y

    def postprocess(self, y):
        return [""]

    def speculative_decode(
        self,
        cur_token: torch.Tensor,
        input_pos: int,
        speculate_k: int,
        **sampling_kwargs
    ) -> torch.Tensor:
        # draft model inference sequentially
        device = cur_token.device
        orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
        draft_tokens, draft_probs = self.decode_n_tokens(
            cur_token=cur_token.view(1, -1),
            input_pos=orig_input_pos.clone(),
            num_new_tokens=speculate_k,
            **sampling_kwargs)

        draft_tokens = torch.cat(draft_tokens)
        # parallel inference on target model using draft tokens
        target_logits = self.model_forward(
            self.model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
        )
        target_probs = self.logits_to_probs(target_logits[0], **sampling_kwargs)
        draft_probs = torch.stack(draft_probs)
        # q: target prob, p: draft prob
        # q >= p: always accept draft token
        # q < p: q/p prob to accept draft token
        p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
        rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

        if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
            accept_length = speculate_k + 1
            last_token = multinomial_sample_one_no_sync(target_probs[-1])
            # fill last token into draft model
            self.model_forward(
                self.draft_model,
                draft_tokens[-1].view(1, -1),
                orig_input_pos + speculate_k,
            )
            return torch.cat([draft_tokens, last_token])
        else:
            accept_length = rejected_locations[0].item()
            p = draft_probs[accept_length]
            q = target_probs[accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            new = new / new.sum()
            next_token = multinomial_sample_one_no_sync(new)
            return torch.cat([draft_tokens[:accept_length], next_token])

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        speculate_k: Optional[int] = 8,
        callback=lambda x: x,
        **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """

        # create an empty tensor of the expected final shape and fill in the current tokens
        T = prompt.size(0)
        T_new = T + max_new_tokens

        max_seq_length = min(T_new, self.model.config.block_size)

        device, dtype = prompt.device, prompt.dtype
        max_seq_length = max_seq_length + speculate_k + 1 if self.is_speculative else max_seq_length
        with torch.device(device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
            if self.is_speculative and self.draft_model is not self.model:
                self.draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        next_token = self.prefill(
            self.model, prompt.view(1, -1), input_pos, **sampling_kwargs
        )
        if self.is_speculative:
            self.prefill(self.draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
        seq[T] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        accept_counts = [0] * (speculate_k + 1)

        #period_id = self.tokenizer.encode(".")[0]
        #text = self.tokenizer.decode([period_id] + next_token.tolist())[1:]
        #send_intermediate_predict_response(
        #    [text],
        #    self.context.request_ids,
        #    "Intermediate Prediction success",
        #    200,
        #    self.context,
        #)

        if self.is_speculative:
            input_pos = input_pos.item()  # for speculative decoding easier to keep on host
            while input_pos < T_new - 1:
                cur_token = next_token.view(())

                next_tokens = self.speculative_decode(
                    cur_token, input_pos, speculate_k, **sampling_kwargs
                )

                accept_counts[len(next_tokens) - 1] += 1
                num_added = min(T_new - input_pos - 1, len(next_tokens))
                seq[input_pos + 1: input_pos + num_added + 1] = next_tokens[: num_added]
                for i in next_tokens[: num_added, ]:
                    callback(i)
                input_pos = input_pos + num_added
                next_token = next_tokens[-1]
        else:
            generated_tokens, _ = self.decode_n_tokens(
                next_token.view(1, -1),
                input_pos,
                max_new_tokens - 1,
                callback=callback,
                **sampling_kwargs,
            )
            seq[T + 1:] = torch.cat(generated_tokens)

        return seq

    def decode_n_tokens(
        self,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        callback=lambda _: _,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        for i in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):  # Actually better for Inductor to codegen attention here
                next_token, next_prob = self.decode_one_token(
                    self.model, cur_token, input_pos, **sampling_kwargs
                )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)
        return new_tokens, new_probs
