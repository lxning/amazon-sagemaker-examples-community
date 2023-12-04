import json
import logging
import os
import time
from typing import Optional
from pathlib import Path
import subprocess
import shutil

import torch
from generate import _load_model, decode_one_token, encode_tokens, prefill, speculative_decode
from model import Transformer
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

    def initialize(self, ctx):
        self.context = ctx
        properties = ctx.system_properties
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
            )

        quantization = ctx.model_yaml_config["handler"]["quantization"]
        model_dir = ctx.system_properties.get("model_dir")
        model_name = ctx.model_yaml_config["handler"]["model_name"]
        os.environ["MODEL_REPO"] = model_name
        cmd = ["sh", "scripts/prepare.sh", model_name]
        run_script = subprocess.Popen(cmd, cwd="/home/model-server/gpt-fast")
        run_script.wait()
        checkpoint_path = Path(f'{model_dir}/{ctx.model_yaml_config["handler"]["converted_ckpt_dir"]}')
        assert checkpoint_path.is_file(), checkpoint_path

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        draft_model_name = ctx.model_yaml_config["handler"]["draft_model_name"]
        os.environ["draft_model_name"] = draft_model_name
        cmd = ["sh", "scripts/prepare.sh", draft_model_name]
        run_script = subprocess.Popen(cmd, cwd="/home/model-server/gpt-fast")
        run_script.wait()
        shutil.move("/home/model-server/gpt-fast/checkpoints", f'{model_dir}/')
        draft_checkpoint_path = ctx.model_yaml_config["handler"].get("draft_checkpoint_dir", None)
        draft_checkpoint_path = Path(f'{model_dir}/{draft_checkpoint_path}') if draft_checkpoint_path else None

        use_tp = False
        if "LOCAL_RANK" in os.environ:
            use_tp = True
        self.speculate_k = ctx.model_yaml_config["handler"].get("speculate_k", 8)

        logger.info("Loading model ...")
        t0 = time.time()
        self.model = _load_model(checkpoint_path, self.device, torch.bfloat16, use_tp)

        if draft_checkpoint_path:
            self.draft_model = _load_model(draft_checkpoint_path, self.device, torch.bfloat16, use_tp)

        torch.cuda.synchronize()
        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        self.tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

        if ctx.model_yaml_config["handler"]["compile"]:
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )
            self.prefill = torch.compile(self.prefill, fullgraph=True, dynamic=True)

        torch.manual_seed(42 * 42)

        self.initialized = True

    @timed
    def preprocess(self, requests):
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
        tokenizer = self.tokenizer
        period_id = tokenizer.encode(".")[0]

        def call_me(x):
            nonlocal period_id, tokenizer
            text = self.tokenizer.decode([period_id] + x.tolist())[1:]
            send_intermediate_predict_response(
                [text],
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )

        y = self.generate(
            input_data["encoded"],
            input_data["max_new_tokens"],
            speculate_k = self.speculate_k,
            callback=call_me,
            temperature=input_data["temperature"],
            top_k=input_data["top_k"],
        )
        logger.info(f"Num tokens = {y.size(0) - self.prompt_length}")
        return y

    def postprocess(self, y):
        return [""]

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        draft_model: Transformer,
        speculate_k: Optional[int] = 8,
        callback=lambda x: x,
        **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """

        is_speculative = draft_model is not None
        # create an empty tensor of the expected final shape and fill in the current tokens
        T = prompt.size(0)
        T_new = T + max_new_tokens

        max_seq_length = min(T_new, self.model.config.block_size)

        device, dtype = prompt.device, prompt.dtype
        max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
        with torch.device(device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
            if is_speculative and draft_model is not model:
                draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        next_token = self.prefill(
            self.model, prompt.view(1, -1), input_pos, **sampling_kwargs
        )

        period_id = self.tokenizer.encode(".")[0]
        text = self.tokenizer.decode([period_id] + next_token.tolist())[1:]
        send_intermediate_predict_response(
            [text],
            self.context.request_ids,
            "Intermediate Prediction success",
            200,
            self.context,
        )

        if is_speculative:
            prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
        seq[T] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        accept_counts = [0] * (speculate_k + 1)

        if is_speculative:
            input_pos = input_pos.item()  # for speculative decoding easier to keep on host
            while input_pos < T_new - 1:
                cur_token = next_token.view(())

                next_tokens = speculative_decode(
                    model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
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