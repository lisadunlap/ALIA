"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template, add_model_args


@torch.inference_mode()
def main(args):
    print(args)
    # assert 1 == 0
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    msg = args.message

    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("prompt ", prompt)

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")

torch.inference_mode()
def load_vicuna(args):
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    return model, tokenizer

@torch.inference_mode()
def vicuna(args, msg, model, tokenizer, verbose=False):
    # model, tokenizer = load_model(
    #     args.model_path,
    #     args.device,
    #     args.num_gpus,
    #     args.max_gpu_memory,
    #     args.load_8bit,
    #     args.cpu_offloading,
    #     revision=args.revision,
    #     debug=args.debug,
    # )

    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if verbose:
        print("prompt ", prompt)

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs

from omegaconf import OmegaConf
from utils import nest_dict, read_unknowns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/base.yaml')
    parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
    flags, unknown = parser.parse_known_args()

    overrides = OmegaConf.from_cli(flags.overrides)
    cfg       = OmegaConf.load(flags.config)
    base      = OmegaConf.load('configs/base.yaml')
    dataset_base = OmegaConf.load(cfg.base_config)
    args      = OmegaConf.merge(base, dataset_base, cfg, overrides)
    if len(unknown) > 0:
        print(unknown)
        config = nest_dict(read_unknowns(unknown))
        to_merge = OmegaConf.create(config)
        args = OmegaConf.merge(args, to_merge)
    args.yaml = flags.config

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
