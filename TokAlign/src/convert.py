import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import random
import argparse

_EMBED_DICT = {
    "gpt_neox": "gpt_neox.embed_in.weight",
    "llama": "model.embed_tokens.weight",
    "mistral": "model.embed_tokens.weight",
    "olmo2": "model.embed_tokens.weight",
    "olmo_1124": "model.embed_tokens.weight",
}

_LMHEAD_DICT = {
    "gpt_neox": "embed_out.weight",
    "llama": "lm_head.weight",
    "mistral": "lm_head.weight",
    "olmo2": "lm_head.weight",
    "olmo_1124": "lm_head.weight",
}


def get_weight_keys(model):
    model_type = model.config.model_type
    if model_type not in _EMBED_DICT:
        raise ValueError(
            f"Unsupported model_type={model_type}. "
            f"Known types: {list(_EMBED_DICT.keys())}"
        )
    return _EMBED_DICT[model_type], _LMHEAD_DICT[model_type]


def trans2switch(
    trans_path,
    src_clm_path,
    tgt_clm_path,
    tgt_tok_path,
    random_shuffle=-1,
):
    print(f"Loading source model from {src_clm_path}")
    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tgt_tok = AutoTokenizer.from_pretrained(
        tgt_tok_path,
        trust_remote_code=True,
    )

    with open(trans_path, "r") as f:
        trans = json.load(f)

    embed_key, lm_head_key = get_weight_keys(src_model)

    state_dict = src_model.state_dict()
    src_embed = state_dict[embed_key]
    src_lm_head = state_dict[lm_head_key]

    assert src_embed.shape[0] == src_lm_head.shape[0], "Embedding and lm_head vocab sizes differ."

    src_len, hid_dim = src_embed.shape
    tgt_len = len(tgt_tok)

    print(f"Source vocab size in model: {src_len}")
    print(f"Target tokenizer size: {tgt_len}")
    print(f"Alignment entries: {len(trans)}")

    if len(trans) != tgt_len:
        raise ValueError(
            f"Alignment matrix size ({len(trans)}) does not match target tokenizer size ({tgt_len})."
        )

    tgt_embed = torch.empty((tgt_len, hid_dim), dtype=src_embed.dtype)
    tgt_lm_head = torch.empty((tgt_len, hid_dim), dtype=src_lm_head.dtype)

    for i in range(tgt_len):
        tj = int(trans[str(i)])
        if tj < 0 or tj >= src_len:
            raise ValueError(f"Source index out of range for target id {i}: {tj}")

        if random_shuffle > 0 and random.random() < random_shuffle:
            tj = random.randint(0, src_len - 1)

        tgt_embed[i] = src_embed[tj]
        tgt_lm_head[i] = src_lm_head[tj]

    src_model.resize_token_embeddings(tgt_len)

    new_state_dict = src_model.state_dict()
    new_state_dict[embed_key] = tgt_embed.contiguous()
    new_state_dict[lm_head_key] = tgt_lm_head.contiguous()

    src_model.load_state_dict(new_state_dict, strict=True)

    print(f"Saving converted model to {tgt_clm_path}")
    src_model.save_pretrained(tgt_clm_path)
    tgt_tok.save_pretrained(tgt_clm_path)
    print("Done.")


def random_permute(
    src_clm_path,
    tgt_clm_path,
    tgt_tok_path,
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tgt_tok = AutoTokenizer.from_pretrained(
        tgt_tok_path,
        trust_remote_code=True,
    )

    embed_key, lm_head_key = get_weight_keys(src_model)

    state_dict = src_model.state_dict()
    src_embed = state_dict[embed_key]
    src_lm_head = state_dict[lm_head_key]

    assert src_embed.shape[0] == src_lm_head.shape[0]

    src_len, hid_dim = src_embed.shape
    tgt_len = len(tgt_tok)

    tgt_embed = torch.empty((tgt_len, hid_dim), dtype=src_embed.dtype)
    tgt_lm_head = torch.empty((tgt_len, hid_dim), dtype=src_lm_head.dtype)

    for i in range(tgt_len):
        tj = random.randint(0, src_len - 1)
        tgt_embed[i] = src_embed[tj]
        tgt_lm_head[i] = src_lm_head[tj]

    src_model.resize_token_embeddings(tgt_len)

    new_state_dict = src_model.state_dict()
    new_state_dict[embed_key] = tgt_embed.contiguous()
    new_state_dict[lm_head_key] = tgt_lm_head.contiguous()

    src_model.load_state_dict(new_state_dict, strict=True)
    src_model.save_pretrained(tgt_clm_path)
    tgt_tok.save_pretrained(tgt_clm_path)


def random_initial_all(
    src_clm_path,
    tgt_clm_path,
    tgt_tok_path,
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tgt_tok = AutoTokenizer.from_pretrained(
        tgt_tok_path,
        trust_remote_code=True,
    )

    src_model.resize_token_embeddings(len(tgt_tok))
    src_model.save_pretrained(tgt_clm_path)
    tgt_tok.save_pretrained(tgt_clm_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--one2one-matrix-path", type=str, required=True)
    parser.add_argument("-s", "--source-model-path", type=str, required=True)
    parser.add_argument("-t", "--target-tokenizer-path", type=str, required=True)
    parser.add_argument("-o", "--output-model-path", type=str, required=True)
    parser.add_argument(
        "-r",
        "--random-shuffle-percentage",
        type=float,
        default=-1,
        help="Percentage of token pairs randomly shuffled instead of mapped.",
    )

    args = parser.parse_args()

    trans2switch(
        trans_path=args.one2one_matrix_path,
        src_clm_path=args.source_model_path,
        tgt_clm_path=args.output_model_path,
        tgt_tok_path=args.target_tokenizer_path,
        random_shuffle=args.random_shuffle_percentage,
    )