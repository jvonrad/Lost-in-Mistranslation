import json
from collections import defaultdict
from datasets import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# -----------------
# Config
# -----------------
JSONL_PATH = "/data/jonathan/Lost-in-Mistranslation/TED2025/multi_way.jsonl"
BASE_MODEL_LLAMA = "meta-llama/Llama-2-7b-hf"
BASE_MODEL_OLMO = "allenai/OLMo-7B-hf"
BASE_MODEL=BASE_MODEL_LLAMA # <-- change this to the desired base model

# For the *stats* part only (optional): enforce all these langs exist
REQ_LANGS = ["de","en","es","fr","ja","ru","zh"]

# Continued-pretrain settings
USE_TAGS = False          # <-- flip to False to try "no tags"
CHUNK_TOKENS = 4096      # long chunks for efficiency (4096)
MAX_STEPS = 120           # small number of steps for quick test; increase for more training

OUTDIR = f"/data/jonathan/Lost-in-Mistranslation/models/{BASE_MODEL.split('/')[-1]}-ted2025-cpt-{MAX_STEPS}steps" + ("-tags" if USE_TAGS else "-notags")

# -----------------
# Tokenizer
# -----------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# -----------------
# Helpers
# -----------------
def has_all_langs(obj, req_langs=REQ_LANGS):
    pd = obj.get("para_data", {})
    return all((l in pd) and (pd[l] is not None) and (str(pd[l]).strip() != "") for l in req_langs)

def normalize_pd(para_data):
    """Keep only non-empty strings."""
    out = {}
    for l, t in para_data.items():
        if t is None:
            continue
        s = str(t).strip()
        if s:
            out[l] = s
    return out

def format_segment(para_data, use_tags: bool, lang_order=None):
    """
    Turn one JSON row's para_data into text to be learned via next-token prediction.
    - use_tags=False: just values concatenated
    - use_tags=True: "<en>\\ntext" style tags (plain text, no tokenizer resize)
    """
    pd = normalize_pd(para_data)
    if not pd:
        return ""

    if lang_order is None:
        langs = list(pd.keys())
    else:
        langs = [l for l in lang_order if l in pd]

    if use_tags:
        parts = [f"<{l}>\n{pd[l]}" for l in langs]
    else:
        parts = [pd[l] for l in langs]

    # EOS boundary between subtitle “rows”
    return "\n\n".join(parts) + tok.eos_token

# -----------------
# Optional: stats on strict 7-way rows
# -----------------
rows = 0
talk_ids = set()
tok_counts_by_lang = defaultdict(int)

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if not has_all_langs(obj):
            continue

        rows += 1
        talk_ids.add(obj.get("talk_id"))

        pd = obj["para_data"]
        for l in REQ_LANGS:
            tok_counts_by_lang[l] += len(tok.encode(pd[l], add_special_tokens=False))

import torch.distributed as dist
is_dist = dist.is_available() and dist.is_initialized()
rank = dist.get_rank() if is_dist else 0

if rank == 0:
    # your stats counting + prints
	print("Strict rows kept (all REQ_LANGS present):", rows)
	print("Unique talks (strict):", len(talk_ids))
	print("Token counts by lang (strict):", dict(tok_counts_by_lang))
	print("Total tokens (strict, sum over langs):", sum(tok_counts_by_lang.values()))

# -----------------
# Per-talk buffered packing generator (works even if a row has only 2-3 langs)
# -----------------
def talk_chunk_generator(
    jsonl_path: str,
    tokenizer,
    use_tags: bool,
    chunk_tokens: int,
    lang_order=None,
    require_all_langs: bool = False,
):
    import torch.distributed as dist

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world = dist.get_world_size() if is_dist else 1

    buffer_text = ""
    buffer_tok = 0
    current_talk = None

    def flush():
        nonlocal buffer_text, buffer_tok
        if buffer_text:
            yield {"text": buffer_text}
        buffer_text, buffer_tok = "", 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # shard by line number across ranks
            if idx % world != rank:
                continue

            if not line.strip():
                continue

            obj = json.loads(line)

            if require_all_langs and (not has_all_langs(obj)):
                continue

            talk_id = obj.get("talk_id")
            para_data = obj.get("para_data", {})

            seg = format_segment(para_data, use_tags=use_tags, lang_order=lang_order)
            if not seg:
                continue

            # cheap token estimate (avoid CPU tokenization here)
            seg_tok = len(seg) // 4

            # talk boundary -> flush to avoid mixing talks (within this rank's shard)
            if current_talk is None:
                current_talk = talk_id
            elif talk_id != current_talk:
                yield from flush()
                current_talk = talk_id

            if seg_tok >= chunk_tokens:
                yield from flush()
                yield {"text": seg}
                continue

            if buffer_tok + seg_tok > chunk_tokens:
                yield from flush()
                buffer_text = seg
                buffer_tok = seg_tok
            else:
                buffer_text += seg
                buffer_tok += seg_tok

    yield from flush()
    
# Choose whether you want to restrict training to rows where all 7 langs exist:
REQUIRE_ALL_7WAY = True  # recommended False for more data

train_ds = IterableDataset.from_generator(
    lambda: talk_chunk_generator(
        JSONL_PATH,
        tokenizer=tok,
        use_tags=USE_TAGS,
        chunk_tokens=CHUNK_TOKENS,
        lang_order=REQ_LANGS,          # fixed order; set None to keep dict order
        require_all_langs=REQUIRE_ALL_7WAY,
    )
)

# -----------------
# Continued pretraining (next-token prediction)
# -----------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="bfloat16",
)
model.config.pad_token_id = tok.pad_token_id

# check model max length vs chunk size before training (mismatch will cause silent truncation)
print("tok.model_max_length:", tok.model_max_length)
print("max_position_embeddings:", getattr(model.config, "max_position_embeddings", None))


from trl import SFTConfig, SFTTrainer



sft_args = SFTConfig(
    max_length=CHUNK_TOKENS,
    packing=False,
    dataset_text_field="text",   # default is "text", but explicit is fine
    output_dir=OUTDIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    max_steps=MAX_STEPS,
    logging_steps=10,
    bf16=True,
    optim="adamw_torch",
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    warmup_steps=int(0.03 * MAX_STEPS),
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    processing_class=tok,        # tokenizer/processor goes here in current API
)


#Print a couple of samples to verify formatting looks reasonable before training
it = iter(train_ds)
for i in range(2):
    ex = next(it)
    print("=== SAMPLE", i, "===")
    print(ex["text"][:1000])
    


trainer.train()
trainer.save_model(f"{OUTDIR}/final")
tok.save_pretrained(f"{OUTDIR}/final")