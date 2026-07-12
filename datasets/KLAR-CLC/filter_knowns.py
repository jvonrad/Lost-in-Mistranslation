import os
import json
import glob
import random
import argparse
import numpy
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    """Globally set random seed."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    "meta-llama/Llama-2-7b-hf": ["ca", "en", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"],
    "bigscience/bloom-560m": ["ar", "ca", "en", "es", "fr", "vi", "zh"]
}
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent", \
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location", \
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation", \
    "manufacturer", "native_language", "occupation", "official_language", \
    "owned_by", "place_of_birth", "place_of_death","religion"
]
valid_langs = set(languages[model_name])
valid_rels = set(relations)

# ====  Group file paths by relation ====
json_paths = glob.glob("klar/*/*.json")
path_map = defaultdict(dict)  # (relation -> lang -> path)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue

    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]

            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

# ==== Apply prompt formatting ====
def apply_prompt(example):
    prompt = example["template"].replace("<subject>", example["subject"]).replace("<mask>", "")
    example["input"] = prompt.strip()
    example["target"] = " " + example["object"]
    return example

dataset = Dataset.from_list([apply_prompt(ex) for ex in samples])

# ==== Tokenizer & Model ====
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")


# ==== Evaluation ====
def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def evaluate(model, dataset, tokenizer, max_new_tokens=10, n_shot=3, save_path="filter_knowns/results.json"):
    model.eval()
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    for idx, ex in enumerate(tqdm(test_data)):
        lang = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index = ex.get("index", None)
        candidates = [c for c in test_data if c.get("index") != index and c.get("language") == lang and c.get("relation") == relation]
        demonstrations = random.sample(candidates, min(n_shot, len(candidates)))

        few_shot_prompt = "".join([f"{d['input']}{d['target']}\n" for d in demonstrations]) + ex["input"]

        input_ids = tokenizer(few_shot_prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = decoded[len(few_shot_prompt):].split('\n')[0].strip()
        target = ex["target"]

        # match = is_nontrivial_prefix(prediction, target)
        match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
        correct_total += match
        total_total += 1
        per_lang_results[lang]["correct"] += match
        per_lang_results[lang]["total"] += 1
        if match:
            per_lang_results[lang]["correct_indices"].append(index)
        
        # print(f"[{lang}] Q: {ex['input']} | Pred: {prediction} | Label: {target} | Match: {match}")

    overall_acc = correct_total / total_total if total_total > 0 else 0
    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")

    results = {"overall_acc": overall_acc, "overall_clc": None, "per_language_acc": {}, "per_language_clc": {}}

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")

    # Compute cross-lingual consistency
    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [overlapping_ratio(per_lang_results[lang]["correct_indices"], per_lang_results[other]["correct_indices"]) for other in others]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency
        
    results["overall_clc"] = sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
    
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')
    for lang in langs:
        print(f'  {lang} cross-lingual consistency: {results["per_language_clc"][lang]:.2%}')
        
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results

save_path = f"filter_knowns/{model_name}_results.json"
evaluate(model, dataset, tokenizer, max_new_tokens=10, n_shot=3, save_path=save_path)
