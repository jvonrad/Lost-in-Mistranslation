from transformers import AutoTokenizer
import numpy as np

# -------------------------------------------------
# Change this to test different tokenizers
# -------------------------------------------------
#MODEL_NAME = "CohereLabs/aya-23-8B"
# MODEL_NAME = "allenai/OLMo-2-1124-7B"
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
# -------------------------------------------------

REQ_LANGS = {
    "en": "This is a simple test sentence to evaluate tokenizer efficiency.",
    "de": "Dies ist ein einfacher Testsatz zur Bewertung der Tokenisierung.",
    "te": "ఇది టోకెనైజర్ పనితీరును పరీక్షించడానికి ఒక సాధారణ వాక్యం.",
    "th": "นี่คือประโยคทดสอบง่าย ๆ เพื่อประเมินประสิทธิภาพของตัวตัดคำ",
    "bn": "এটি টোকেনাইজারের কার্যকারিতা মূল্যায়নের জন্য একটি সাধারণ বাক্য।",
    "sw": "Hii ni sentensi rahisi ya kupima ufanisi wa tokenaiza.",
    "es": "Esta es una oración simple para evaluar la eficiencia del tokenizador.",
    "ru": "Это простое тестовое предложение для оценки эффективности токенизации.",
    "fr": "Ceci est une phrase simple pour évaluer l'efficacité du tokeniseur.",
    "ja": "これはトークナイザーの効率を評価するための簡単な文です。",
    "zh-cn": "这是一句用于评估分词器效率的简单句子。"
}

# NLLB language code mapping
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "te": "tel_Telu",
    "th": "tha_Thai",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "zh-cn": "zho_Hans"
}

print(f"Loading tokenizer: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("\n=== TOKENIZER DIAGNOSTIC ===\n")

for lang, text in REQ_LANGS.items():

    # Special handling for NLLB
    if "nllb" in MODEL_NAME.lower():
        tok.src_lang = NLLB_LANG_MAP[lang]
        enc = tok(text, add_special_tokens=False)
    else:
        enc = tok(text, add_special_tokens=False)

    tokens = enc["input_ids"]
    decoded_tokens = [tok.decode([t]) for t in tokens]

    total_tokens = len(tokens)
    total_chars = len(text)
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0

    token_lengths = [len(t) for t in decoded_tokens]
    avg_token_length = np.mean(token_lengths)
    single_char_ratio = sum(l == 1 for l in token_lengths) / total_tokens

    print(f"{lang.upper():7} | "
          f"tokens={total_tokens:3d} | "
          f"chars/token={chars_per_token:5.2f} | "
          f"avg_token_len={avg_token_length:5.2f} | "
          f"%single-char={single_char_ratio*100:5.1f}")