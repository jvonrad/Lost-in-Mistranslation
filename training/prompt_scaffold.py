"""Native-language free-form prompt scaffold.

Base (non-instruct) models are pure continuation engines, so wrapping a Bengali
question in an English "Question:/Answer text:" scaffold creates code-switched
context that biases continuation toward English — the model answers in the wrong
language and then fails option matching, which scores as a knowledge failure
when it is really a scaffold artifact. This module localises the whole scaffold.

Scope: the FREE-FORM (generation) path only — GRPO rollouts and the periodic
`polyfact/freeform_*` eval. Deliberately NOT applied to:
  * the log-likelihood MCQ path (`build_prompt_eval`, "Question: {q}\\nAnswer:"),
    which stays English to remain comparable with evaluate_accuracy.py, lm-eval
    conventions and every previously reported number. The scaffold is identical
    across the 4 candidates there anyway, so its effect is far smaller.
  * evaluate_klar.py, which is already fully native — its templates come from
    klar_root/<lang>/<rel>.json and contain no English scaffold at all.

The option letters stay Latin A–D in every language: they are the MCQ labels the
reward matcher keys on, and mixed-script letter labels are not a convention any
of these models will have seen.

TRANSLATION PROVENANCE: these strings were written by Claude, not by native
speakers or a professional translation pass. They are grammatical but should be
spot-checked before they appear in a paper — Bengali and Swahili especially.
Set --prompt_scaffold en to fall back to the previous English scaffold for an
apples-to-apples ablation.
"""

from typing import Dict

# Per language: intro, only_text, no_letter, no_explain, question_label, answer_label
SCAFFOLD: Dict[str, Dict[str, str]] = {
    "en": {
        "intro": "You will be given one factual multiple-choice question in English.",
        "only_text": "Return only the full answer text in English.",
        "no_letter": "Do not return the letter.",
        "no_explain": "Do not explain.",
        "question_label": "Question:",
        "answer_label": "Answer text:",
    },
    "de": {
        "intro": "Sie erhalten eine Multiple-Choice-Wissensfrage auf Deutsch.",
        "only_text": "Geben Sie nur den vollständigen Antworttext auf Deutsch zurück.",
        "no_letter": "Geben Sie nicht den Buchstaben an.",
        "no_explain": "Erklären Sie nicht.",
        "question_label": "Frage:",
        "answer_label": "Antworttext:",
    },
    "es": {
        "intro": "Recibirás una pregunta factual de opción múltiple en español.",
        "only_text": "Devuelve únicamente el texto completo de la respuesta en español.",
        "no_letter": "No devuelvas la letra.",
        "no_explain": "No expliques.",
        "question_label": "Pregunta:",
        "answer_label": "Texto de la respuesta:",
    },
    "fr": {
        "intro": "Vous recevrez une question factuelle à choix multiples en français.",
        "only_text": "Renvoyez uniquement le texte complet de la réponse en français.",
        "no_letter": "Ne renvoyez pas la lettre.",
        "no_explain": "N'expliquez pas.",
        "question_label": "Question :",
        "answer_label": "Texte de la réponse :",
    },
    "pt": {
        "intro": "Você receberá uma pergunta factual de múltipla escolha em português.",
        "only_text": "Retorne apenas o texto completo da resposta em português.",
        "no_letter": "Não retorne a letra.",
        "no_explain": "Não explique.",
        "question_label": "Pergunta:",
        "answer_label": "Texto da resposta:",
    },
    "id": {
        "intro": "Anda akan diberikan satu pertanyaan faktual pilihan ganda dalam bahasa Indonesia.",
        "only_text": "Kembalikan hanya teks jawaban lengkap dalam bahasa Indonesia.",
        "no_letter": "Jangan kembalikan hurufnya.",
        "no_explain": "Jangan menjelaskan.",
        "question_label": "Pertanyaan:",
        "answer_label": "Teks jawaban:",
    },
    "ru": {
        "intro": "Вам будет дан один фактологический вопрос с вариантами ответа на русском языке.",
        "only_text": "Верните только полный текст ответа на русском языке.",
        "no_letter": "Не возвращайте букву.",
        "no_explain": "Не объясняйте.",
        "question_label": "Вопрос:",
        "answer_label": "Текст ответа:",
    },
    "ar": {
        "intro": "سيتم إعطاؤك سؤال معرفي واحد من متعدد الخيارات باللغة العربية.",
        "only_text": "أعد نص الإجابة الكامل باللغة العربية فقط.",
        "no_letter": "لا تعد الحرف.",
        "no_explain": "لا تشرح.",
        "question_label": "السؤال:",
        "answer_label": "نص الإجابة:",
    },
    "bn": {
        "intro": "আপনাকে বাংলায় একটি তথ্যভিত্তিক বহুনির্বাচনী প্রশ্ন দেওয়া হবে।",
        "only_text": "শুধুমাত্র বাংলায় সম্পূর্ণ উত্তরের পাঠ্য ফেরত দিন।",
        "no_letter": "অক্ষরটি ফেরত দেবেন না।",
        "no_explain": "ব্যাখ্যা করবেন না।",
        "question_label": "প্রশ্ন:",
        "answer_label": "উত্তরের পাঠ্য:",
    },
    "sw": {
        "intro": "Utapewa swali moja la maarifa lenye machaguo mengi kwa Kiswahili.",
        "only_text": "Rudisha maandishi kamili ya jibu kwa Kiswahili pekee.",
        "no_letter": "Usirudishe herufi.",
        "no_explain": "Usitoe maelezo.",
        "question_label": "Swali:",
        "answer_label": "Maandishi ya jibu:",
    },
    "ja": {
        "intro": "日本語で書かれた事実に関する多肢選択問題が1問与えられます。",
        "only_text": "日本語で答えの本文のみを返してください。",
        "no_letter": "記号は返さないでください。",
        "no_explain": "説明はしないでください。",
        "question_label": "質問:",
        "answer_label": "答えの本文:",
    },
    "zh": {
        "intro": "你将获得一道用中文表述的事实性多项选择题。",
        "only_text": "只需用中文返回完整的答案文本。",
        "no_letter": "不要返回字母。",
        "no_explain": "不要解释。",
        "question_label": "问题:",
        "answer_label": "答案文本:",
    },
}

# Every answer label, longest first, so a model echoing the label in ANY language
# gets it stripped before option matching. Without this the native scaffold would
# depress freeform_resolution_rate purely by making the model echo a prefix the
# old English-only regex did not know about.
ANSWER_LABELS = sorted(
    {v["answer_label"].rstrip(": ：").strip() for v in SCAFFOLD.values()}
    | {"Answer", "Answer text"},
    key=len,
    reverse=True,
)


def strip_answer_label(text: str) -> str:
    """Remove a leading answer label in ANY of the 12 languages (or English).

    The old extract_answer_text only knew the English 'Answer:' / 'Answer text:'
    prefixes. Under a native scaffold the model echoes the native label instead,
    which would leave e.g. '答案文本: 北京' unmatchable against the options and
    silently depress freeform accuracy/resolution.
    """
    t = text.lstrip()
    for label in ANSWER_LABELS:                     # longest-first
        if t.startswith(label):
            rest = t[len(label):].lstrip()
            # Only treat it as a label if a separator followed it, otherwise
            # "Rome" would lose a prefix that is genuinely part of the answer.
            if rest[:1] in (":", "：", "-", "—"):
                return rest[1:].lstrip()
            if t[len(label):][:1] in (" ", "\n", "\t"):
                return rest
    return t


def build_single_language_prompt(
    lang: str, question: str, options: Dict[str, str], scaffold: str = "native",
    task_format: str = "mcq",
) -> str:
    """Free-form prompt. scaffold='native' localises it, 'en' keeps English.

    task_format:
      'mcq'      — list the four candidates (the original behaviour). Trains
                   SELECTION AMONG SHOWN OPTIONS.
      'freeform' — closed-book: no candidate list, the model must recall the
                   answer. This is the task KLAR actually measures, and the
                   mismatch is a leading suspect for why GRPO transfers weakly
                   to generation while CM-Align (which samples free text) does
                   not. The reward matcher is unchanged — it still resolves the
                   generated string against the four options — so only the
                   PROMPT differs, not the scoring.
    """
    s = SCAFFOLD["en"] if scaffold == "en" else SCAFFOLD.get(lang, SCAFFOLD["en"])
    if task_format == "freeform":
        # Drop the option list, the "do not return the letter" instruction
        # (meaningless with no letters on screen), and the intro — every
        # localised intro says "multiple-choice question", which is now false.
        return (
            f"{s['only_text']}\n"
            f"{s['no_explain']}\n\n"
            f"{s['question_label']} {question}\n\n"
            f"{s['answer_label']}"
        )
    return (
        f"{s['intro']}\n"
        f"{s['only_text']}\n"
        f"{s['no_letter']}\n"
        f"{s['no_explain']}\n\n"
        f"{s['question_label']} {question}\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n\n"
        f"{s['answer_label']}"
    )
