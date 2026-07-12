# KLAR-CLC

This repository provides the dataset for:

>[Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](https://www.arxiv.org/pdf/2504.04264)

## 💡 Introduction

We apply mechanistic interpretability methods to analyze cross-lingual inconsistencies in multilingual masked language models (MLMs). Our findings reveal that MLMs represent knowledge in a shared, language-independent space through most layers, transitioning to language-specific representations only in the final layers. Errors often occur during this transition, leading to incorrect predictions in the target language despite correct answers in others. These insights offer a lightweight and effective strategy for improving factual consistency across languages.

This repository hosts KLAR, the dataset we created for multilingual knowledge probing, covering 17 languages. The accompanying code is available [here](https://github.com/cisnlp/KLAR-CLC).


## 📙 Citation
If you found our work useful for your research, please cite it as follows:

```latex

@inproceedings{wang-etal-2025-lost-multilinguality,
    title = "Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models",
    author = {Wang, Mingyang and Adel, Heike and Lange, Lukas and Liu, Yihong and Nie, Ercong and Strötgen, Jannik and Schütze, Hinrich},
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.253/",
}
```

