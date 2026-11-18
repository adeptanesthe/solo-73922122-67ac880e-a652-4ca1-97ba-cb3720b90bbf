# -*- coding: utf-8 -*-
"""医疗健康文章切句与“需核查 / 无需核查”二分类器。

* 只处理主题为「医疗健康」的内置文章；
* 按中文句末标点 。！？ 切句，保留标点，句子顺序与原文一致；
* 分类器只在已确认标注上拟合字符 2~3 元 TF-IDF，未标注句只做 transform；
* 每轮推荐全局预测概率最接近 0.5 的至多 3 条未标句，且同一篇最多一条。
"""
from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .articles_data import ARTICLES

MEDICAL_TOPIC = "医疗健康"
LABEL_NEEDS_CHECK = "needs_check"
LABEL_NO_CHECK = "no_check"
LABELS = (LABEL_NEEDS_CHECK, LABEL_NO_CHECK)
MIN_PER_CLASS = 2
MAX_SUGGESTIONS = 3

_SPLIT_RE = re.compile(r"(?<=[。！？])")


def split_sentences(body: str) -> list[str]:
    """按 。！？ 切句（标点保留在句尾），去掉空白与空句。"""
    parts = [p.strip() for p in _SPLIT_RE.split(body or "")]
    return [p for p in parts if p]


def medical_articles() -> list[dict]:
    """内置医疗健康主题文章，保持原始顺序。"""
    return [a for a in ARTICLES if a["topic"] == MEDICAL_TOPIC]


def all_sentences() -> list[dict]:
    """全部医疗健康句子，按文章原顺序、句内原顺序排列。"""
    result: list[dict] = []
    for article in medical_articles():
        for index, text in enumerate(split_sentences(article["body"])):
            result.append(
                {
                    "article_id": article["id"],
                    "article_title": article["title"],
                    "author": article["author"],
                    "sentence_index": index,
                    "text": text,
                }
            )
    return result


def sentence_lookup() -> dict[tuple[int, int], dict]:
    return {
        (s["article_id"], s["sentence_index"]): s for s in all_sentences()
    }


def label_counts(labels: dict[tuple[int, int], str]) -> dict[str, int]:
    counter = Counter(labels.values())
    total = len(all_sentences())
    needs = counter.get(LABEL_NEEDS_CHECK, 0)
    no = counter.get(LABEL_NO_CHECK, 0)
    return {
        "total": total,
        "needs_check": needs,
        "no_check": no,
        "unlabeled": total - needs - no,
    }


def threshold_met(counts: dict[str, int]) -> bool:
    return (
        counts["needs_check"] >= MIN_PER_CLASS
        and counts["no_check"] >= MIN_PER_CLASS
    )


def build_vectorizer() -> TfidfVectorizer:
    """字符 2~3 元 TF-IDF 向量化器；只在已确认标注句上 fit。"""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 3),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )


def train_and_suggest(labels: dict[tuple[int, int], str]) -> dict:
    """在已确认标注上训练二分类器，并返回本轮推荐。

    训练/预测异常直接抛出，由后台 worker 记录为 failed（标签本身不动）。
    """
    counts = label_counts(labels)
    if not threshold_met(counts):
        raise ValueError(
            "两类标注均至少需要 2 句："
            f"当前需核查 {counts['needs_check']} 句、"
            f"无需核查 {counts['no_check']} 句"
        )

    sentences = all_sentences()
    train_texts: list[str] = []
    train_labels: list[str] = []
    unlabeled: list[dict] = []
    for sentence in sentences:
        key = (sentence["article_id"], sentence["sentence_index"])
        label = labels.get(key)
        if label in LABELS:
            train_texts.append(sentence["text"])
            train_labels.append(label)
        else:
            unlabeled.append(sentence)

    # 字符 2~3 元 TF-IDF：只在已确认标注句上 fit。
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(train_texts)
    feature_count = len(vectorizer.vocabulary_)
    if matrix.shape[1] == 0:
        raise ValueError("标注文本没有产生可用的字符 2~3 元特征")

    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=0,
    )
    classifier.fit(matrix, train_labels)
    needs_index = list(classifier.classes_).index(LABEL_NEEDS_CHECK)

    suggestions: list[dict] = []
    if unlabeled:
        unlabeled_matrix = vectorizer.transform([s["text"] for s in unlabeled])
        probabilities = classifier.predict_proba(unlabeled_matrix)[:, needs_index]

        # 同篇最多一条：先取每篇概率最接近 0.5 的未标句。
        best_by_article: dict[int, dict] = {}
        for sentence, probability in zip(unlabeled, probabilities):
            p = float(probability)
            item = {
                "article_id": sentence["article_id"],
                "article_title": sentence["article_title"],
                "sentence_index": sentence["sentence_index"],
                "text": sentence["text"],
                "probability": p,
            }
            current = best_by_article.get(sentence["article_id"])
            if current is None or abs(p - 0.5) < abs(current["probability"] - 0.5):
                best_by_article[sentence["article_id"]] = item

        # 全局再按“离 0.5 的绝对距离”升序（同距离按文章/句序确定性排序）。
        ranked = sorted(
            best_by_article.values(),
            key=lambda item: (
                abs(item["probability"] - 0.5),
                item["article_id"],
                item["sentence_index"],
            ),
        )
        suggestions = ranked[:MAX_SUGGESTIONS]

    return {
        "suggestions": suggestions,
        "model": {
            "labels": counts,
            "training_sentences": len(train_texts),
            "feature_count": feature_count,
            "algorithm": "char 2~3-gram TF-IDF + logistic regression",
            "max_suggestions": MAX_SUGGESTIONS,
        },
    }
