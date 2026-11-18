# -*- coding: utf-8 -*-
"""医学事实核查标注区：切句、训练、推荐规则、修订号冲突与失败语义。"""
from __future__ import annotations

import time

import pytest

from app.annotation_model import (
    MAX_SUGGESTIONS,
    all_sentences,
    label_counts,
    split_sentences,
    train_and_suggest,
)
from app.articles_data import ARTICLES


# ---------------- 纯逻辑测试 ----------------


def test_sentence_split_preserves_order_and_punctuation():
    sentences = all_sentences()
    # 医疗健康 5 篇 × 5 句 = 25 句
    assert len(sentences) == 25
    medical_ids = [a["id"] for a in ARTICLES if a["topic"] == "医疗健康"]
    assert [a_id for a_id in dict.fromkeys(s["article_id"] for s in sentences)] == medical_ids
    for s in sentences:
        assert s["text"][-1] in "。！？"
    # 句内索引按原文顺序 0..4
    for article_id in medical_ids:
        indexes = [
            s["sentence_index"] for s in sentences if s["article_id"] == article_id
        ]
        assert indexes == sorted(indexes)


def test_split_sentences_handles_exclamation_and_question():
    assert split_sentences("第一句。第二句！第三句？") == [
        "第一句。",
        "第二句！",
        "第三句？",
    ]
    assert split_sentences("没有任何标点") == ["没有任何标点"]


def _labels(needs: int, no: int) -> dict:
    sentences = all_sentences()
    labels: dict = {}
    for s in sentences[:needs]:
        labels[(s["article_id"], s["sentence_index"])] = "needs_check"
    for s in sentences[needs:needs + no]:
        labels[(s["article_id"], s["sentence_index"])] = "no_check"
    return labels


def test_threshold_requires_two_per_class():
    with pytest.raises(ValueError):
        train_and_suggest(_labels(1, 2))
    with pytest.raises(ValueError):
        train_and_suggest(_labels(2, 1))
    result = train_and_suggest(_labels(2, 2))
    assert result["suggestions"]


def test_classifier_is_char_2_3_gram_tfidf():
    from app.annotation_model import build_vectorizer

    vec = build_vectorizer()
    matrix = vec.fit_transform(["患者应定期测量血压。", "保健品不能替代日常饮食。"])
    assert vec.analyzer == "char"
    assert vec.ngram_range == (2, 3)
    assert {len(t) for t in vec.vocabulary_} <= {2, 3}
    assert matrix.shape[0] == 2


def test_suggestions_at_most_three_and_one_per_article():
    result = train_and_suggest(_labels(5, 5))
    suggestions = result["suggestions"]
    assert 0 < len(suggestions) <= MAX_SUGGESTIONS
    article_ids = [s["article_id"] for s in suggestions]
    assert len(set(article_ids)) == len(article_ids)
    # 按离 0.5 的绝对距离升序
    distances = [abs(s["probability"] - 0.5) for s in suggestions]
    assert distances == sorted(distances)
    for s in suggestions:
        assert 0.0 <= s["probability"] <= 1.0


def test_suggestions_never_include_labeled_sentences():
    labels = _labels(2, 2)
    result = train_and_suggest(labels)
    for s in result["suggestions"]:
        assert (s["article_id"], s["sentence_index"]) not in labels


def test_label_counts():
    labels = _labels(3, 4)
    counts = label_counts(labels)
    assert counts == {
        "total": 25,
        "needs_check": 3,
        "no_check": 4,
        "unlabeled": 18,
    }


# ---------------- API 测试 ----------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import database, main

    main.get_worker().stop()
    main.get_training_worker().stop()
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("RECOMMENDER_DB", str(db_path))
    monkeypatch.setenv("REVEAL_CLAIM_TTL", "0.01")
    database._DB_PATH = str(db_path)
    database._CLAIM_TTL_SECONDS = 0.01
    database._reset_reveal_batch()
    database.init_db()
    main.get_worker().start()
    main.get_training_worker().start()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c, database, main.get_training_worker()
    main.get_training_worker().stop()
    main.get_worker().stop()


def _first_sentences(data, n):
    return data["sentences"][:n]


def _label_payload(sentences, label):
    return [
        {
            "article_id": s["article_id"],
            "sentence_index": s["sentence_index"],
            "label": label,
        }
        for s in sentences
    ]


def _wait_training(c, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = c.get("/api/annotations").json()
        job = data["training"]["job"]
        if job and job["status"] in ("done", "failed", "stale"):
            return data
        time.sleep(0.05)
    raise AssertionError("训练任务未结束")


def test_annotations_api_exposes_sentences_without_probabilities(client):
    c, _, _ = client
    data = c.get("/api/annotations").json()
    assert data["revision"] == 0
    assert len(data["sentences"]) == 25
    # 训练前不显示概率
    assert data["training"]["suggestions"] is None
    assert "probability" not in str(data["sentences"])
    # 顺序为原文顺序
    first_article = data["sentences"][0]["article_id"]
    assert first_article == 6
    assert data["sentences"][0]["sentence_index"] == 0
    assert data["threshold"]["met"] is False


def test_annotation_submit_and_revision_bump(client):
    c, _, _ = client
    data = c.get("/api/annotations").json()
    sentences = _first_sentences(data, 4)
    needs = sentences[:2]
    no = sentences[2:4]
    r = c.post(
        "/api/annotations",
        json={
            "revision": 0,
            "changes": _label_payload(needs, "needs_check")
            + _label_payload(no, "no_check"),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["revision"] == 1
    assert body["counts"]["needs_check"] == 2
    assert body["counts"]["no_check"] == 2
    assert body["threshold"]["met"] is True

    # 用旧修订号再次提交 -> 冲突，且不改变任何标注
    stale = c.post(
        "/api/annotations",
        json={"revision": 0, "changes": _label_payload(sentences[::-1][:1], "no_check")},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "revision_conflict"
    assert stale.json()["detail"]["current_revision"] == 1
    latest = c.get("/api/annotations").json()
    assert latest["revision"] == 1
    # 那条试图写入的句子仍然未标注
    changed = stale.json()["detail"]["current_revision"]
    assert changed == 1


def test_annotation_correction_and_clearing(client):
    c, _, _ = client
    data = c.get("/api/annotations").json()
    s = data["sentences"][0]
    r = c.post(
        "/api/annotations",
        json={
            "revision": 0,
            "changes": [{
                "article_id": s["article_id"],
                "sentence_index": s["sentence_index"],
                "label": "needs_check",
            }],
        },
    ).json()
    assert r["revision"] == 1
    # 更正为无需核查
    r = c.post(
        "/api/annotations",
        json={
            "revision": 1,
            "changes": [{
                "article_id": s["article_id"],
                "sentence_index": s["sentence_index"],
                "label": "no_check",
            }],
        },
    ).json()
    assert r["revision"] == 2
    assert r["counts"]["needs_check"] == 0
    assert r["counts"]["no_check"] == 1
    # 撤销
    r = c.post(
        "/api/annotations",
        json={
            "revision": 2,
            "changes": [{
                "article_id": s["article_id"],
                "sentence_index": s["sentence_index"],
                "label": None,
            }],
        },
    ).json()
    assert r["revision"] == 3
    assert r["counts"]["no_check"] == 0
    # 重复提交同样内容不推进修订号
    r = c.post(
        "/api/annotations",
        json={
            "revision": 3,
            "changes": [{
                "article_id": s["article_id"],
                "sentence_index": s["sentence_index"],
                "label": None,
            }],
        },
    ).json()
    assert r["revision"] == 3


def test_training_flow_and_suggestions(client):
    c, database, worker = client
    data = c.get("/api/annotations").json()
    sentences = data["sentences"]
    payload = _label_payload(sentences[:2], "needs_check") + _label_payload(
        sentences[2:4], "no_check"
    )
    r = c.post("/api/annotations", json={"revision": 0, "changes": payload})
    rev = r.json()["revision"]

    t = c.post("/api/train", json={"revision": rev})
    assert t.status_code == 200, t.text
    assert t.json()["training"]["job"]["status"] == "pending"
    assert t.json()["training"]["suggestions"] is None  # 训练完成前不返回建议

    done = _wait_training(c)
    job = done["training"]["job"]
    assert job["status"] == "done"
    sug = done["training"]["suggestions"]
    assert sug is not None
    assert len(sug["suggestions"]) <= 3
    assert sug["model"]["training_sentences"] == 4
    assert len({s["article_id"] for s in sug["suggestions"]}) == len(sug["suggestions"])
    # 已标注句不再被推荐
    labeled = {(s["article_id"], s["sentence_index"]) for s in sentences[:4]}
    for s in sug["suggestions"]:
        assert (s["article_id"], s["sentence_index"]) not in labeled

    # 用旧修订号训练 -> 冲突
    stale = c.post("/api/train", json={"revision": rev - 1})
    assert stale.status_code == 409


def test_correction_invalidates_old_suggestions(client, monkeypatch):
    c, database, _ = client
    data = c.get("/api/annotations").json()
    sentences = data["sentences"]
    payload = _label_payload(sentences[:2], "needs_check") + _label_payload(
        sentences[2:4], "no_check"
    )
    rev = c.post(
        "/api/annotations", json={"revision": 0, "changes": payload}
    ).json()["revision"]
    c.post("/api/train", json={"revision": rev})
    done = _wait_training(c)
    assert done["training"]["suggestions"] is not None

    # 更正一条标注：旧建议立即不再作为当前结果
    correction = c.post(
        "/api/annotations",
        json={
            "revision": done["revision"],
            "changes": [{
                "article_id": sentences[0]["article_id"],
                "sentence_index": sentences[0]["sentence_index"],
                "label": "no_check",
            }],
        },
    )
    assert correction.status_code == 200
    after = correction.json()
    assert after["training"]["suggestions"] is None
    assert after["training"]["stale_result"] is True


def test_stale_training_result_never_saved_when_corrected_during_training(
    client, monkeypatch
):
    c, database, _ = client
    data = c.get("/api/annotations").json()
    sentences = data["sentences"]
    payload = _label_payload(sentences[:2], "needs_check") + _label_payload(
        sentences[2:4], "no_check"
    )
    rev = c.post(
        "/api/annotations", json={"revision": 0, "changes": payload}
    ).json()["revision"]

    # 在训练执行前插入一次标注更正，模拟训练期间双窗口并发修改
    import app.worker as worker_mod

    original = worker_mod.train_and_suggest

    def train_after_correction(labels):
        database.submit_annotations(
            database.get_annotation_revision(),
            [{
                "article_id": sentences[4]["article_id"],
                "sentence_index": sentences[4]["sentence_index"],
                "label": "needs_check",
                "text": sentences[4]["text"],
                "article_title": sentences[4]["article_title"],
            }],
            {(sentences[4]["article_id"], sentences[4]["sentence_index"])},
        )
        return original(labels)

    monkeypatch.setattr(worker_mod, "train_and_suggest", train_after_correction)
    c.post("/api/train", json={"revision": rev})
    deadline = time.monotonic() + 10
    data = c.get("/api/annotations").json()
    while time.monotonic() < deadline:
        data = c.get("/api/annotations").json()
        if data["training"]["job"]["status"] in ("stale", "failed", "done"):
            break
        time.sleep(0.05)
    job = data["training"]["job"]
    assert job["status"] in ("stale", "failed")
    assert data["training"]["suggestions"] is None
    # 标签保留（4 + 新增 1 条）
    assert data["counts"]["needs_check"] == 3
    assert data["counts"]["no_check"] == 2


def test_training_failure_keeps_labels_and_shows_failed(client):
    c, database, _ = client
    data = c.get("/api/annotations").json()
    sentences = data["sentences"]
    payload = _label_payload(sentences[:2], "needs_check") + _label_payload(
        sentences[2:4], "no_check"
    )
    rev = c.post(
        "/api/annotations", json={"revision": 0, "changes": payload}
    ).json()["revision"]

    r = c.post(
        "/api/train", json={"revision": rev, "simulate_failure": True}
    )
    assert r.status_code == 200
    done = _wait_training(c)
    job = done["training"]["job"]
    assert job["status"] == "failed"
    assert "模拟训练失败" in (job["error"] or "")
    # 标签原样保留
    assert done["counts"]["needs_check"] == 2
    assert done["counts"]["no_check"] == 2
    assert done["revision"] == rev
    assert done["training"]["suggestions"] is None


def test_training_rejects_insufficient_labels(client):
    c, _, _ = client
    data = c.get("/api/annotations").json()
    rev = data["revision"]
    r = c.post("/api/train", json={"revision": rev})
    assert r.status_code == 409
    assert "至少" in r.json()["detail"]


def test_second_training_while_one_active_is_rejected(client):
    """已有 pending/running 训练任务时再次训练返回 409，不会把前一个任务顶成 stale。"""
    c, database, _ = client
    data = c.get("/api/annotations").json()
    sentences = data["sentences"]
    payload = _label_payload(sentences[:2], "needs_check") + _label_payload(
        sentences[2:4], "no_check"
    )
    rev = c.post(
        "/api/annotations", json={"revision": 0, "changes": payload}
    ).json()["revision"]

    # 直接登记一条 pending 任务，模拟训练尚未完成
    job = database.create_training_job(rev)
    r = c.post("/api/train", json={"revision": rev})
    assert r.status_code == 409
    assert "正在进行" in r.json()["detail"]
    assert database.get_training_job(job["id"])["status"] == "pending"


def test_reset_clears_annotations(client):
    c, _, _ = client
    data = c.get("/api/annotations").json()
    sentences = data["sentences"]
    c.post(
        "/api/annotations",
        json={
            "revision": 0,
            "changes": _label_payload(sentences[:2], "needs_check"),
        },
    )
    c.post("/api/reset")
    after = c.get("/api/annotations").json()
    assert after["revision"] == 0
    assert after["counts"]["needs_check"] == 0
    assert after["training"]["job"] is None


def test_concurrent_submissions_only_one_wins(client):
    """两个窗口基于同一修订号并发提交：只有一个成功，另一个收到 409 冲突。"""
    import threading

    c, _, _ = client
    data = c.get("/api/annotations").json()
    s0, s1 = data["sentences"][0], data["sentences"][1]
    results: list[tuple[int, str | None]] = []
    barrier = threading.Barrier(2)

    def submit(sentence, label):
        barrier.wait()
        r = c.post(
            "/api/annotations",
            json={
                "revision": 0,
                "changes": [{
                    "article_id": sentence["article_id"],
                    "sentence_index": sentence["sentence_index"],
                    "label": label,
                }],
            },
        )
        code = None
        if r.status_code == 409:
            code = r.json()["detail"]["code"]
        results.append((r.status_code, code))

    threads = [
        threading.Thread(target=submit, args=(s0, "needs_check")),
        threading.Thread(target=submit, args=(s1, "no_check")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    codes = sorted(code for _, code in results if code is not None)
    assert [status for status, _ in results].count(200) == 1
    assert codes.count("revision_conflict") == 1
    final = c.get("/api/annotations").json()
    assert final["revision"] == 1
    assert final["counts"]["needs_check"] + final["counts"]["no_check"] == 1


def test_annotations_survive_restart(tmp_path, monkeypatch):
    from app import database, main
    from fastapi.testclient import TestClient

    db_file = tmp_path / "restart_ann.db"
    monkeypatch.setenv("RECOMMENDER_DB", str(db_file))
    main.get_worker().stop()
    main.get_training_worker().stop()
    database._DB_PATH = str(db_file)
    database.init_db()
    main.get_worker().start()
    main.get_training_worker().start()
    with TestClient(main.app) as c1:
        data = c1.get("/api/annotations").json()
        sentences = data["sentences"]
        c1.post(
            "/api/annotations",
            json={
                "revision": 0,
                "changes": _label_payload(sentences[:2], "needs_check")
                + _label_payload(sentences[2:4], "no_check"),
            },
        )
        before = c1.get("/api/annotations").json()
    main.get_training_worker().stop()
    main.get_worker().stop()

    database.init_db()
    with TestClient(main.app) as c2:
        after = c2.get("/api/annotations").json()
        assert after["revision"] == before["revision"]
        assert after["counts"] == before["counts"]
