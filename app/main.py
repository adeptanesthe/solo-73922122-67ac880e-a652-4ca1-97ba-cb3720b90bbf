# -*- coding: utf-8 -*-
"""FastAPI 应用：推荐队列的条件管理、异步重查、分组揭示与演示页面。"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import database
from .annotation_model import (
    LABEL_NEEDS_CHECK,
    LABEL_NO_CHECK,
    all_sentences,
    label_counts,
    sentence_lookup,
    threshold_met,
)
from .articles_data import ARTICLES, TOPICS
from .tfidf_model import get_model
from .worker import get_training_worker, get_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recommender")

DEFAULT_INTEREST = "我想了解人工智能模型训练与数据驱动的分析方法"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class ConditionIn(BaseModel):
    interest: str | None = Field(default=None, description="兴趣短文")
    quotas: dict[str, int] | None = Field(default=None, description="各主题精确篇数")
    compute_delay_ms: int = Field(default=0, ge=0, le=30000)


class ExclusionIn(BaseModel):
    article_id: int = Field(ge=1)
    compute_delay_ms: int = Field(default=0, ge=0, le=30000)


class AnnotationChange(BaseModel):
    article_id: int = Field(ge=1)
    sentence_index: int = Field(ge=0)
    # needs_check / no_check 新增或更正；null 表示撤销该句标注
    label: str | None = Field(default=None)


class AnnotationsIn(BaseModel):
    revision: int = Field(ge=0, description="提交所基于的标注修订号")
    changes: list[AnnotationChange] = Field(min_length=1, max_length=200)


class TrainIn(BaseModel):
    revision: int = Field(ge=0, description="提交所基于的标注修订号")
    simulate_failure: bool = Field(
        default=False, description="演示开关：让本次训练失败以验证失败状态"
    )


def _annotation_payload() -> dict:
    """标注区完整数据：待标句按原文顺序 + 标注 + 修订号 + 最近训练任务。

    训练前不暴露任何概率；概率只出现在 done 状态且修订号匹配的训练结果里。
    """
    ann_state = database.get_annotation_state()
    label_rows = {
        (item["article_id"], item["sentence_index"]): item
        for item in ann_state["labels"]
    }
    sentences = []
    for sentence in all_sentences():
        key = (sentence["article_id"], sentence["sentence_index"])
        item = label_rows.get(key)
        label = item["label"] if item else None
        sentences.append(
            {
                **sentence,
                "label": label,
                "label_display": (
                    "需核查" if label == LABEL_NEEDS_CHECK
                    else "无需核查" if label == LABEL_NO_CHECK
                    else None
                ),
                "updated_at": item["updated_at"] if item else None,
            }
        )
    counts = label_counts(
        {
            (item["article_id"], item["sentence_index"]): item["label"]
            for item in ann_state["labels"]
        }
    )
    job = database.get_latest_training_job()
    # 训练结果只有在“已完成且基于当前修订号”时才算当前建议
    suggestions = None
    stale_result = False
    if job and job["status"] == "done" and job["result"]:
        if job["base_revision"] == ann_state["revision"]:
            suggestions = job["result"]
        else:
            stale_result = True
    return {
        "revision": ann_state["revision"],
        "updated_at": ann_state["updated_at"],
        "sentences": sentences,
        "counts": counts,
        "threshold": {"needs_check": 2, "no_check": 2, "met": threshold_met(counts)},
        "training": {
            "job": job,
            "suggestions": suggestions,
            "stale_result": stale_result,
            "suggestions_based_on_revision": (
                job["base_revision"] if suggestions is not None else None
            ),
        },
    }

def _validate_quotas(quotas: dict[str, int]) -> None:
    unknown = set(quotas) - set(TOPICS)
    if unknown:
        raise HTTPException(422, f"未知主题: {sorted(unknown)}")
    missing = set(TOPICS) - set(quotas)
    if missing:
        raise HTTPException(422, f"缺少主题配额: {sorted(missing)}")
    if any(v < 0 for v in quotas.values()):
        raise HTTPException(422, "配额不能为负数")
    if sum(quotas.values()) != 6:
        raise HTTPException(422, f"六个主题配额之和必须为 6，当前为 {sum(quotas.values())}")


def _validate_interest(interest: str) -> None:
    if not interest or not interest.strip():
        raise HTTPException(422, "兴趣短文不能为空")
    if len(interest) > 2000:
        raise HTTPException(422, "兴趣短文过长（上限 2000 字）")


def _article_brief(a: dict) -> dict:
    return {"id": a["id"], "title": a["title"], "topic": a["topic"], "author": a["author"]}


def _status_payload() -> dict:
    state = database.get_state()
    job = database.get_latest_job()
    plan = None
    if job and job["status"] == "done" and job["generation"] == state["generation"]:
        plan = job["result"]
    return {
        "state": state,
        "job": job,
        "plan": plan,
        "topics": TOPICS,
        "articles": [_article_brief(a) for a in ARTICLES],
        "annotation_revision": database.get_annotation_revision(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    # 预热 TF-IDF（仅文章拟合），失败不影响启动，worker 内会再试
    try:
        get_model()
    except Exception:  # pragma: no cover
        log.exception("TF-IDF 预热失败")
    get_worker().start()
    get_training_worker().start()
    try:
        yield
    finally:
        get_training_worker().stop()
        get_worker().stop()


app = FastAPI(title="技术编辑选题推荐系统", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/annotations")
def annotations_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "annotations.html"))


@app.get("/api/status")
def status() -> dict:
    return _status_payload()


@app.get("/api/articles/{article_id}")
def article_detail(article_id: int) -> dict:
    for a in ARTICLES:
        if a["id"] == article_id:
            return a
    raise HTTPException(404, "文章不存在")


@app.put("/api/condition")
def update_condition(body: ConditionIn) -> dict:
    """修改兴趣和/或配额，开启新世代并触发异步重算。"""
    current = database.get_state()
    new_quotas = body.quotas if body.quotas is not None else current["quotas"]
    _validate_quotas(new_quotas)
    new_interest = body.interest if body.interest is not None else current["interest"]
    _validate_interest(new_interest)

    # 与已揭示前缀的兼容性前置检查（详细不可行原因由重算任务给出）
    from .optimizer import solve_queue  # 局部导入避免循环
    model = get_model()
    check = solve_queue(
        relevance=model.relevance_vector(new_interest),
        quotas=new_quotas,
        prefix=current["revealed"],
        excluded_ids=set(current["excluded_ids"]),
    )
    if not check.feasible:
        raise HTTPException(409, f"新条件与已揭示前缀不兼容: {check.reason}")

    database.begin_new_generation(
        interest=new_interest,
        quotas=new_quotas,
        compute_delay_ms=body.compute_delay_ms,
    )
    return _status_payload()


@app.post("/api/exclusions")
def add_exclusion(body: ExclusionIn) -> dict:
    if not any(a["id"] == body.article_id for a in ARTICLES):
        raise HTTPException(404, "文章不存在")
    try:
        database.add_exclusion(body.article_id, body.compute_delay_ms)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _status_payload()


@app.delete("/api/exclusions/{article_id}")
def remove_exclusion(article_id: int, compute_delay_ms: int = 0) -> dict:
    try:
        database.remove_exclusion(article_id, compute_delay_ms)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _status_payload()


@app.post("/api/reveal")
def reveal() -> dict:
    """揭示下一组两篇；同条件同进度下并发只推进一次。"""
    try:
        outcome = database.reveal_next_two()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    payload = _status_payload()
    payload["reveal_outcome"] = outcome
    return payload


@app.post("/api/reset")
def reset_demo() -> dict:
    """重置演示：清空揭示、排除、标注与任务，世代回到 1。"""
    database.reset_all(DEFAULT_INTEREST, {t: 1 for t in TOPICS})
    return _status_payload()


# ---------------- 医学事实核查标注区 ----------------


@app.get("/api/annotations")
def get_annotations() -> dict:
    return _annotation_payload()


@app.post("/api/annotations")
def submit_annotations(body: AnnotationsIn) -> dict:
    """提交标注修改；revision 必须是提交者看到的当前修订号，否则冲突。"""
    lookup = sentence_lookup()
    changes = []
    for change in body.changes:
        key = (change.article_id, change.sentence_index)
        if key not in lookup:
            raise HTTPException(
                404, f"句子不存在于医疗健康语料: 文章 {change.article_id} "
                     f"第 {change.sentence_index} 句"
            )
        if change.label not in (None, LABEL_NEEDS_CHECK, LABEL_NO_CHECK):
            raise HTTPException(422, f"未知标注: {change.label}")
        changes.append(
            {
                "article_id": change.article_id,
                "sentence_index": change.sentence_index,
                "label": change.label,
                "text": lookup[key]["text"],
            }
        )
    try:
        database.submit_annotations(body.revision, changes, set(lookup))
    except database.RevisionConflictError as exc:
        raise HTTPException(
            409,
            {
                "code": "revision_conflict",
                "message": str(exc),
                "expected_revision": exc.expected,
                "current_revision": exc.current,
            },
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _annotation_payload()


@app.post("/api/train")
def train(body: TrainIn) -> dict:
    """基于已确认标注训练分类器并推荐下一轮待核查句。"""
    current_rev = database.get_annotation_revision()
    if body.revision != current_rev:
        raise HTTPException(
            409,
            {
                "code": "revision_conflict",
                "message": (
                    f"标注修订号冲突：提交基于修订 {body.revision}，"
                    f"当前已是修订 {current_rev}；请刷新后重试"
                ),
                "expected_revision": body.revision,
                "current_revision": current_rev,
            },
        )
    labels = database.get_annotations()
    counts = label_counts(labels)
    if not threshold_met(counts):
        raise HTTPException(
            409,
            f"标注数量不足：需核查与无需核查各至少 2 句，"
            f"当前为 {counts['needs_check']} / {counts['no_check']}",
        )
    latest = database.get_latest_training_job()
    if latest and latest["status"] in ("pending", "running"):
        raise HTTPException(409, "已有训练任务正在进行，请稍候")
    try:
        job = database.create_training_job(current_rev, body.simulate_failure)
    except database.RevisionConflictError as exc:
        raise HTTPException(
            409,
            {
                "code": "revision_conflict",
                "message": str(exc),
                "expected_revision": exc.expected,
                "current_revision": exc.current,
            },
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    payload = _annotation_payload()
    payload["training"]["job"] = job
    return payload
