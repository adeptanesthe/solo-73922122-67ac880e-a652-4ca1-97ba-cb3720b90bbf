/* 医学事实核查标注区前端（原生 JS） */
"use strict";

const $ = (sel) => document.querySelector(sel);
const app = {
  data: null,          // 最近一次 /api/annotations 响应
  revision: 0,         // 当前已知服务端修订号
  pollTimer: null,
  toastTimer: null,
};

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = body.detail;
    const err = new Error(
      typeof detail === "string"
        ? detail
        : (detail && detail.message) || `请求失败 (${resp.status})`
    );
    err.status = resp.status;
    err.detail = detail;
    throw err;
  }
  return body;
}

function toast(text, kind = "ok") {
  const el = $("#toast");
  el.textContent = text;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(app.toastTimer);
  app.toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
}

/* ---------- 渲染 ---------- */

function renderCounts() {
  const c = app.data.counts;
  $("#count-needs").textContent = c.needs_check;
  $("#count-no").textContent = c.no_check;
  $("#count-unlabeled").textContent = c.unlabeled;
  const met = app.data.threshold.met;
  const el = $("#threshold-msg");
  el.className = `threshold-msg ${met ? "met" : "not-met"}`;
  el.textContent = met
    ? "已满足训练条件：两类各至少 2 句"
    : `两类各至少需要 2 句才能训练（需核查还差 ${Math.max(0, 2 - c.needs_check)} 句，` +
      `无需核查还差 ${Math.max(0, 2 - c.no_check)} 句）`;
  $("#btn-train").disabled = !met;
  $("#revision-badge").textContent = `修订 ${app.data.revision}`;
}

function renderSuggestions() {
  const box = $("#suggestions");
  const t = app.data.training;
  const job = t.job;
  box.innerHTML = "";

  const statusEl = $("#train-status");
  if (!job) {
    statusEl.className = "train-status";
    statusEl.textContent = "尚未训练。完成两类各 2 句标注后点击训练。";
  } else if (job.status === "pending" || job.status === "running") {
    statusEl.className = "train-status running";
    statusEl.textContent = `训练任务 #${job.id} ${job.status === "pending" ? "排队中" : "训练中"}…` +
      (job.base_revision === app.data.revision
        ? ""
        : `（该任务基于修订 ${job.base_revision}，当前已是修订 ${app.data.revision}，结果将作废）`);
  } else if (job.status === "failed") {
    statusEl.className = "train-status failed";
    statusEl.textContent =
      `训练失败（任务 #${job.id}，基于修订 ${job.base_revision}）：` +
      `${job.error || "未知错误"}；已确认标签已保留，可修正后重试。`;
  } else if (job.status === "stale") {
    statusEl.className = "train-status stale";
    statusEl.textContent =
      `训练任务 #${job.id} 已作废：训练期间标注被更正或已有更新的训练，结果不可用作当前建议。`;
  } else if (job.status === "done") {
    if (t.stale_result) {
      statusEl.className = "train-status stale";
      statusEl.textContent =
        `最近一次成功训练（任务 #${job.id}）基于修订 ${job.base_revision}，` +
        `当前已是修订 ${app.data.revision}：标注已更正，旧建议不再作为当前结果。请重新训练。`;
    } else {
      const m = t.suggestions && t.suggestions.model;
      statusEl.className = "train-status done";
      statusEl.textContent = m
        ? `训练完成（任务 #${job.id}）：${m.training_sentences} 句标注、` +
          `${m.feature_count} 个字符 2~3 元特征；本轮推荐 ${t.suggestions.suggestions.length} 句。`
        : `训练完成（任务 #${job.id}）。`;
    }
  }

  if (!t.suggestions) {
    if (job && job.status === "done" && t.stale_result) {
      box.innerHTML = `<p class="hint">旧建议已失效，不在此展示；请基于最新标注重新训练。</p>`;
    }
    return;
  }

  const list = t.suggestions.suggestions;
  if (!list.length) {
    box.innerHTML = `<p class="hint">已没有未标注句，本轮无推荐。</p>`;
    return;
  }
  const head = document.createElement("p");
  head.className = "hint";
  head.textContent =
    "以下句子按预测概率最接近 0.5（最不确定）排序，同篇最多一条；点击卡片可直接标注。";
  box.appendChild(head);
  list.forEach((s) => {
    const card = document.createElement("div");
    card.className = "suggestion-card";
    card.innerHTML = `
      <div class="sc-head">
        <span class="sc-article">#${s.article_id} ${escapeHtml(s.article_title)}</span>
        <span class="sc-prob">概率 ${s.probability.toFixed(4)}</span>
      </div>
      <div class="sc-text">${escapeHtml(s.text)}</div>
      <div class="sc-actions">
        <button class="btn btn-small needs" data-label="needs_check">需核查</button>
        <button class="btn btn-small no" data-label="no_check">无需核查</button>
      </div>`;
    card.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () =>
        submitChanges([{
          article_id: s.article_id,
          sentence_index: s.sentence_index,
          label: btn.dataset.label,
        }], "已标注推荐句")
      );
    });
    box.appendChild(card);
  });
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderSentences() {
  const onlyUnlabeled = $("#only-unlabeled").checked;
  const filter = $("#sentence-filter").value.trim().toLowerCase();
  const container = $("#article-blocks");
  container.innerHTML = "";

  const byArticle = new Map();
  app.data.sentences.forEach((s) => {
    if (!byArticle.has(s.article_id)) {
      byArticle.set(s.article_id, {
        id: s.article_id,
        title: s.article_title,
        author: s.author,
        sentences: [],
      });
    }
    byArticle.get(s.article_id).sentences.push(s);
  });

  let shown = 0;
  byArticle.forEach((article) => {
    const visible = article.sentences.filter((s) => {
      if (onlyUnlabeled && s.label) return false;
      if (!filter) return true;
      return `${article.title} ${s.text}`.toLowerCase().includes(filter);
    });
    if (!visible.length) return;

    const block = document.createElement("div");
    block.className = "article-block";
    block.innerHTML = `
      <div class="ab-head">
        <span>#${article.id} ${escapeHtml(article.title)}</span>
        <span class="tag author">${escapeHtml(article.author)}</span>
      </div>`;

    visible.forEach((s) => {
      shown += 1;
      const row = document.createElement("div");
      row.className = `sentence-row ${s.label ? "labeled " + s.label : ""}`;
      row.innerHTML = `
        <div class="sr-text">${escapeHtml(s.text)}</div>
        <div class="sr-actions">
          <button class="btn btn-small needs" data-label="needs_check">需核查</button>
          <button class="btn btn-small no" data-label="no_check">无需核查</button>
          <button class="btn btn-small ghost undo" data-label="">撤销</button>
          <span class="sr-state">${s.label_display || "未标注"}</span>
        </div>`;
      row.querySelectorAll("button[data-label]").forEach((btn) => {
        const label = btn.dataset.label || null;
        if (label !== null && label === s.label) {
          btn.classList.add("active");
        }
        btn.addEventListener("click", () => {
          if (label === null && !s.label) return;
          submitChanges([{
            article_id: s.article_id,
            sentence_index: s.sentence_index,
            label,
          }], label ? "标注已保存" : "已撤销标注");
        });
      });
      block.appendChild(row);
    });
    container.appendChild(block);
  });

  if (!shown) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = onlyUnlabeled
      ? "所有句子都已标注，取消勾选“仅显示未标注句”可查看全部。"
      : "没有匹配的句子。";
    container.appendChild(empty);
  }
}

function render() {
  renderCounts();
  renderSuggestions();
  renderSentences();
}

/* ---------- 交互 ---------- */

async function submitChanges(changes, successText) {
  // 防冲突：每次提交都带上当前已知修订号；服务端不一致则拒绝
  try {
    const data = await api("/api/annotations", {
      method: "POST",
      body: JSON.stringify({ revision: app.revision, changes }),
    });
    app.data = data;
    app.revision = data.revision;
    render();
    toast(successText, "ok");
  } catch (err) {
    if (err.status === 409 && err.detail && err.detail.code === "revision_conflict") {
      toast(
        `提交冲突：你基于修订 ${err.detail.expected_revision}，` +
        `服务端已是修订 ${err.detail.current_revision}。已刷新为最新标注，请确认后再提交。`,
        "error"
      );
      await refresh({ keepMessage: true });
    } else {
      toast(err.message, "error");
    }
  }
}

async function train() {
  const msg = $("#train-msg");
  msg.textContent = "";
  try {
    $("#btn-train").disabled = true;
    await api("/api/train", {
      method: "POST",
      body: JSON.stringify({
        revision: app.revision,
        simulate_failure: $("#force-fail").checked,
      }),
    });
    $("#force-fail").checked = false;
    toast("训练任务已提交", "ok");
    schedulePoll();
    await refresh();
  } catch (err) {
    if (err.status === 409 && err.detail && err.detail.code === "revision_conflict") {
      toast(
        `训练请求冲突：你基于修订 ${err.detail.expected_revision}，` +
        `服务端已是修订 ${err.detail.current_revision}。已刷新，请重新训练。`,
        "error"
      );
      await refresh();
    } else {
      msg.textContent = err.message;
      msg.className = "msg";
    }
  } finally {
    renderCounts();
  }
}

async function refresh(opts = {}) {
  const data = await api("/api/annotations");
  app.data = data;
  app.revision = data.revision;
  render();
  schedulePoll();
  if (!opts.keepMessage) {
    const msg = $("#train-msg");
    msg.textContent = "";
  }
}

function schedulePoll() {
  clearTimeout(app.pollTimer);
  const job = app.data && app.data.training.job;
  if (job && (job.status === "pending" || job.status === "running")) {
    app.pollTimer = setTimeout(() => refresh().catch(() => {}), 500);
  }
}

async function resetDemo() {
  if (!confirm("确定重置全部演示数据？标注、训练记录、推荐进度都会被清空。")) return;
  await api("/api/reset", { method: "POST" });
  await refresh();
  toast("已重置", "ok");
}

$("#btn-train").addEventListener("click", train);
$("#btn-reset").addEventListener("click", resetDemo);
$("#only-unlabeled").addEventListener("change", renderSentences);
$("#sentence-filter").addEventListener("input", renderSentences);

refresh().catch((err) => toast(`加载失败：${err.message}`, "error"));
