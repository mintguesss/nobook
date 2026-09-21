/* 課堂即時轉錄前端（規格 §8）。
 *
 * 重點：
 *  - AudioWorklet 採集 16kHz PCM16，每 250ms 一包，binary frame 上傳（§4.1）
 *  - 斷線時繼續錄音並在本地 buffer 累積（上限 60 秒），指數退避重連（§5.3）
 *  - wakeLock 必須拿到，且在 visibilitychange 後重新申請（§8.2）
 */
'use strict';

const SR = 16000;
const CHUNK_SAMPLES = 4000;              // 250ms
const BUFFER_LIMIT = 60 / 0.25;          // 本地最多留 60 秒
const BACKOFF = [1000, 2000, 4000, 8000];
const PING_MS = 20000;
const LONG_PRESS_MS = 500;

const $ = (id) => document.getElementById(id);
const el = {
  dot: $('dot'), elapsed: $('elapsed'), course: $('course'), health: $('health'),
  transcript: $('transcript'), toLatest: $('to-latest'), summaries: $('summaries'),
  note: $('note'), mark: $('mark'), pause: $('pause'), end: $('end'),
  toast: $('toast'), overlay: $('overlay'), start: $('start'),
  courseSelect: $('course-select'),
  final: $('final'), finalMd: $('final-md'), finalTitle: $('final-title'),
  history: $('history'), historyList: $('history-list'),
  notes: $('notes'), copyView: $('copy-view'), copyBody: $('copy-body'),
  sumBar: $('sumbar'), sumCount: $('sumcount'), sumToggle: $('sum-toggle'),
  copyCount: $('copy-count'),
  detail: $('detail'), detailBody: $('detail-body'), detailTitle: $('detail-title'),
};

const state = {
  ws: null, sessionId: null, courseId: 'example',
  seq: 0, pending: [],                   // 斷線期間累積的 frame
  running: false, paused: false, ended: false,
  startedAt: 0, elapsedBase: 0,
  autoScroll: true, retries: 0,
  wakeLock: null, ctx: null, node: null, stream: null,
  hiddenAt: 0, gaps: [], viewingSession: null,
  notesBusy: false, notesMd: '', endingTimer: null, finalReady: null,
  sections: new Map(), batches: new Map(), markBusy: false, finalMd: '',
};

// ── 小工具 ────────────────────────────────────────────────────────────
function fmt(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = String(Math.floor(sec / 3600)).padStart(2, '0');
  const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

let toastTimer = null;
function toast(msg, ms = 3200) {
  el.toast.textContent = msg;
  el.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove('show'), ms);
}

function setLight(kind) {
  el.dot.className = 'dot' + (kind === 'ok' ? ' ok' : kind === 'warn' ? ' warn' : '');
}

// ── wake lock（規格 §8.2）─────────────────────────────────────────────
async function acquireWakeLock() {
  if (!('wakeLock' in navigator)) {
    toast('此瀏覽器不支援 wake lock，請手動把螢幕逾時關掉');
    return;
  }
  try {
    state.wakeLock = await navigator.wakeLock.request('screen');
    state.wakeLock.addEventListener('release', () => { state.wakeLock = null; });
  } catch (e) {
    toast('無法保持螢幕喚醒：' + e.message);
  }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    // 頁面被凍結時 AudioWorklet 整個停掉，這段音訊是真的消失，
    // 不像斷線那樣能靠本地 buffer 補送。記下時間點，回來時告訴使用者
    // 逐字稿缺了哪一段——不講的話畫面上完全看不出來。
    if (state.running && !state.ended && !state.paused) state.hiddenAt = Date.now();
    return;
  }
  // 切走再切回 wake lock 會失效，必須重新申請
  if (state.running && !state.wakeLock) acquireWakeLock();
  if (state.hiddenAt) {
    const gap = (Date.now() - state.hiddenAt) / 1000;
    state.hiddenAt = 0;
    if (gap > 1.5) {
      state.gaps.push(gap);
      markGap(gap);
      toast(`錄音中斷了 ${gap.toFixed(0)} 秒（畫面被切走），這段沒有錄到`, 6000);
    }
  }
});

function markGap(seconds) {
  const div = document.createElement('div');
  div.className = 'seg gap';
  div.innerHTML = '<span class="t">⚠</span><span class="x"></span>';
  div.querySelector('.x').textContent =
    `— 錄音中斷約 ${seconds.toFixed(0)} 秒，此處有缺漏 —`;
  el.transcript.appendChild(div);
  if (state.autoScroll) el.transcript.scrollTop = el.transcript.scrollHeight;
}

// ── 錄音 ──────────────────────────────────────────────────────────────
async function startAudio() {
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: SR,
      // echoCancellation / noiseSuppression 是為語音通話設計的，
      // 會削掉教室遠處講者的頻段，反而傷害 ASR 準確度（規格 §4.1）
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: true,
    },
  });
  state.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  await state.ctx.audioWorklet.addModule('recorder-worklet.js');
  const src = state.ctx.createMediaStreamSource(state.stream);
  state.node = new AudioWorkletNode(state.ctx, 'recorder');
  state.node.port.onmessage = (e) => sendChunk(new Int16Array(e.data));
  src.connect(state.node);
  // worklet 必須連到 destination 才會被 pull，但經過 gain=0
  // 免得麥克風訊號從喇叭放出來造成回授
  state.sink = state.ctx.createGain();
  state.sink.gain.value = 0;
  state.node.connect(state.sink);
  state.sink.connect(state.ctx.destination);
}

function stopAudio() {
  if (state.node) { try { state.node.disconnect(); } catch (e) { /* ignore */ } }
  if (state.ctx) { try { state.ctx.close(); } catch (e) { /* ignore */ } }
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.node = state.ctx = state.stream = null;
}

function sendChunk(pcm16) {
  // 8 bytes header：uint32 seq + uint32 sampleCount（小端序，規格 §4.1）
  const frame = new ArrayBuffer(8 + pcm16.byteLength);
  const dv = new DataView(frame);
  dv.setUint32(0, state.seq++, true);
  dv.setUint32(4, pcm16.length, true);
  new Int16Array(frame, 8).set(pcm16);

  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(frame);
  } else {
    // 斷線期間繼續錄音並在本地累積，超過 60 秒丟最舊的（規格 §5.3）
    state.pending.push(frame);
    while (state.pending.length > BUFFER_LIMIT) state.pending.shift();
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────
function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws/session?course_id=${encodeURIComponent(state.courseId)}`;
}

function connect() {
  setLight(state.sessionId ? 'warn' : 'bad');
  const ws = new WebSocket(wsUrl());
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.retries = 0;
    setLight('ok');
    if (state.sessionId) {
      ws.send(JSON.stringify({ type: 'resume', session_id: state.sessionId }));
    } else {
      ws.send(JSON.stringify({
        type: 'start', course_id: state.courseId, client_ts: Date.now(),
      }));
    }
    // 補送斷線期間累積的音訊；伺服端以 seq 去重
    const backlog = state.pending;
    state.pending = [];
    for (const f of backlog) ws.send(f);
    if (backlog.length) toast(`已補送 ${(backlog.length * 0.25).toFixed(0)} 秒錄音`);
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    handleEvent(msg);
  };

  ws.onclose = () => {
    state.ws = null;
    if (state.ended || !state.running) { setLight('bad'); return; }
    setLight('warn');
    const wait = BACKOFF[Math.min(state.retries++, BACKOFF.length - 1)];
    setTimeout(connect, wait);
  };

  ws.onerror = () => { try { ws.close(); } catch (e) { /* ignore */ } };
}

setInterval(() => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'ping' }));
  }
}, PING_MS);

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
    return true;
  }
  toast('目前離線，稍後會自動重連');
  return false;
}

// ── 下行事件（規格 §5.2）──────────────────────────────────────────────
function handleEvent(msg) {
  switch (msg.type) {
    case 'session_started':
      state.sessionId = msg.session_id;
      // 接回既有 session 時，seq 要接在伺服器看過的編號之後。
      // 從 0 開始的話會全部撞上 seen_seqs 被當重複丟掉，
      // 畫面上錄音正常在跑，但一句逐字稿都不會出現。
      if (typeof msg.next_seq === 'number' && msg.next_seq > state.seq) {
        state.seq = msg.next_seq;
      }
      break;
    case 'segment':
      addSegment(msg);
      break;
    case 'summary_pending':
      addPendingSection(msg.section_id);
      break;
    case 'summary':
      addSummary(msg);
      setMarkBusy(false);
      break;
    case 'notes_pending':
      setNotesBusy(true);
      break;
    case 'notes':
      setNotesBusy(false);
      clearEnding();
      showHandcopy(msg.markdown, msg.sections, msg.check);
      break;
    case 'final_summary_pending':
      toast('正在切換到總結模型並生成完整筆記…', 20000);
      break;
    case 'final_summary':
      clearEnding();
      state.viewingSession = msg.session_id;
      // 整理完就回首頁。筆記已存進資料庫，從「歷史筆記」隨時叫得回來；
      // 留在錄完的畫面上沒有任何用處，還會讓人以為要等什麼。
      // 手抄版正開著在抄的話就別打斷。
      if (el.copyView.classList.contains('show')) {
        toast('課後整理完成，已存入歷史筆記', 6000);
      } else {
        goHome();
        toast('課後整理完成，已存入歷史筆記', 6000);
      }
      break;
    case 'stats':
      el.health.textContent =
        `RTF ${msg.rtf.toFixed(2)} · 佇列 ${msg.queue_depth} · VRAM ${Math.round(msg.vram_used_mb)}MB`
        + (msg.summary_degraded ? ' · 摘要降級' : '');
      break;
    case 'error':
      toast(msg.message);
      if (['LLM_UNAVAILABLE', 'VRAM_OOM', 'SUMMARY_FAILED', 'EMPTY_BUFFER'].includes(msg.code)) {
        setMarkBusy(false);
        removePending();
      }
      if (['NOTES_FAILED', 'NO_CONTENT', 'LLM_UNAVAILABLE'].includes(msg.code)) {
        setNotesBusy(false);
      }
      break;
    default:
      break;
  }
}

// ── 逐字稿（規格 §8.1-2）──────────────────────────────────────────────
function addSegment(seg) {
  const div = document.createElement('div');
  div.className = 'seg' + (seg.avg_logprob < -0.9 ? ' low' : '');
  div.dataset.start = seg.start;
  div.innerHTML =
    `<span class="t">${fmt(seg.start)}</span><span class="x"></span>`;
  div.querySelector('.x').textContent = seg.text;
  el.transcript.appendChild(div);
  if (state.autoScroll) el.transcript.scrollTop = el.transcript.scrollHeight;
  else el.toLatest.classList.add('show');
}

el.transcript.addEventListener('scroll', () => {
  const nearBottom = el.transcript.scrollHeight - el.transcript.scrollTop
    - el.transcript.clientHeight < 40;
  state.autoScroll = nearBottom;
  el.toLatest.classList.toggle('show', !nearBottom);
});

el.toLatest.addEventListener('click', () => {
  state.autoScroll = true;
  el.toLatest.classList.remove('show');
  el.transcript.scrollTop = el.transcript.scrollHeight;
});

// ── 摘要側欄（規格 §8.1-4）────────────────────────────────────────────
function addPendingSection(id) {
  if (state.sections.has(id)) return;   // 合併既有段落時不另開一則
  const d = document.createElement('details');
  d.className = 'sec pending';
  d.dataset.pending = '1';
  d.innerHTML = `<summary><span class="ts">--:--:--</span>
    <span class="title"><span class="spin"></span> 生成中…</span></summary>`;
  el.summaries.prepend(d);
}

function removePending() {
  const p = el.summaries.querySelector('.sec[data-pending]');
  if (p) p.remove();
}

// 一次「即時整理」可能切出好幾段（逐字稿太長時），把它們收在同一個
// 容器裡。不然按五次就變成一串二十幾個平鋪的段落，找不到剛剛那次生的。
function batchBox(msg) {
  const id = msg.batch != null ? msg.batch : msg.section_id;
  let box = state.batches.get(id);
  if (box) return box;
  box = document.createElement('details');
  box.className = 'batch';
  box.open = true;
  box.innerHTML = '<summary><span class="bn"></span><span class="bmeta"></span>' +
                  '</summary><div class="bbody"></div>';
  box.querySelector('.bn').textContent = `第 ${state.batches.size + 1} 次整理`;
  box.querySelector('.bmeta').textContent = fmt(msg.start);
  // 新的一批展開，前面的收起來
  for (const b of state.batches.values()) b.open = false;
  el.summaries.prepend(box);
  state.batches.set(id, box);
  return box;
}

function addSummary(msg) {
  removePending();
  const existing = state.sections.get(msg.section_id);
  const d = existing || document.createElement('details');
  d.className = 'sec';
  d.open = !existing;
  d.innerHTML = `<summary>
      <span class="ts" role="button">${fmt(msg.start)}</span>
      <span class="title"></span>
    </summary><div class="secbody"></div>`;
  d.querySelector('.title').textContent = msg.title;
  renderSectionBody(d.querySelector('.secbody'), msg);
  if (msg.user_note) {
    const n = document.createElement('div');
    n.className = 'note';
    n.textContent = '使用者標註：' + msg.user_note;
    d.appendChild(n);
  }
  d.querySelector('.ts').addEventListener('click', (e) => {
    e.preventDefault();
    jumpTo(msg.start);
  });
  if (!existing) {
    const box = batchBox(msg);
    box.querySelector('.bbody').appendChild(d);
    state.sections.set(msg.section_id, d);
    const n = box.querySelectorAll('.sec').length;
    box.querySelector('.bmeta').textContent =
      n > 1 ? `${fmt(msg.start)} · ${n} 段` : fmt(msg.start);
    // 同一批裡只有第一段預設展開，其餘收起來
    d.open = n === 1;
  }
  updateSumBar();
}

// 整理列的計數與「全部收起／全部展開」
function updateSumBar() {
  if (!el.sumBar) return;
  const nb = state.batches.size;
  const ns = state.sections.size;
  el.sumBar.hidden = nb === 0;
  if (!nb) return;
  el.sumCount.textContent =
    ns > nb ? `即時整理 ${nb} 次 · 共 ${ns} 段` : `即時整理 ${nb} 次`;
}

// 縮成一行。原本是「全部收起」＝把每一則 details 關起來，但那樣整理列
// 還是佔著整個下半螢幕；上課時要看的是逐字稿，整理好的東西收掉就好。
function toggleAllSections() {
  const wrap = $('sumwrap');
  if (!wrap) return;
  const folded = wrap.classList.toggle('folded');
  el.sumToggle.title = folded ? '展開即時整理' : '收合即時整理';
  el.sumToggle.setAttribute('aria-label', folded ? '展開' : '收合');
}

// 段落內容：導言 + 子題 + 要點。一段裡二十幾個同一層級的句子不是筆記，
// 所以顯示端也要照著層次走，不能攤平。舊紀錄沒有 groups 就只畫要點。
function renderSectionBody(box, sec) {
  box.innerHTML = '';
  if (sec.summary) {
    const p = document.createElement('p');
    p.className = 'secsum';
    p.textContent = sec.summary;
    box.appendChild(p);
  }
  const groups = (sec.groups && sec.groups.length)
    ? sec.groups
    : [{ heading: '', points: sec.bullets || [] }];
  for (const g of groups) {
    const pts = (g.points || []).filter((x) => x && String(x).trim());
    if (!pts.length) continue;
    if (g.heading) {
      const h = document.createElement('div');
      h.className = 'sechead';
      h.textContent = g.heading;
      box.appendChild(h);
    }
    const ul = document.createElement('ul');
    for (const b of pts) {
      const li = document.createElement('li');
      li.textContent = b;
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
}

function jumpTo(startS) {
  state.autoScroll = false;
  let target = null;
  for (const node of el.transcript.children) {
    if (parseFloat(node.dataset.start) >= startS) { target = node; break; }
  }
  if (!target) target = el.transcript.lastElementChild;
  if (!target) return;
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  target.classList.add('flash');
  setTimeout(() => target.classList.remove('flash'), 1600);
  el.toLatest.classList.add('show');
}

// ── 標記按鈕（規格 §8.1-3、§6.2）──────────────────────────────────────
function setMarkBusy(busy) {
  state.markBusy = busy;
  el.mark.disabled = busy;
  el.mark.textContent = busy ? '整 理 中 …' : '即 時 整 理';
}

function doMark() {
  if (state.markBusy || !state.running || state.ended) return;
  const note = el.note.value.trim();
  // 前端同步進入 disabled：使用者緊張連按三下不應產生三份摘要
  setMarkBusy(true);
  if (!send({ type: 'mark', note: note || undefined })) {
    setMarkBusy(false);
    return;
  }
  el.note.value = '';
  el.note.classList.remove('show');
  if (navigator.vibrate) navigator.vibrate(30);
}

let pressTimer = null;
let longPressed = false;
el.mark.addEventListener('pointerdown', () => {
  longPressed = false;
  pressTimer = setTimeout(() => {
    longPressed = true;
    el.note.classList.add('show');
    el.note.focus();
    if (navigator.vibrate) navigator.vibrate([20, 40, 20]);
  }, LONG_PRESS_MS);
});
const cancelPress = () => clearTimeout(pressTimer);
el.mark.addEventListener('pointerup', () => {
  cancelPress();
  if (!longPressed) doMark();
});
el.mark.addEventListener('pointerleave', cancelPress);
el.mark.addEventListener('pointercancel', cancelPress);
el.note.addEventListener('keydown', (e) => { if (e.key === 'Enter') doMark(); });

// ── 暫停 / 結束 ───────────────────────────────────────────────────────
el.pause.addEventListener('click', () => {
  state.paused = !state.paused;
  if (state.node) state.node.port.postMessage({ type: 'mute', value: state.paused });
  send({ type: state.paused ? 'pause' : 'resume' });
  el.pause.textContent = state.paused ? '繼續錄音' : '暫停';
  if (state.paused) state.elapsedBase += (Date.now() - state.startedAt) / 1000;
  else state.startedAt = Date.now();
  toast(state.paused ? '已暫停錄音' : '已繼續錄音');
});

el.end.addEventListener('click', () => {
  if (!state.running) return;
  if (!confirm('結束課程並生成完整筆記？')) return;
  state.ended = true;
  send({ type: 'end' });
  stopAudio();
  if (state.wakeLock) { state.wakeLock.release(); state.wakeLock = null; }
  el.mark.disabled = true;
  el.pause.disabled = true;
  el.notes.disabled = true;
  showEnding();
});

// 結束後的等待畫面。整理在伺服器上跑，關掉頁面也會跑完並存進資料庫——
// 這件事一定要講，否則使用者會不敢離開，乾等一段不知道多久的時間。
function showEnding() {
  const t0 = Date.now();
  el.copyView.classList.remove('show');
  el.finalTitle.textContent = '整理中';
  el.finalMd.textContent = '';
  el.final.classList.add('show');
  const box = el.finalMd;
  const tick = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    box.textContent = [
      `正在整理這堂課的筆記…（已經過 ${s} 秒）`,
      '',
      '步驟：切換到較大的模型（約 5 秒）→ 產生手抄版 → 產生完整版',
      '通常 20～30 秒完成。',
      '',
      '★ 可以直接關掉這個畫面或切到別的 App。',
      '　 整理是在筆電上跑的，關掉頁面也會跑完，',
      '　 結果會存進資料庫，之後從「歷史筆記」就看得到。',
    ].join('\n');
  }, 500);
  state.endingTimer = tick;
}

function clearEnding() {
  if (state.endingTimer) { clearInterval(state.endingTimer); state.endingTimer = null; }
}

// ── 計時 ──────────────────────────────────────────────────────────────
setInterval(() => {
  if (!state.running || state.paused) return;
  const secs = state.elapsedBase + (Date.now() - state.startedAt) / 1000;
  el.elapsed.textContent = fmt(secs);
}, 500);

// ── 期末筆記 ──────────────────────────────────────────────────────────
function showFinal(md, title, sessionId) {
  state.finalMd = md;
  state.viewingSession = sessionId || state.sessionId;
  el.finalTitle.textContent = title || '完整筆記';
  el.finalMd.textContent = md;
  el.final.classList.add('show');
}

// ── 回首頁 ────────────────────────────────────────────────────────────
// 課程結束後原本卡在結束畫面沒路可走。這裡把狀態整個歸零，
// 回到開始畫面就能再開一堂新的。
function goHome() {
  if (state.running && !state.ended) {
    if (!confirm('錄音還在進行中，確定要離開嗎？\n離開後這堂課不會有課後整理。')) return;
    try { send({ type: 'end' }); } catch (e) { /* ignore */ }
  }
  stopAudio();
  if (state.wakeLock) { try { state.wakeLock.release(); } catch (e) {} state.wakeLock = null; }
  if (state.ws) { try { state.ws.close(); } catch (e) {} state.ws = null; }
  clearEnding();

  state.running = false;
  state.ended = false;
  state.paused = false;
  state.sessionId = null;
  state.seq = 0;
  state.pending = [];
  state.sections.clear();
  state.batches.clear();
  updateSumBar();
  state.elapsedBase = 0;
  state.gaps = [];
  state.hiddenAt = 0;

  el.transcript.innerHTML = '';
  el.summaries.innerHTML = '';
  el.elapsed.textContent = '00:00:00';
  el.health.textContent = 'RTF — · 佇列 — · VRAM —';
  el.course.textContent = '—';
  el.note.value = '';
  el.note.classList.remove('show');
  setLight('bad');
  setMarkBusy(false);
  setNotesBusy(false);
  el.mark.disabled = false;
  el.pause.disabled = false;
  el.pause.textContent = '暫停';
  el.end.disabled = false;
  el.toLatest.classList.remove('show');

  el.final.classList.remove('show');
  el.copyView.classList.remove('show');
  el.detail.classList.remove('show');
  el.history.classList.remove('show');
  el.overlay.classList.remove('hide');
  el.start.disabled = false;
  loadCourses();
}

$('home').addEventListener('click', goHome);
$('final-home').addEventListener('click', goHome);

// ── 手抄版筆記 ────────────────────────────────────────────────────────
// 使用者上課邊聽邊手寫、下課要立刻交，所以必須能隨時叫出「現在就能抄」
// 的版本，而不是等到按下結束課程才生成。
function setNotesBusy(busy) {
  state.notesBusy = busy;
  el.notes.disabled = busy;
  el.notes.textContent = busy ? '整理中…' : '產生筆記';
  const r = $('copy-refresh');
  if (r) { r.disabled = busy; r.textContent = busy ? '整理中…' : '重新產生'; }
}

function requestNotes() {
  if (state.notesBusy || !state.running) return;
  if (send({ type: 'notes' })) setNotesBusy(true);
}

function showHandcopy(md, nSections, check) {
  state.notesMd = md;
  const wrap = el.copyBody;
  wrap.innerHTML = '';
  let ul = null;
  for (const raw of md.split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('<!--')) continue;
    if (line.startsWith('## ')) {
      const h = document.createElement('h2');
      const t = line.slice(3);
      if (t.startsWith('★')) h.className = 'exam';
      h.textContent = t;
      wrap.appendChild(h);
      ul = document.createElement('ul');
      wrap.appendChild(ul);
    } else if (line.startsWith('# ')) {
      const h = document.createElement('h1');
      h.textContent = line.slice(2);
      wrap.appendChild(h);
      ul = null;
    } else if (line.startsWith('- ')) {
      if (!ul) { ul = document.createElement('ul'); wrap.appendChild(ul); }
      const li = document.createElement('li');
      li.textContent = line.slice(2);
      ul.appendChild(li);
    }
  }
  const chars = md.replace(/<!--[\s\S]*?-->/g, '').replace(/\s/g, '').length;
  el.copyCount.textContent = `約 ${chars} 字 · 取自 ${nSections || 0} 段`;
  renderCheck(wrap, check);
  el.copyView.classList.add('show');
}

// 正確性提示。刻意放在筆記**後面**而不是前面——它是輔助判斷，
// 不該擋在要抄的內容前面。
function renderCheck(wrap, check) {
  if (!check) return;
  const un = check.ungrounded || [];
  const low = check.low_confidence || [];
  if (!un.length && !low.length) return;

  const box = document.createElement('div');
  box.className = 'checkbox';
  const h = document.createElement('h2');
  h.className = 'exam';
  h.textContent = '⚠ 這幾點建議自己確認';
  box.appendChild(h);

  if (un.length) {
    const p = document.createElement('p');
    p.className = 'cnote';
    p.textContent = '這幾點冒出了逐字稿裡沒有的內容，可能是模型自己補的：';
    box.appendChild(p);
    const ul = document.createElement('ul');
    for (const u of un) {
      const li = document.createElement('li');
      li.textContent = u.claim;
      if (u.novel && u.novel.length) {
        const sp = document.createElement('span');
        sp.className = 'novel';
        sp.textContent = ` ← 逐字稿沒有「${u.novel.join('」「')}」`;
        li.appendChild(sp);
      }
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  if (low.length) {
    const p = document.createElement('p');
    p.className = 'cnote';
    p.textContent = '這幾段語音辨識信心較低，內容可能有誤：';
    box.appendChild(p);
    const ul = document.createElement('ul');
    for (const l of low) {
      const li = document.createElement('li');
      li.textContent = `[${fmt(l.start_s)}] ${l.text}…`;
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  const foot = document.createElement('p');
  foot.className = 'cnote dim';
  foot.textContent = check.note || '';
  box.appendChild(foot);
  wrap.appendChild(box);
}

// ── 歷史筆記（規格 §10 的資料都在，只是原本沒有介面看）────────────────
async function openHistory() {
  el.history.classList.add('show');
  showBuild();
  el.historyList.innerHTML = '';
  showUsage();
  let rows;
  try {
    rows = await (await fetch('/api/sessions?limit=100')).json();
  } catch (e) {
    toast('讀取歷史紀錄失敗：' + e.message);
    return;
  }
  const courses = new Map();
  try {
    for (const c of await (await fetch('/api/courses')).json()) courses.set(c.id, c.name);
  } catch (e) { /* 課名拿不到就顯示 id */ }

  await renderMergeGroups(courses);

  // 照課名分資料夾。一門課上十八週就是十八筆，平鋪的話找上禮拜那堂
  // 要滑很久；而且同一門課的紀錄本來就該放在一起看。
  const byCourse = new Map();
  for (const r of rows) {
    const name = courses.get(r.course_id) || r.course_id;
    if (!byCourse.has(name)) byCourse.set(name, []);
    byCourse.get(name).push(r);
  }
  // 最近上過的課排前面
  const order = [...byCourse.entries()].sort(
    (a, b) => (b[1][0].started_at || '').localeCompare(a[1][0].started_at || ''));

  for (const [name, list] of order) {
    const box = document.createElement('details');
    box.className = 'cfold';
    box.innerHTML = '<summary><span class="cname"></span>' +
                    '<span class="cmeta"></span></summary>';
    box.querySelector('.cname').textContent = name;
    const latest = new Date(list[0].started_at);
    box.querySelector('.cmeta').textContent =
      `${list.length} 堂 · 最近 ` +
      latest.toLocaleDateString('zh-TW', { month: 'numeric', day: 'numeric' });
    // 有沒結束的課就自動展開，那是使用者現在需要處理的
    if (list.some((r) => !r.ended_at)) box.open = true;

    for (const r of list) box.appendChild(historyRow(r, name));
    el.historyList.appendChild(box);
  }
}

// 一列紀錄。課名已經在資料夾標題上，這裡只寫日期時間與長度。
function historyRow(r, courseName) {
  const btn = document.createElement('button');
  btn.className = 'hrow';
  const started = new Date(r.started_at);
  const dur = r.duration_s ? `${Math.round(r.duration_s / 60)} 分鐘` : '未完成';
  const done = !!r.ended_at;
  btn.innerHTML =
    '<span class="hmain"><span class="hcourse"></span>' +
    '<span class="hmeta"></span></span>' +
    `<span class="hbadge${done ? ' done' : ''}"></span>`;
  btn.querySelector('.hcourse').textContent =
    started.toLocaleString('zh-TW', { dateStyle: 'short', timeStyle: 'short' });
  btn.querySelector('.hmeta').textContent = dur;
  btn.querySelector('.hbadge').textContent = done ? '已完成' : '未結束';
  btn.addEventListener('click', () => openDetail(r.id, courseName));

  const live = state.running && !state.ended && r.id === state.sessionId;
  if (live) btn.querySelector('.hbadge').textContent = '錄音中';

  const del = document.createElement('button');
  del.className = 'hdel';
  del.textContent = '刪除';
  del.disabled = live;          // 正在錄的那堂不給刪
  del.addEventListener('click', async (ev) => {
    ev.stopPropagation();
    if (!confirm(`刪除「${courseName} ${started.toLocaleDateString('zh-TW')}」？
逐字稿、摘要、錄音都會一起刪掉，無法復原。`)) return;
    try {
      const res = await fetch(`/api/sessions/${r.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      toast('已刪除');
      openHistory();
    } catch (e) { toast('刪除失敗：' + e.message); }
  });

  const wrap = document.createElement('div');
  wrap.className = 'hitem';
  wrap.appendChild(btn);
  wrap.appendChild(del);
  return wrap;
}

// 投影片對照：把逐字稿對到教材的頁碼。
// 涵蓋率通常不高（實測一堂 21%）——老師講投影片以外的東西、Q&A、
// 純圖片的頁都對不上。所以這裡把涵蓋率直接寫出來，不要讓使用者
// 以為沒列到的部分是漏掉了。
function renderAlign(id, d, addSec, a) {
  const has = a && a.pages && a.pages.length;
  const sec = addSec('投影片對照',
                     has ? null : '還沒對照過，或這堂沒有對到任何投影片');

  const row = document.createElement('div');
  row.className = 'row';
  const go = document.createElement('button');
  go.textContent = has ? '重新對照' : '對照投影片';
  go.addEventListener('click', async () => {
    go.disabled = true;
    go.textContent = '對照中…';
    try {
      const res = await fetch(`/api/sessions/${id}/align`, { method: 'POST' });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail || ('HTTP ' + res.status));
      toast(`對到 ${j.pages.length} 段、涵蓋 ${Math.round(j.coverage * 100)}%`);
      openDetail(id, el.detailTitle.textContent.split(' · ')[0]);
    } catch (e) {
      toast('對照失敗：' + e.message);
      go.disabled = false;
      go.textContent = has ? '重新對照' : '對照投影片';
    }
  });
  row.appendChild(go);
  sec.appendChild(row);
  if (!has) return;

  const note = document.createElement('p');
  note.className = 'cnote';
  note.textContent =
    `涵蓋 ${Math.round(a.coverage * 100)}%（${a.matched}/${a.chunks} 段）。` +
    '沒列到的時間是老師講投影片以外的內容，或那幾頁只有圖。';
  sec.appendChild(note);

  for (const m of a.materials || []) {
    const h = document.createElement('div');
    h.className = 'sechead';
    h.textContent = `${m.material}（${m.pages} 頁）`;
    sec.appendChild(h);
    const ul = document.createElement('ul');
    ul.className = 'plist';
    for (const e of a.pages.filter((x) => x.material_id === m.id)) {
      const li = document.createElement('li');
      li.innerHTML = '<button class="pjump"></button><span class="pt"></span>';
      li.querySelector('.pjump').textContent = 'p' + e.page;
      li.querySelector('.pt').textContent =
        `${fmt(e.start_s)}–${fmt(e.end_s)}　${e.score.toFixed(2)}`;
      // 詳細頁裡沒有即時逐字稿可以跳，要跳到對應的那一則課中整理
      li.querySelector('.pjump').addEventListener('click', () => jumpToNote(e.start_s));
      ul.appendChild(li);
    }
    sec.appendChild(ul);
  }
}

// 接回一個還沒結束的 session：把已經有的逐字稿與段落倒回畫面，
// 再用 resume 接上 WebSocket 繼續錄。
// 不倒回內容的話畫面會是空的，看起來像重新開了一堂新的課。
async function resumeSession(d) {
  const sid = d.session.id;
  try {
    await startAudio();
  } catch (e) {
    alert('無法取得麥克風：' + e.message);
    return;
  }
  await acquireWakeLock();

  el.detail.classList.remove('show');
  el.history.classList.remove('show');
  el.overlay.classList.add('hide');

  state.sessionId = sid;
  state.courseId = d.session.course_id;
  state.running = true;
  state.ended = false;
  state.paused = false;
  state.sections.clear();
  state.batches.clear();
  el.summaries.innerHTML = '';
  el.transcript.innerHTML = '';
  updateSumBar();

  el.course.textContent = d.courseName || d.session.course_id;
  // 經過的時間照原本的開始時間算，不是從現在重新計時
  state.startedAt = new Date(d.session.started_at).getTime();

  for (const g of d.segments) {
    addSegment({ start: g.start_s, text: g.text,
                 avg_logprob: g.avg_logprob });
  }
  for (const sc of d.sections) {
    addSummary({
      section_id: sc.seq || sc.id, batch: sc.seq || sc.id,
      title: sc.title, bullets: sc.bullets, summary: sc.summary,
      groups: sc.groups, start: sc.start_s, end: sc.end_s,
      user_note: sc.user_note,
    });
  }
  toast(`接回 ${d.segments.length} 段逐字稿，繼續錄音`);
  connect();
}

// 這段時間對到哪幾頁投影片，寫成「p8–p11」。多份教材就各寫一段。
function slideTag(align, startS, endS) {
  if (!align || !align.pages) return '';
  const byMat = new Map();
  for (const e of align.pages) {
    if (e.end_s <= startS || e.start_s >= endS) continue;
    if (!byMat.has(e.material)) byMat.set(e.material, []);
    byMat.get(e.material).push(e.page);
  }
  if (!byMat.size) return '';
  const parts = [];
  for (const ps of byMat.values()) {
    ps.sort((a, b) => a - b);
    parts.push(ps.length === 1 ? `p${ps[0]}` : `p${ps[0]}–p${ps[ps.length - 1]}`);
  }
  return parts.join('、');
}

// 從投影片對照跳到對應的那一則課中整理（詳細頁裡沒有即時逐字稿）
function jumpToNote(startS) {
  const all = [...el.detailBody.querySelectorAll('.sec[data-start]')];
  if (!all.length) return;
  let target = all[0];
  for (const n of all) {
    if (parseFloat(n.dataset.start) <= startS + 1) target = n;
  }
  const holder = target.closest('details.dsec');
  if (holder) holder.open = true;
  target.open = true;
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  target.classList.add('flash');
  setTimeout(() => target.classList.remove('flash'), 1600);
}

// 版本號。手機／平板的 Service Worker 沒更新時，畫面看起來跟舊版一模一樣，
// 沒有這個就只能猜。對不上就是快取還沒換掉。
async function showBuild() {
  const el2 = $('build');
  if (!el2) return;
  try {
    const t = await (await fetch('sw.js', { cache: 'no-store' })).text();
    const m = t.match(/const VERSION = '([^']+)'/);
    const server = m ? m[1] : '';
    const tag = document.querySelector('script[src*="app.js"]');
    const mine = (tag && (tag.getAttribute('src').split('v=')[1] || '')) || '';
    el2.textContent = server;
    if (mine && server && mine !== server) {
      // 這種狀況畫面看起來完全正常，但按鈕會沒反應——一定要講出來
      el2.textContent = mine + ' ≠ ' + server;
      el2.style.color = 'var(--warn)';
      el2.title = '快取沒更新，把 App 完全關掉再開';
    } else {
      el2.style.color = '';
      el2.title = '';
    }
  } catch (e) { el2.textContent = ''; }
}

// 同一天被拆成好幾段錄的課，合併成一堂重新產生總筆記
async function renderMergeGroups(courses) {
  let groups = [];
  try {
    groups = await (await fetch('/api/merge/groups')).json();
  } catch (e) { return; }
  if (!groups.length) return;
  const box = document.createElement('div');
  box.className = 'mergebox';
  const h = document.createElement('div');
  h.className = 'mergehead';
  h.textContent = '同一天分成好幾段錄的課，可以合併成一堂';
  box.appendChild(h);
  for (const g of groups) {
    const row = document.createElement('div');
    row.className = 'hitem';
    const label = document.createElement('span');
    label.className = 'hmain';
    const t = document.createElement('span');
    t.className = 'mtitle';
    t.textContent = courses.get(g.course_id) || g.course_id;
    const m = document.createElement('span');
    m.className = 'mmeta';
    m.textContent = `${g.date} · ${g.count} 段 · 共 ${Math.round(g.total_s / 60)} 分鐘`;
    label.appendChild(t);
    label.appendChild(m);
    const btn = document.createElement('button');
    btn.className = 'hmerge';
    btn.textContent = '合併';
    btn.addEventListener('click', async () => {
      if (!confirm(`把這 ${g.count} 段接成一堂並重新產生總筆記？
原本那幾筆不會被刪掉，合併後會多一筆新紀錄。
錄音檔也會接起來，過程約三到五分鐘。`)) return;
      btn.disabled = true;
      btn.textContent = '合併中…';
      try {
        const res = await fetch('/api/sessions/merge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ids: g.ids }),
        });
        const j = await res.json();
        if (!res.ok) throw new Error(j.detail || ('HTTP ' + res.status));
        toast(`已合併：${j.sections} 段、${j.chars} 字`);
        openHistory();
      } catch (e) {
        toast('合併失敗：' + e.message);
        btn.disabled = false;
        btn.textContent = '合併';
      }
    });
    row.appendChild(label);
    row.appendChild(btn);
    box.appendChild(row);
  }
  el.historyList.appendChild(box);
}

async function showUsage() {
  const bar = $('usage');
  try {
    const u = await (await fetch('/api/storage')).json();
    const gb = (n) => (n / 1e9).toFixed(1);
    const mb = (n) => Math.round(n / 1e6);
    let txt = `${u.sessions} 堂課 · 資料庫 ${mb(u.db_bytes)} MB`;
    if (u.audio_files) {
      const perHour = u.audio_bytes / Math.max(1, u.audio_files) / 1e6;
      txt += ` · 錄音 ${u.audio_files} 個共 ${mb(u.audio_bytes)} MB`;
    }
    txt += ` · 硬碟剩 ${gb(u.disk_free_bytes)} GB`;
    if (!u.save_audio) txt += '（未開啟錄音保存）';
    bar.textContent = txt;
  } catch (e) { bar.textContent = ''; }
}

// 單堂課的完整紀錄：錄音、課中整理、課後整理、逐字稿放在同一頁
async function openDetail(id, courseName) {
  el.detail.classList.add('show');
  el.detailTitle.textContent = courseName || '課堂紀錄';
  el.detailBody.innerHTML = '<div class="dsec"><span class="empty">載入中…</span></div>';
  let d;
  try {
    d = await (await fetch(`/api/sessions/${id}`)).json();
  } catch (e) {
    el.detailBody.innerHTML = '';
    addSec('讀取失敗', e.message);
    return;
  }
  d.courseName = courseName;
  const started = new Date(d.session.started_at);
  el.detailTitle.textContent =
    `${courseName} · ${started.toLocaleString('zh-TW', { dateStyle: 'short', timeStyle: 'short' })}`;
  el.detailBody.innerHTML = '';

  // 1. 錄音
  if (d.has_audio) {
    const sec = addSec('錄音', null);
    const a = document.createElement('audio');
    a.controls = true;
    a.preload = 'none';
    a.src = `/api/sessions/${id}/audio`;
    sec.appendChild(a);
  } else {
    addSec('錄音', '沒有保存（要保存請在伺服器設 LS_SAVE_AUDIO=1）');
  }

  // 1.5 沒有正常結束的課：給一個把它收尾的出口。
  //     手機睡著或網路斷掉就會留下這種紀錄，逐字稿都在，只差沒收尾。
  //     沒有這顆按鈕的話，那堂課永遠停在「未結束」，既不能繼續也不能結束。
  if (!d.session.ended_at) {
    const sec = addSec('這堂課沒有正常結束',
                       d.resumable
                         ? `逐字稿有 ${d.segments.length} 段。可以接回去繼續錄，`
                           + '或直接收尾產生筆記'
                         : `逐字稿有 ${d.segments.length} 段。伺服器重啟過，`
                           + '接不回去了，只能收尾',
                       null, true);

    // 這兩顆要用詳細頁自己的按鈕樣式（.dbtns）。之前掛的是 .primary，
    // 但那個只在 #overlay 底下有定義，在這裡等於完全沒樣式。
    const bar = document.createElement('div');
    bar.className = 'dbtns';
    sec.appendChild(bar);

    // 手機睡著、網路斷掉之後回來，多半是想把剩下的課錄完，不是想結束。
    // session 還在伺服器的重連佇列裡就接得回去。
    if (d.resumable) {
      const r = document.createElement('button');
      r.className = 'act';
      r.textContent = '接續錄音';
      r.addEventListener('click', () => resumeSession(d));
      bar.appendChild(r);
    }

    const b = document.createElement('button');
    if (!d.resumable) b.className = 'act';
    b.textContent = '結束並產生筆記';
    b.addEventListener('click', async () => {
      b.disabled = true;
      b.textContent = '產生中…（課後模型，約一到三分鐘）';
      try {
        const res = await fetch(`/api/sessions/${id}/finish`, { method: 'POST' });
        const j = await res.json();
        if (!res.ok) throw new Error(j.detail || ('HTTP ' + res.status));
        toast('已收尾');
        openDetail(id, courseName);
      } catch (e) {
        toast('收尾失敗：' + e.message);
        b.disabled = false;
        b.textContent = '結束並產生筆記';
      }
    });
    bar.appendChild(b);
  }

  // 1.8 投影片對照。包 try：任何一節出錯都不該讓後面的筆記整個不渲染
  //（發生過一次——addSec 不在作用域裡，結果詳細頁只剩錄音那一節）
  let align = null;
  try {
    align = await (await fetch(`/api/sessions/${id}/align`)).json();
    if (!align || !align.pages || !align.pages.length) align = null;
  } catch (e) { align = null; }
  try {
    renderAlign(id, d, addSec, align);
  } catch (e) {
    addSec('投影片對照', '載入失敗：' + e.message);
  }

  // 2. 課中整理：每次按「即時整理」產生的段落
  const secWrap = addSec(`課中整理`, d.sections.length ? null : '這堂課沒有按過即時整理',
                         d.sections.length ? `${d.sections.length} 段` : '');
  for (const sc of d.sections) {
    const det = document.createElement('details');
    det.className = 'sec';
    det.innerHTML = '<summary><span class="ts"></span><span class="title"></span>' +
                    '</summary><div class="secbody"></div>';
    det.querySelector('.ts').textContent = fmt(sc.start_s);
    det.querySelector('.title').textContent = sc.title;
    det.dataset.start = sc.start_s;
    // 這一則對到哪幾頁投影片。沒對到就不標——標一個錯的頁碼比不標更糟，
    // 使用者會照著去翻然後發現不是那頁。
    const tag = slideTag(align, sc.start_s, sc.end_s);
    if (tag) {
      const b = document.createElement('span');
      b.className = 'ptag';
      b.textContent = tag;
      det.querySelector('summary').appendChild(b);
    }
    renderSectionBody(det.querySelector('.secbody'), sc);
    if (sc.user_note) {
      const n = document.createElement('div');
      n.className = 'note';
      n.textContent = '標註：' + sc.user_note;
      det.appendChild(n);
    }
    secWrap.appendChild(det);
  }

  // 3. 課後整理
  const hc = d.session.handcopy_md;
  const sec3 = addSec('課後整理 · 手抄版',
                      hc ? null : '這堂課沒有按「結束課程」，所以沒有課後整理');
  if (hc) {
    const pre = document.createElement('pre');
    pre.textContent = hc.replace(/<!--[\s\S]*?-->/g, '').trim();
    sec3.appendChild(pre);
    sec3.appendChild(btnRow([
      ['複製', () => copyText(hc)],
      ['.md', () => dl(id, 'md', 'handcopy')],
      ['Word', () => dl(id, 'docx', 'handcopy')],
    ]));
  }

  const fm = d.session.final_md;
  const sec4 = addSec('課後整理 · 完整版', fm ? null : '同上');
  if (fm) {
    const pre = document.createElement('pre');
    pre.textContent = fm;
    sec4.appendChild(pre);
    sec4.appendChild(btnRow([
      ['複製', () => copyText(fm)],
      ['.md', () => dl(id, 'md', 'full')],
      ['Word', () => dl(id, 'docx', 'full')],
    ]));
  }

  // 4. 逐字稿
  const sec5 = addSec('逐字稿', d.segments.length ? null : '無', `${d.segments.length} 段`);
  if (d.segments.length) {
    const box = document.createElement('div');
    box.className = 'dsegs';
    for (const g of d.segments) {
      const ln = document.createElement('div');
      ln.className = 'ln';
      ln.innerHTML = '<span class="t"></span><span class="x"></span>';
      ln.querySelector('.t').textContent = fmt(g.start_s);
      ln.querySelector('.x').textContent = g.text;
      box.appendChild(ln);
    }
    sec5.appendChild(box);
    sec5.appendChild(btnRow([
      ['.txt', () => dl(id, 'txt', 'transcript')],
      ['Word', () => dl(id, 'docx', 'transcript')],
    ]));
  }

  // 每一節都可以收合，而且預設收起。一堂課的詳細頁有錄音、投影片對照、
  // 課中整理、手抄版、完整版、逐字稿六節，全部攤開要滑很久才找得到
  // 想看的那一節。open=true 只留給「需要你動作」的那種（沒正常結束）。
  function addSec(title, emptyMsg, count, open) {
    const div = document.createElement('details');
    div.className = 'dsec';
    div.open = !!open;
    const h = document.createElement('summary');
    const t = document.createElement('span');
    t.className = 'dtitle';
    t.textContent = title;
    h.appendChild(t);
    if (count) {
      const n = document.createElement('span');
      n.className = 'n';
      n.textContent = count;
      h.appendChild(n);
    }
    div.appendChild(h);
    const body = document.createElement('div');
    body.className = 'dbody';
    div.appendChild(body);
    if (emptyMsg) {
      const e = document.createElement('span');
      e.className = 'empty';
      e.textContent = emptyMsg;
      body.appendChild(e);
    }
    el.detailBody.appendChild(div);
    // 回傳 body：呼叫端 appendChild 的東西要進到可收合的區域裡，
    // 不是跟 summary 同層，否則收起來之後內容還留在畫面上
    return body;
  }
  function btnRow(pairs) {
    const r = document.createElement('div');
    r.className = 'dbtns';
    for (const [label, fn] of pairs) {
      const b = document.createElement('button');
      b.textContent = label;
      b.addEventListener('click', fn);
      r.appendChild(b);
    }
    return r;
  }
}

function dl(id, format, which) {
  location.href = `/api/sessions/${id}/export?format=${format}&which=${which}`;
}

async function copyText(t) {
  try {
    await navigator.clipboard.writeText(t.replace(/<!--[\s\S]*?-->/g, '').trim());
    toast('已複製');
  } catch (e) { toast('複製失敗：' + e.message); }
}
$('final-close').addEventListener('click', () => el.final.classList.remove('show'));
$('copy-md').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(state.finalMd); toast('已複製'); }
  catch (e) { toast('複製失敗：' + e.message); }
});
$('download-md').addEventListener('click', () => {
  const id = state.viewingSession || state.sessionId;
  if (!id) return;
  location.href = `/api/sessions/${id}/export?format=md`;
});
$('sum-toggle').addEventListener('click', toggleAllSections);
$('open-history').addEventListener('click', openHistory);
// 標頭那顆：錄音中也能看紀錄。原本入口只在開始畫面上，一開始錄就被
// 蓋掉了，而標頭唯一的「首頁」按下去是會結束這堂課的。
$('hist-top').addEventListener('click', openHistory);
el.notes.addEventListener('click', requestNotes);
$('copy-close').addEventListener('click', () => el.copyView.classList.remove('show'));
$('copy-refresh').addEventListener('click', requestNotes);
$('copy-dl-md').addEventListener('click', () => {
  const id = state.viewingSession || state.sessionId;
  if (id) dl(id, 'md', 'handcopy');
});
$('copy-docx').addEventListener('click', () => {
  const id = state.viewingSession || state.sessionId;
  if (id) dl(id, 'docx', 'handcopy');
});
$('copy-text').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(state.notesMd.replace(/<!--.*?-->/gs, '').trim());
    toast('已複製');
  } catch (e) { toast('複製失敗：' + e.message); }
});
$('history-close').addEventListener('click', () => el.history.classList.remove('show'));
$('detail-close').addEventListener('click', () => {
  el.detail.classList.remove('show');
  if (el.history.classList.contains('show')) openHistory();
});

// ── 啟動 ──────────────────────────────────────────────────────────────
async function loadCourses() {
  // 連不上時**不要**默默塞一個假的課程進去。原本的寫法會顯示「範例課程」，
  // 看起來像是課程檔案不見了，實際上只是伺服器沒開——誤導比報錯更糟。
  try {
    const r = await fetch('/api/courses');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const list = await r.json();
    if (!list.length) throw new Error('伺服器沒有回報任何課程');
    el.courseSelect.innerHTML = list
      .map((c) => `<option value="${c.id}">${c.name}${c.instructor ? ' — ' + c.instructor : ''}</option>`)
      .join('');
    state.courseId = list[0].id;
    el.start.disabled = false;
    $('load-error').textContent = '';
  } catch (e) {
    el.courseSelect.innerHTML = '<option value="">（讀不到課程）</option>';
    el.start.disabled = true;
    $('load-error').textContent =
      `連不上伺服器（${e.message}）。確認筆電上的服務有啟動、Tailscale 有連線，然後重新整理。`;
  }
}

$('retry-load').addEventListener('click', () => {
  $('load-error').textContent = '重新連線中…';
  loadCourses();
});
el.courseSelect.addEventListener('change', (e) => { state.courseId = e.target.value; });

el.start.addEventListener('click', async () => {
  el.start.disabled = true;
  try {
    await startAudio();
  } catch (e) {
    el.start.disabled = false;
    alert('無法取得麥克風：' + e.message
      + '\n\n提示：getUserMedia 只在 HTTPS 下可用，請走 Tailscale Serve 的網址。');
    return;
  }
  await acquireWakeLock();
  state.running = true;
  state.startedAt = Date.now();
  el.course.textContent = el.courseSelect.selectedOptions[0]
    ? el.courseSelect.selectedOptions[0].textContent : state.courseId;
  el.overlay.classList.add('hide');
  connect();
});

window.addEventListener('beforeunload', (e) => {
  if (state.running && !state.ended) { e.preventDefault(); e.returnValue = ''; }
});

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').then((reg) => {
    // 有新版就立刻換掉並重新整理。先前 cache-first 的寫法讓改版
    // 永遠送不到手機上，而且外觀完全看不出來在跑舊程式。
    reg.addEventListener('updatefound', () => {
      const w = reg.installing;
      if (!w) return;
      w.addEventListener('statechange', () => {
        if (w.state === 'installed' && navigator.serviceWorker.controller) {
          toast('已更新到新版本，正在重新載入…');
          setTimeout(() => location.reload(), 800);
        }
      });
    });
    reg.update();
  }).catch(() => { /* 沒有 SW 也能用 */ });
}

loadCourses();
