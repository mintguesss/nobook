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
  notesOpen: $('notes-open'),
  selectBar: $('select-bar'), selectCount: $('select-count'),
  selectWhy: $('select-why'), selectMerge: $('select-merge'),
  sumBar: $('sumbar'), sumCount: $('sumcount'), sumToggle: $('sum-toggle'),
  copyCount: $('copy-count'),
  detail: $('detail'), detailBody: $('detail-body'), detailTitle: $('detail-title'),
  upload: $('upload'),
};

const state = {
  ws: null, sessionId: null, courseId: 'example',
  seq: 0, pending: [],                   // 斷線期間累積的 frame
  running: false, paused: false, ended: false,
  startedAt: 0, elapsedBase: 0,
  autoScroll: true, retries: 0,
  wakeLock: null, ctx: null, node: null, stream: null,
  hiddenAt: 0, gaps: [], viewingSession: null,
  notesBusy: false, notesMd: '', notesSections: 0, notesCheck: null,
  endingTimer: null, finalReady: null,
  // 歷史清單的多選（長按進入）。合併的入口從清單頂端的區塊改到這裡——
  // 放在頂端的話，要合併哪幾段是系統幫你決定的，使用者沒得挑。
  selecting: false, selected: new Set(), rowsById: new Map(),
  sections: new Map(), batches: new Map(), markBusy: false,
  courseNames: new Map(), detailId: null, detailTab: null, conn: 'bad',
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
  state.conn = kind;
  el.dot.className = 'dot' + (kind === 'ok' && !state.paused ? ' ok' : kind === 'warn' ? ' warn' : '');
  showLiveStatus();
}

function showLiveStatus(extra) {
  const base = state.conn === 'warn' ? '重新連線中，錄音沒有中斷'
    : state.conn !== 'ok' ? '尚未連線'
      : state.paused ? '已暫停' : '錄音中';
  el.health.textContent = extra || base;
}

// ── 對話框 ────────────────────────────────────────────────────────────
// 取代 confirm/alert：系統對話框在 iPhone 主畫面版上標題是網址，按鈕也只有
// 「確定／取消」，看不出按下去會發生什麼事。回傳 false 代表取消。
function ask({ title, message = '', ok = '確定', cancel = '取消', danger = false,
               input = null, select = null }) {
  const dlg = $('dlg');
  dlg.innerHTML = '<h3></h3><p></p><div class="field" hidden></div><div class="btns"></div>';
  dlg.querySelector('h3').textContent = title;
  const p = dlg.querySelector('p');
  p.textContent = message;
  p.hidden = !message;
  const field = dlg.querySelector('.field');
  let ctrl = null;
  if (input) {
    ctrl = document.createElement('input');
    ctrl.type = 'text';
    ctrl.value = input.value || '';
    ctrl.placeholder = input.placeholder || '';
    ctrl.maxLength = input.maxLength || 40;
  } else if (select) {
    ctrl = document.createElement('select');
    for (const [v, label] of select.options) ctrl.add(new Option(label, v));
    ctrl.value = String(select.value);
  }
  if (ctrl) { field.hidden = false; field.appendChild(ctrl); }
  const btns = dlg.querySelector('.btns');
  return new Promise((resolve) => {
    const done = (v) => { dlg.close(); resolve(v); };
    if (cancel) {
      const c = document.createElement('button');
      c.className = 'btn';
      c.textContent = cancel;
      c.addEventListener('click', () => done(false));
      btns.appendChild(c);
    }
    const o = document.createElement('button');
    o.className = 'btn ' + (danger ? 'danger' : 'primary');
    o.textContent = ok;
    const accept = () => done(ctrl ? ctrl.value : true);
    o.addEventListener('click', accept);
    if (ctrl && input) {
      ctrl.addEventListener('keydown', (e) => { if (e.key === 'Enter') accept(); });
    }
    btns.appendChild(o);
    dlg.oncancel = (e) => { e.preventDefault(); done(false); };
    dlg.showModal();
  });
}

// ── wake lock（規格 §8.2）─────────────────────────────────────────────
async function acquireWakeLock(quiet) {
  if (!('wakeLock' in navigator)) {
    if (!quiet) toast('這個瀏覽器沒辦法保持螢幕開著，請手動把自動鎖定關掉');
    return;
  }
  try {
    state.wakeLock = await navigator.wakeLock.request('screen');
    state.wakeLock.addEventListener('release', () => { state.wakeLock = null; });
  } catch (e) {
    if (!quiet) toast('沒辦法保持螢幕開著，請手動把自動鎖定關掉');
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
  if ((state.running || up.running) && !state.wakeLock) acquireWakeLock(true);
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
  // iOS 接電話或被別的 App 搶走音訊後，AudioContext 會停在 interrupted，不會自己回來
  if (state.ctx.state === 'suspended') await state.ctx.resume();
  state.ctx.addEventListener('statechange', () => {
    if (state.ctx && state.ctx.state !== 'running' && state.ctx.state !== 'closed') {
      state.ctx.resume().catch(() => {});
    }
  });
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

// session id 要撐過頁面重載。只存在記憶體的話，平板一重新整理（或被
// 系統回收後重開）就認不得自己剛剛在錄哪一堂，那堂課會變成孤兒——
// 伺服器端還活著，但前端再也接不回去，只能重開一堂。
const SID_KEY = 'nobook.live';
function rememberSession() {
  try {
    if (state.sessionId && !state.ended) {
      localStorage.setItem(SID_KEY, JSON.stringify(
        { id: state.sessionId, course: state.courseId, at: Date.now() }));
    } else {
      localStorage.removeItem(SID_KEY);
    }
  } catch (e) { /* 無痕模式之類沒有 localStorage，不影響錄音 */ }
}
function recallSession() {
  try {
    const raw = localStorage.getItem(SID_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw);
    // 放太久的多半是忘了收尾的舊紀錄，別自動接回去嚇人
    if (!v || !v.id || Date.now() - (v.at || 0) > 12 * 3600 * 1000) return null;
    return v;
  } catch (e) { return null; }
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
      rememberSession();
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
      toast('正在產生完整筆記…', 20000);
      break;
    case 'final_summary':
      clearEnding();
      state.viewingSession = msg.session_id;
      // 整理完就回首頁。筆記已存進資料庫，從「歷史筆記」隨時叫得回來；
      // 留在錄完的畫面上沒有任何用處，還會讓人以為要等什麼。
      // 手抄版正開著在抄的話就別打斷。
      if (el.copyView.classList.contains('show')) {
        toast('筆記整理好了，已存進紀錄', 6000);
      } else {
        goHome();
        toast('筆記整理好了，已存進紀錄', 6000);
      }
      break;
    case 'stats':
      el.health.title =
        `RTF ${msg.rtf.toFixed(2)} · 佇列 ${msg.queue_depth} · VRAM ${Math.round(msg.vram_used_mb)}MB`
        + (msg.summary_degraded ? ' · 摘要降級' : '');
      showLiveStatus(msg.queue_depth > 8 ? '文字慢了一點，稍後會補上' : '');
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
  el.mark.textContent = busy ? '整理中…' : '整理目前內容';
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
  el.pause.textContent = state.paused ? '繼續' : '暫停';
  setLight(state.conn);
  if (state.paused) state.elapsedBase += (Date.now() - state.startedAt) / 1000;
  else state.startedAt = Date.now();
  toast(state.paused ? '已暫停錄音' : '已繼續錄音');
});

el.end.addEventListener('click', async () => {
  if (!state.running) return;
  if (!await ask({ title: '結束錄音？', message: '結束後會自動整理筆記。', ok: '結束錄音' })) return;
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
  el.copyView.classList.remove('show');
  el.finalTitle.textContent = '整理中';
  el.finalMd.innerHTML = '<span class="spin"></span><h2>正在整理筆記</h2>'
    + '<p class="hint">通常不到一分鐘。可以直接離開，整理好會存進紀錄。</p>';
  el.final.classList.add('show');
}

function clearEnding() {
}

// ── 計時 ──────────────────────────────────────────────────────────────
setInterval(() => {
  if (!state.running || state.paused) return;
  const secs = state.elapsedBase + (Date.now() - state.startedAt) / 1000;
  el.elapsed.textContent = fmt(secs);
}, 500);

// ── 回首頁 ────────────────────────────────────────────────────────────
// 課程結束後原本卡在結束畫面沒路可走。這裡把狀態整個歸零，
// 回到開始畫面就能再開一堂新的。
async function goHome() {
  if (state.running && !state.ended) {
    if (!await ask({
      title: '結束錄音並離開？',
      message: '還在錄音。已錄的內容會保留，並自動整理筆記。',
      ok: '結束並離開', cancel: '繼續錄音', danger: true,
    })) return;
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
  rememberSession();
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
  el.health.textContent = '準備中…';
  el.health.title = '';
  el.course.textContent = '—';
  el.note.value = '';
  el.note.classList.remove('show');
  setLight('bad');
  setMarkBusy(false);
  setNotesBusy(false);
  el.mark.disabled = false;
  el.pause.disabled = false;
  el.pause.textContent = '暫停';
  el.notes.disabled = false;
  el.end.disabled = false;
  el.toLatest.classList.remove('show');

  el.final.classList.remove('show');
  el.copyView.classList.remove('show');
  closeDetail();
  $('live-setup').classList.remove('show');
  el.history.classList.remove('show');
  el.overlay.classList.remove('hide');
  el.start.disabled = false;
  // 上一堂的手抄版不要留到下一堂
  state.notesMd = '';
  state.notesSections = 0;
  state.notesCheck = null;
  el.notesOpen.hidden = true;
  loadCourses();
  renderRecent();
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
  // 存起來才能關掉之後再打開。不存的話唯一的重看方式是「重新產生」，
  // 那要再等模型跑一輪，而且內容還會跟手上抄到一半的不一樣。
  state.notesMd = md;
  state.notesSections = nSections;
  state.notesCheck = check;
  el.notesOpen.hidden = false;
  const wrap = el.copyBody;
  renderMd(wrap, md);
  const chars = md.replace(/<!--[\s\S]*?-->/g, '').replace(/\s/g, '').length;
  el.copyCount.textContent = `約 ${chars} 字 · 取自 ${nSections || 0} 段`;
  renderCheck(wrap, check);
  el.copyView.classList.add('show');
}

// 把筆記的 Markdown 畫出來：標題、條列、段落。逐字顯示 # 和 - 給一般人看沒有意義。
function renderMd(wrap, md) {
  wrap.innerHTML = '';
  let ul = null;
  for (const raw of (md || '').split('\n')) {
    const line = raw.trim();
    if (!line) { ul = null; continue; }
    if (line.startsWith('<!--')) continue;
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const h = document.createElement(`h${Math.min(heading[1].length, 3)}`);
      if (heading[2].startsWith('★')) h.className = 'exam';
      h.textContent = heading[2].replace(/\*\*/g, '');
      wrap.appendChild(h);
      ul = null;
    } else if (/^[-*]\s+/.test(line)) {
      if (!ul) { ul = document.createElement('ul'); wrap.appendChild(ul); }
      const li = document.createElement('li');
      li.textContent = line.replace(/^[-*]\s+/, '').replace(/\*\*/g, '');
      ul.appendChild(li);
    } else {
      const p = document.createElement('p');
      p.textContent = line.replace(/\*\*/g, '');
      wrap.appendChild(p);
      ul = null;
    }
  }
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

// ── 紀錄清單 ──────────────────────────────────────────────────────────
function jobOf(r) {
  try { return JSON.parse(r.job || 'null') || {}; } catch (e) { return {}; }
}
const jobBusy = (j) => j.status === 'queued' || j.status === 'running';
const notesBusy = (j) => j.notes_status === 'queued' || j.notes_status === 'running';

const STAGE_TEXT = {
  queued: '排隊中',
  decoding: '讀取檔案',
  speakers: '分辨講者',
  transcribing: '轉文字',
  waiting_live: '等現場錄音結束',
};
function jobLabel(j) {
  const t = STAGE_TEXT[j.stage] || '處理中';
  return j.stage === 'transcribing' ? `${t} ${Math.round((j.progress || 0) * 100)}%` : t;
}

async function fetchCourseNames() {
  if (state.courseNames.size) return state.courseNames;
  try {
    for (const c of await (await fetch('/api/courses')).json()) state.courseNames.set(c.id, c.name);
  } catch (e) { /* 課名拿不到就顯示 id */ }
  return state.courseNames;
}

async function openHistory() {
  el.history.classList.add('show');
  exitSelect();
  showBuild();
  showUsage();
  await renderHistory();
}

async function renderHistory() {
  let rows;
  try {
    rows = await (await fetch('/api/sessions?limit=300')).json();
  } catch (e) {
    toast('讀取紀錄失敗：' + e.message);
    return;
  }
  const courses = await fetchCourseNames();
  state.rowsById.clear();
  const q = $('history-search').value.trim().toLowerCase();
  const openNames = new Set([...el.historyList.querySelectorAll('.cfold[open] .cname')]
    .map((n) => n.textContent));
  el.historyList.innerHTML = '';

  // 照分類分資料夾。一門課上十八週就是十八筆，平鋪的話要滑很久。
  const byCourse = new Map();
  for (const r of rows) {
    const name = courses.get(r.course_id) || r.course_id;
    state.rowsById.set(r.id, r);
    if (q && !`${r.note || ''} ${name} ${rowDate(r)}`.toLowerCase().includes(q)) continue;
    if (!byCourse.has(name)) byCourse.set(name, []);
    byCourse.get(name).push(r);
  }
  const order = [...byCourse.entries()].sort(
    (a, b) => (b[1][0].started_at || '').localeCompare(a[1][0].started_at || ''));
  for (const [name, list] of order) {
    const box = document.createElement('details');
    box.className = 'cfold';
    box.innerHTML = '<summary><span class="cname"></span><span class="cmeta"></span></summary>';
    box.querySelector('.cname').textContent = name;
    box.querySelector('.cmeta').textContent = `${list.length} 筆`;
    // 搜尋中、有需要處理的、或只有一個資料夾時直接展開
    box.open = !!q || order.length === 1 || openNames.has(name)
      || list.some((r) => jobBusy(jobOf(r)) || !r.ended_at);
    for (const r of list) box.appendChild(historyRow(r, name, false));
    el.historyList.appendChild(box);
  }
  if (!order.length) {
    el.historyList.innerHTML = `<div class="empty-note">${q ? '找不到符合的紀錄' : '還沒有任何紀錄'}</div>`;
  }
  scheduleJobPoll(rows);
}

function rowDate(r) {
  return new Date(r.started_at).toLocaleString('zh-TW',
    { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
}

// 一列紀錄。沒取名字的就用日期當標題。
function historyRow(r, courseName, showCourse) {
  const job = jobOf(r);
  const live = state.running && !state.ended && r.id === state.sessionId;
  const btn = document.createElement('button');
  btn.className = 'hrow';
  btn.innerHTML = '<span class="hpick" aria-hidden="true"></span>' +
    '<span class="hmain"><span class="hname"></span><span class="hmeta"></span></span>' +
    '<span class="hbadge"></span>';
  btn.querySelector('.hname').textContent = r.note || rowDate(r);
  const when = new Date(r.started_at);
  const tile = document.createElement('span');
  tile.className = 'htile' + (r.course_id === 'meeting' ? ' meeting' : '');
  tile.innerHTML = '<b></b><small></small>';
  tile.firstChild.textContent = when.getDate();
  tile.lastChild.textContent = `${when.getMonth() + 1} 月`;
  btn.insertBefore(tile, btn.querySelector('.hmain'));
  const bits = [];
  if (r.note) bits.push(rowDate(r));
  if (showCourse) bits.push(courseName);
  const dur = r.duration_s || job.duration_s;
  if (dur) bits.push(`${Math.max(1, Math.round(dur / 60))} 分鐘`);
  let mergedN = 0;
  try { mergedN = (JSON.parse(r.merged_from || '[]') || []).length; } catch (e) { }
  if (mergedN) bits.push(`合併 ${mergedN} 段`);
  btn.querySelector('.hmeta').textContent = bits.join(' · ');

  const badge = btn.querySelector('.hbadge');
  if (live) { badge.textContent = '錄音中'; badge.classList.add('busy'); }
  else if (jobBusy(job)) { badge.textContent = jobLabel(job); badge.classList.add('busy'); }
  else if (notesBusy(job)) {
    badge.textContent = `整理中 ${Math.round((job.notes_progress || 0) * 100)}%`;
    badge.classList.add('busy');
  }
  else if (job.status === 'failed') { badge.textContent = '失敗'; badge.classList.add('fail'); }
  else if (!r.ended_at) { badge.textContent = '未完成'; badge.classList.add('todo'); }
  else badge.hidden = true;

  btn.addEventListener('click', () => {
    if (state.selecting) { toggleSelect(r); return; }
    openDetail(r.id, courseName);
  });
  // 長按多選還是留著給熟的人用，但入口已經是清單上方的「選取」
  bindLongPress(btn, () => {
    if (!state.selecting) startSelect();
    toggleSelect(r);
  });

  const wrap = document.createElement('div');
  wrap.className = 'hitem';
  if (state.selected.has(r.id)) wrap.classList.add('picked');
  wrap.dataset.sid = r.id;
  wrap.appendChild(btn);
  return wrap;
}

// 首頁的最近紀錄
// 剛打開時 Tailscale 可能還沒連上。畫面已經從快取開起來了，連上之後把資料補回來，
// 不用使用者關掉重開。
let onlineRetry = null;
// 網路卡住時請求不會失敗、只會一直等，不設時限就永遠不會重試
function fetchSoon(url, ms = 4000) {
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), ms);
  return fetch(url, { signal: abort.signal }).finally(() => clearTimeout(timer));
}
function retryWhenOnline() {
  if (onlineRetry) return;
  let trying = false;
  onlineRetry = setInterval(async () => {
    if (trying) return;
    trying = true;
    try {
      const r = await fetchSoon('/api/health', 1800);
      if (!r.ok) return;
    } catch (e) { return; } finally { trying = false; }
    clearInterval(onlineRetry);
    onlineRetry = null;
    loadCourses();
    renderRecent();
    if (swReg) swReg.update().catch(() => { /* 下次再檢查 */ });
  }, 2000);
}

async function renderRecent() {
  let rows;
  try {
    rows = await (await fetchSoon('/api/sessions?limit=6')).json();
  } catch (e) { retryWhenOnline(); return; }
  const courses = await fetchCourseNames();
  const list = $('recent-list');
  list.innerHTML = '';
  for (const r of rows) {
    state.rowsById.set(r.id, r);
    list.appendChild(historyRow(r, courses.get(r.course_id) || r.course_id, true));
  }
  $('recent-empty').hidden = rows.length > 0;
  scheduleJobPoll(rows);
}

// 有檔案在轉文字時定期更新進度。沒有的話不打 API。
let jobPollTimer = null;
function scheduleJobPoll(rows) {
  clearTimeout(jobPollTimer);
  if (!rows.some((r) => jobBusy(jobOf(r)) || notesBusy(jobOf(r)))) return;
  jobPollTimer = setTimeout(() => {
    if (document.visibilityState === 'hidden') { scheduleJobPoll(rows); return; }
    if (el.detail.classList.contains('show')) { scheduleJobPoll(rows); return; }
    if (el.history.classList.contains('show')) {
      if (!state.selecting) renderHistory(); else scheduleJobPoll(rows);
    } else if (!el.overlay.classList.contains('hide')) renderRecent();
  }, 4000);
}

// ── 歷史清單的多選與合併 ──────────────────────────────────────────────
// 長按進入選取模式，左邊出現圓圈。合併只在「同一門課、同一天」之間
// 成立——跨課或跨天接起來，時間軸跟筆記都沒有意義。
function bindLongPress(node, fn) {
  let timer = null;
  let moved = false;
  const start = () => {
    moved = false;
    timer = setTimeout(() => { timer = null; if (!moved) fn(); }, 480);
  };
  const cancel = () => { if (timer) { clearTimeout(timer); timer = null; } };
  node.addEventListener('touchstart', start, { passive: true });
  node.addEventListener('touchmove', () => { moved = true; cancel(); },
                        { passive: true });
  node.addEventListener('touchend', cancel);
  node.addEventListener('touchcancel', cancel);
  node.addEventListener('mousedown', start);
  node.addEventListener('mousemove', () => { moved = true; cancel(); });
  node.addEventListener('mouseup', cancel);
  node.addEventListener('mouseleave', cancel);
  // 長按在桌面瀏覽器會跳出右鍵選單，蓋住剛出現的圓圈
  node.addEventListener('contextmenu', (e) => e.preventDefault());
}

function startSelect() {
  state.selecting = true;
  state.selected.clear();
  el.history.classList.add('selecting');
  $('select-start').textContent = '完成';
}

function exitSelect() {
  state.selecting = false;
  state.selected.clear();
  el.history.classList.remove('selecting');
  $('select-start').textContent = '選取';
  el.historyList.querySelectorAll('.hitem.picked')
    .forEach((n) => n.classList.remove('picked'));
  updateSelectBar();
}

function toggleSelect(r) {
  if (state.selected.has(r.id)) state.selected.delete(r.id);
  else state.selected.add(r.id);
  const node = el.historyList.querySelector(`.hitem[data-sid="${r.id}"]`);
  if (node) node.classList.toggle('picked', state.selected.has(r.id));
  if (!state.selected.size) { exitSelect(); return; }
  updateSelectBar();
}

/** 選到的這幾筆能不能合併：同課、同日、至少兩筆、都已結束。 */
function mergeability() {
  const rows = [...state.selected].map((id) => state.rowsById.get(id))
    .filter(Boolean);
  if (rows.length < 2) return { ok: false, why: '' };
  const course = rows[0].course_id;
  const day = (rows[0].started_at || '').slice(0, 10);
  if (rows.some((r) => r.course_id !== course)) {
    return { ok: false, why: '只有同一個分類的才能合併' };
  }
  if (rows.some((r) => (r.started_at || '').slice(0, 10) !== day)) {
    return { ok: false, why: '只有同一天的才能合併' };
  }
  const open = rows.filter((r) => !r.ended_at);
  if (open.length) {
    return { ok: false, why: `有 ${open.length} 筆還沒完成，不能合併` };
  }
  return { ok: true, rows };
}

function updateSelectBar() {
  const bar = el.selectBar;
  if (!state.selecting || !state.selected.size) { bar.classList.remove('show'); return; }
  bar.classList.add('show');
  const m = mergeability();
  el.selectCount.textContent = `已選 ${state.selected.size} 筆`;
  el.selectWhy.textContent = m.ok ? '' : m.why;
  el.selectMerge.disabled = !m.ok;
  el.selectMerge.hidden = state.selected.size < 2;
}

async function mergeSelected() {
  const m = mergeability();
  if (!m.ok) return;
  const mins = Math.round(
    m.rows.reduce((s, r) => s + (r.duration_s || 0), 0) / 60);
  if (!await ask({
    title: `把 ${m.rows.length} 段接成一筆？`,
    message: `共約 ${mins} 分鐘，會重新產生筆記，大約要三到五分鐘。\n原本那幾筆不會被刪掉，合併後會多一筆新紀錄。`,
    ok: '合併',
  })) return;
  el.selectMerge.disabled = true;
  el.selectMerge.textContent = '合併中…';
  try {
    const ids = m.rows
      .sort((a, b) => (a.started_at || '').localeCompare(b.started_at || ''))
      .map((r) => r.id);
    const res = await fetch('/api/sessions/merge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || ('HTTP ' + res.status));
    toast(`已合併：${j.sections} 段、${j.chars} 字`);
    exitSelect();
    openHistory();
  } catch (e) {
    toast('合併失敗：' + e.message);
  } finally {
    el.selectMerge.textContent = '合併';
    updateSelectBar();
  }
}

async function deleteSelected() {
  const n = state.selected.size;
  if (!n) return;
  if (!await ask({
    title: `刪除這 ${n} 筆紀錄？`,
    message: '逐字稿、筆記和錄音都會一起刪掉，無法復原。',
    ok: '刪除', danger: true,
  })) return;
  let failed = 0;
  for (const id of [...state.selected]) {
    try {
      const res = await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
      if (!res.ok) failed += 1;
    } catch (e) { failed += 1; }
  }
  toast(failed ? `有 ${failed} 筆刪除失敗` : '已刪除');
  exitSelect();
  renderHistory();
}

// 投影片對照：把逐字稿對到教材的頁碼。
// 涵蓋率通常不高（實測一堂 21%）——老師講投影片以外的東西、Q&A、
// 純圖片的頁都對不上。所以這裡把涵蓋率直接寫出來，不要讓使用者
// 以為沒列到的部分是漏掉了。
function renderAlign(id, d, sec, a, onJump) {
  const has = a && a.pages && a.pages.length;

  const row = document.createElement('div');
  row.className = 'btns';
  const go = document.createElement('button');
  go.className = 'btn';
  go.textContent = has ? '重新對照' : '對照投影片';
  go.addEventListener('click', async () => {
    go.disabled = true;
    go.textContent = '對照中…';
    try {
      const res = await fetch(`/api/sessions/${id}/align`, { method: 'POST' });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail || ('HTTP ' + res.status));
      toast(`對到 ${j.pages.length} 段、涵蓋 ${Math.round(j.coverage * 100)}%`);
      openDetail(id, d.courseName, true);
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
      li.querySelector('.pjump').addEventListener('click', () => {
        onJump();
        jumpToNote(e.start_s);
      });
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
    await ask({ title: '沒辦法使用麥克風', message: micHelp(e), ok: '知道了', cancel: null });
    return;
  }
  await acquireWakeLock();

  closeDetail();
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

async function showUsage() {
  const bar = $('usage');
  try {
    const u = await (await fetch('/api/storage')).json();
    const gb = (n) => (n / 1e9).toFixed(1);
    let txt = `${u.sessions} 筆紀錄`;
    txt += ` · 硬碟還剩 ${gb(u.disk_free_bytes)} GB`;
    bar.textContent = txt;
  } catch (e) { bar.textContent = ''; }
}

// ── 單筆紀錄 ──────────────────────────────────────────────────────────
function speakerName(d, n) {
  return (d.names && d.names[String(n)]) || `講者 ${n}`;
}
const spkClass = (n) => `spk-${((n - 1) % 6) + 1}`;

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(j.detail || ('HTTP ' + res.status));
  return j;
}
const postJson = (url, body) => fetchJson(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});

let detailPoll = null;
async function openDetail(id, courseName, keepTab) {
  clearTimeout(detailPoll);
  el.detail.classList.add('show');
  state.detailId = id;
  const prevTab = keepTab ? state.detailTab : null;
  let d;
  try {
    d = await (await fetch(`/api/sessions/${id}`)).json();
    if (!d.session) throw new Error('找不到這筆紀錄');
  } catch (e) {
    el.detailBody.innerHTML = '';
    el.detailBody.appendChild(infoCard('bad', '讀取失敗', e.message));
    return;
  }
  if (state.detailId !== id) return;
  const s = d.session;
  d.courseName = courseName;
  d.job = jobOf(s);
  try { d.names = JSON.parse(s.speaker_names || '{}') || {}; } catch (e) { d.names = {}; }
  const busy = jobBusy(d.job);
  const body = el.detailBody;
  body.innerHTML = '';
  el.detailTitle.textContent = courseName || '紀錄';

  // 名稱：同一天好幾筆時，光看時間認不出是哪一筆
  const name = document.createElement('input');
  name.type = 'text';
  name.className = 'd-name';
  name.value = s.note || '';
  name.placeholder = '幫這筆紀錄取個名字';
  name.maxLength = 40;
  name.setAttribute('aria-label', '名稱');
  name.addEventListener('change', async () => {
    try {
      await postJson(`/api/sessions/${id}/note`, { note: name.value });
      toast('名稱已存');
    } catch (e) { toast('儲存失敗：' + e.message); }
  });
  name.addEventListener('keydown', (e) => { if (e.key === 'Enter') name.blur(); });
  body.appendChild(name);

  const metaBits = [new Date(s.started_at).toLocaleString('zh-TW',
    { dateStyle: 'medium', timeStyle: 'short', hour12: false })];
  const dur = s.duration_s || d.job.duration_s;
  if (dur) metaBits.push(`${Math.max(1, Math.round(dur / 60))} 分鐘`);
  const meta = document.createElement('p');
  meta.className = 'd-meta';
  meta.textContent = metaBits.join(' · ');
  body.appendChild(meta);

  if (busy) {
    renderJobCard(body, d);
    detailPoll = setTimeout(() => {
      if (el.detail.classList.contains('show') && state.detailId === id) openDetail(id, courseName, true);
    }, 4000);
    return;                      // 處理中沒有內容可看
  }
  if (d.job.status === 'failed') renderFailedCard(body, d, id, courseName);
  if (!s.ended_at && d.job.status !== 'failed') renderUnfinishedCard(body, d, id, courseName);
  if (d.job.notes_error && s.course_id !== 'meeting') {
    body.appendChild(infoCard('warn', '逐字稿好了，但筆記沒有產生', d.job.notes_error));
  }
  if (d.job.empty) {
    body.appendChild(infoCard('warn', '沒有辨識到語音', '這個檔案裡可能沒有人說話，或聲音太小。'));
  }

  let player = null;
  if (d.has_audio) {
    const holder = document.createElement('div');
    holder.className = 'player';
    player = document.createElement('audio');
    player.controls = true;
    player.preload = 'metadata';
    player.playsInline = true;
    player.src = `/api/sessions/${id}/audio`;
    // 播不了的話時間按鈕也沒用，講清楚而不是放一個一直轉圈的播放器
    player.addEventListener('error', () => {
      holder.textContent = '這台裝置播不了這段錄音，可以到電腦上聽。';
      holder.className = 'hint d-block';
    });
    holder.appendChild(player);
    body.appendChild(holder);
  }

  // 分頁：只放這筆真的有內容的。原本六個收合區塊全部列出來，一半是空的。
  const tabs = document.createElement('div');
  tabs.className = 'tabs';
  tabs.setAttribute('role', 'tablist');
  body.appendChild(tabs);
  const panels = new Map();
  const addTab = (key, title, count) => {
    const b = document.createElement('button');
    b.className = 'tab';
    b.setAttribute('role', 'tab');
    b.textContent = title;
    if (count) {
      const n = document.createElement('span');
      n.className = 'n';
      n.textContent = count;
      b.appendChild(n);
    }
    b.addEventListener('click', () => selectTab(key));
    tabs.appendChild(b);
    const p = document.createElement('div');
    p.className = 'panel';
    body.appendChild(p);
    panels.set(key, { tab: b, panel: p });
    return p;
  };
  const selectTab = (key) => {
    state.detailTab = key;
    for (const [k, v] of panels) {
      v.tab.classList.toggle('on', k === key);
      v.tab.setAttribute('aria-selected', k === key ? 'true' : 'false');
      v.panel.classList.toggle('on', k === key);
    }
  };

  if (s.course_id === 'meeting' && d.segments.length) {
    renderMeetingTab(addTab('meeting', '會議記錄'), d, id, player);
  }
  const hc = s.handcopy_md;
  const fm = s.final_md;
  if (hc || fm) renderNotesTab(addTab('notes', '筆記'), id, hc, fm);
  renderTranscriptTab(addTab('transcript', '逐字稿'), d, id, courseName, player);

  if (d.sections.length) {
    const p = addTab('sections', '課中整理', String(d.sections.length));
    let align = null;
    try {
      align = await (await fetch(`/api/sessions/${id}/align`)).json();
      if (!align || !align.pages || !align.pages.length) align = null;
    } catch (e) { align = null; }
    renderSections(p, d.sections, align);
  }

  // 投影片對照只有這門課放了教材才有意義
  try {
    const m = await (await fetch('/api/materials')).json();
    const hasMaterial = (m.groups || []).some((g) => g.known
      && (g.folder === s.course_id || g.course === courseName || g.folder === courseName));
    if (hasMaterial) {
      let align = null;
      try {
        align = await (await fetch(`/api/sessions/${id}/align`)).json();
        if (!align || !align.pages || !align.pages.length) align = null;
      } catch (e) { align = null; }
      const p = addTab('slides', '投影片');
      renderAlign(id, d, p, align, () => {
        selectTab('sections');
      });
    }
  } catch (e) { /* 拿不到教材清單就不顯示這一頁 */ }

  const wanted = prevTab && panels.has(prevTab) ? prevTab
    : (panels.has('meeting') ? 'meeting' : panels.has('notes') ? 'notes' : 'transcript');
  selectTab(wanted);

  $('detail-delete').onclick = async () => {
    if (!await ask({
      title: '刪除這筆紀錄？',
      message: '逐字稿、筆記和錄音都會一起刪掉，無法復原。',
      ok: '刪除', danger: true,
    })) return;
    try {
      await fetchJson(`/api/sessions/${id}`, { method: 'DELETE' });
      toast('已刪除');
      closeDetail();
    } catch (e) { toast('刪除失敗：' + e.message); }
  };
}

function infoCard(kind, title, msg) {
  const c = document.createElement('div');
  c.className = `card ${kind} d-block`;
  const h = document.createElement('h3');
  h.textContent = title;
  c.appendChild(h);
  if (msg) {
    const p = document.createElement('p');
    p.textContent = msg;
    c.appendChild(p);
  }
  return c;
}

function renderJobCard(body, d) {
  const j = d.job;
  const c = infoCard('', jobLabel(j), j.stage === 'waiting_live'
    ? '有人正在現場錄音，等錄音結束會自動繼續。'
    : '在伺服器上處理，可以關掉這個畫面。一小時的會議錄音大約要 10 到 20 分鐘，轉完會接著整理。');
  c.classList.add('d-block');
  const pct = j.stage === 'transcribing' ? (j.progress || 0) : 0;
  const track = document.createElement('div');
  track.className = 'bar-track';
  track.innerHTML = '<div class="bar-fill"></div>';
  track.firstChild.style.width = `${Math.max(3, pct * 100)}%`;
  c.appendChild(track);
  body.appendChild(c);
  $('detail-delete').onclick = async () => {
    if (!await ask({ title: '取消並刪除這筆？', message: '上傳的錄音也會一起刪掉。', ok: '刪除', danger: true })) return;
    try {
      await fetchJson(`/api/sessions/${d.session.id}`, { method: 'DELETE' });
      toast('已取消');
      closeDetail();
    } catch (e) { toast('取消失敗：' + e.message); }
  };
}

function renderFailedCard(body, d, id, courseName) {
  const c = infoCard('bad', '轉文字失敗', d.job.error || '');
  const b = document.createElement('button');
  b.className = 'btn';
  b.textContent = '重試';
  b.addEventListener('click', async () => {
    b.disabled = true;
    try {
      await postJson(`/api/sessions/${id}/retranscribe`, { speakers: d.job.speakers || 0 });
      openDetail(id, courseName);
    } catch (e) { toast('重試失敗：' + e.message); b.disabled = false; }
  });
  c.appendChild(b);
  body.appendChild(c);
}

// 手機睡著或網路斷掉會留下這種紀錄，逐字稿都在，只差沒收尾。
function renderUnfinishedCard(body, d, id, courseName) {
  const c = infoCard('warn', '這筆錄音沒有正常結束',
    d.resumable ? '可以接回去繼續錄，或直接結束並產生筆記。' : '已有的逐字稿都還在，可以直接結束並產生筆記。');
  const bar = document.createElement('div');
  bar.className = 'btns';
  c.appendChild(bar);
  if (d.resumable) {
    const r = document.createElement('button');
    r.className = 'btn primary';
    r.textContent = '繼續錄音';
    r.addEventListener('click', () => resumeSession(d));
    bar.appendChild(r);
  }
  const b = document.createElement('button');
  b.className = d.resumable ? 'btn' : 'btn primary';
  b.textContent = '結束並產生筆記';
  b.addEventListener('click', async () => {
    b.disabled = true;
    b.textContent = '產生中，約一到三分鐘…';
    try {
      await postJson(`/api/sessions/${id}/finish`);
      toast('完成');
      openDetail(id, courseName);
    } catch (e) {
      toast('失敗：' + e.message);
      b.disabled = false;
      b.textContent = '結束並產生筆記';
    }
  });
  bar.appendChild(b);
  body.appendChild(c);
}

// 跟下載的 Excel 同樣五欄，順序也一樣，對照起來才不會亂
const MEETING_COLUMNS = [
  ['literature', '文獻整理重點', 'var(--spk-1)'],
  ['topic', '題目整理建議', 'var(--spk-2)'],
  ['next', '下次目標', 'var(--spk-4)'],
  ['ideas', '可用觀點記錄', 'var(--spk-3)'],
  ['misc', '小事記錄', 'var(--spk-5)'],
];

function renderMeetingTab(p, d, id, player) {
  let job = d.job;
  let notes = null;
  try { notes = JSON.parse(d.session.meeting_notes || 'null'); } catch (e) { notes = null; }

  const request = async (btn) => {
    btn.disabled = true;
    try {
      await postJson(`/api/sessions/${id}/meeting-notes`);
      job = { ...job, notes_status: 'queued', notes_progress: 0, notes_error: null };
      paint();
    } catch (e) { toast('無法開始：' + e.message); btn.disabled = false; }
  };
  const poll = () => {
    clearTimeout(detailPoll);
    detailPoll = setTimeout(async () => {
      if (state.detailId !== id) return;
      try {
        const fresh = await fetchJson(`/api/sessions/${id}`);
        if (state.detailId !== id) return;
        job = jobOf(fresh.session);
        try { notes = JSON.parse(fresh.session.meeting_notes || 'null'); } catch (e) { notes = null; }
        paint();
      } catch (e) { poll(); }
    }, 3000);
  };

  function paint() {
    p.innerHTML = '';
    if (notesBusy(job)) {
      const c = infoCard('', `正在整理會議記錄 ${Math.round((job.notes_progress || 0) * 100)}%`,
        '大約一兩分鐘，可以先看逐字稿，也可以關掉這個畫面。');
      const track = document.createElement('div');
      track.className = 'bar-track';
      track.innerHTML = '<div class="bar-fill"></div>';
      track.firstChild.style.width = `${Math.max(3, (job.notes_progress || 0) * 100)}%`;
      c.appendChild(track);
      p.appendChild(c);
      poll();
    } else if (job.notes_error) {
      const c = infoCard('warn', '這次沒有整理成功', job.notes_error);
      const b = document.createElement('button');
      b.className = 'btn';
      b.textContent = '再試一次';
      b.addEventListener('click', () => request(b));
      c.appendChild(b);
      p.appendChild(c);
    }
    if (!notes) {
      if (!notesBusy(job) && !job.notes_error) {
        const c = infoCard('', '還沒有會議記錄', '用這份逐字稿整理成五欄的會議記錄，大約一兩分鐘。');
        const b = document.createElement('button');
        b.className = 'btn primary';
        b.textContent = '開始整理';
        b.addEventListener('click', () => request(b));
        c.appendChild(b);
        p.appendChild(c);
      }
      return;
    }
    const head = document.createElement('div');
    head.className = 'card m-head';
    head.innerHTML = '<div class="field"><label>Excel 工作表名稱</label><input type="text" maxlength="31"></div><div class="btns"></div>';
    const name = head.querySelector('input');
    head.querySelector('label').htmlFor = name.id = 'm-sheet';
    name.value = notes.sheet_name || '';
    const fixes = Object.entries((d.job || {}).auto_fixes || {});
    if (fixes.length) {
      const note = document.createElement('p');
      note.className = 'hint';
      note.textContent = '自動改正了聽錯的詞：' + fixes.map(([a, b]) => `${a} → ${b}`).join('、');
      head.querySelector('.field').after(note);
    }
    name.addEventListener('change', async () => {
      try {
        const r = await postJson(`/api/sessions/${id}/meeting-sheet`, { name: name.value });
        notes.sheet_name = name.value = r.name;
        toast('名稱已存');
      } catch (e) { toast('儲存失敗：' + e.message); name.value = notes.sheet_name || ''; }
    });
    const btns = head.querySelector('.btns');
    const main = document.createElement('button');
    main.className = 'btn primary';
    main.textContent = '下載 Excel';
    main.addEventListener('click', () => openFile(`/api/sessions/${id}/meeting-notes.xlsx`));
    btns.appendChild(main);
    const all = document.createElement('button');
    all.className = 'btn';
    all.textContent = '全部會議合成一檔';
    all.addEventListener('click', () => openFile('/api/meeting-notes.xlsx'));
    btns.appendChild(all);
    p.appendChild(head);

    const cols = document.createElement('div');
    cols.className = 'm-cols';
    for (const [key, title, color] of MEETING_COLUMNS) {
      const items = (notes.columns && notes.columns[key]) || [];
      const c = document.createElement('section');
      c.className = 'mcol';
      c.style.setProperty('--c', color);
      const h = document.createElement('h3');
      h.textContent = title;
      if (items.length) {
        const n = document.createElement('small');
        n.textContent = `${items.length} 則`;
        h.appendChild(n);
      }
      c.appendChild(h);
      if (!items.length) {
        const none = document.createElement('p');
        none.className = 'none';
        none.textContent = '這次沒有';
        c.appendChild(none);
      } else {
        const ol = document.createElement('ol');
        for (const it of items) {
          const li = document.createElement('li');
          const text = document.createElement('span');
          text.textContent = it.text;
          li.appendChild(text);
          const t = document.createElement('button');
          t.className = 'mt' + (player ? ' can' : '');
          t.textContent = fmt(it.t);
          if (player) {
            t.setAttribute('aria-label', `聽 ${fmt(it.t)} 的原音`);
            t.addEventListener('click', () => seekAudio(player, it.t));
          } else t.disabled = true;
          li.appendChild(t);
          ol.appendChild(li);
        }
        c.appendChild(ol);
      }
      cols.appendChild(c);
    }
    p.appendChild(cols);

    const foot = document.createElement('p');
    foot.className = 'hint m-foot';
    foot.textContent = notes.degraded
      ? '有一段沒有整理成功，內容可能缺一部分，可以按重新整理。'
      : '這是 AI 從逐字稿整理的，交出去前請看過一次。點時間可以聽那段原音。';
    p.appendChild(foot);
    if (!notesBusy(job)) {
      const again = document.createElement('button');
      again.className = 'link';
      again.textContent = '重新整理';
      again.addEventListener('click', async () => {
        if (!await ask({ title: '重新整理這份會議記錄？', message: '會用逐字稿重新產生，結果會取代目前的內容。', ok: '重新整理' })) return;
        request(again);
      });
      p.appendChild(again);
    }
  }
  paint();
}

function renderTranscriptTab(p, d, id, courseName, player) {
  const segs = d.segments;
  if (!segs.length) {
    const e = document.createElement('div');
    e.className = 'empty-note';
    e.textContent = '這筆沒有逐字稿';
    p.appendChild(e);
    return;
  }
  const tools = document.createElement('div');
  tools.className = 'tools';
  p.appendChild(tools);

  const search = document.createElement('input');
  search.type = 'search';
  search.placeholder = '搜尋逐字稿';
  search.setAttribute('aria-label', '搜尋逐字稿');
  tools.appendChild(search);

  const speakers = [...new Set(segs.map((g) => g.speaker).filter(Boolean))].sort((a, b) => a - b);
  if (speakers.length) {
    const chips = document.createElement('div');
    chips.className = 'chips';
    for (const n of speakers) {
      const c = document.createElement('button');
      c.className = `chip ${spkClass(n)}`;
      c.textContent = speakerName(d, n);
      c.title = '改名';
      c.addEventListener('click', async () => {
        const v = await ask({
          title: `這位是誰？`,
          message: `取了名字之後，逐字稿和下載的檔案都會用這個名字。`,
          input: { value: d.names[String(n)] || '', placeholder: `講者 ${n}`, maxLength: 20 },
          ok: '儲存',
        });
        if (v === false) return;
        const names = { ...d.names };
        if (v.trim()) names[String(n)] = v.trim(); else delete names[String(n)];
        try {
          await postJson(`/api/sessions/${id}/speakers`, { names });
          openDetail(id, courseName, true);
        } catch (e) { toast('儲存失敗：' + e.message); }
      });
      chips.appendChild(c);
    }
    tools.appendChild(chips);
    const hint = document.createElement('p');
    hint.className = 'hint';
    hint.textContent = '講者是用聲音自動分的，可能會認錯。點名字可以改名，點時間可以聽原音確認。';
    tools.appendChild(hint);
  }
  // 分錯人數是最常見的狀況，要給一條能自己修的路
  if (d.has_audio && d.job.diarize !== false && d.session.course_id === 'meeting') {
    const again = document.createElement('button');
    again.className = 'btn';
    again.textContent = speakers.length ? '人數不對？重新分辨講者' : '標出誰講了什麼';
    again.addEventListener('click', async () => {
      const v = await ask({
        title: '這段錄音有幾個人講話？',
        message: '會用保存的錄音重新轉一次，一小時大約要 10 到 20 分鐘。已取的名字要重新對照。',
        select: {
          value: d.job.speakers || 0,
          options: [[0, '不確定，自動判斷'], ...[1, 2, 3, 4, 5, 6, 7, 8].map((n) => [n, `${n} 人`])],
        },
        ok: '開始',
      });
      if (v === false) return;
      try {
        await postJson(`/api/sessions/${id}/retranscribe`, { speakers: Number(v) });
        openDetail(id, courseName);
      } catch (e) { toast('無法開始：' + e.message); }
    });
    tools.appendChild(again);
  }

  const box = document.createElement('div');
  box.className = 'lines';
  p.appendChild(box);
  const rows = [];
  let prevSpk = null;
  for (const g of segs) {
    const ln = document.createElement('div');
    ln.className = 'ln' + (g.avg_logprob != null && g.avg_logprob < -0.9 ? ' low' : '');
    const t = document.createElement('button');
    t.className = 't' + (player ? ' can' : '');
    t.textContent = fmt(g.start_s);
    if (player) {
      t.setAttribute('aria-label', `從 ${fmt(g.start_s)} 開始播放`);
      t.addEventListener('click', () => seekAudio(player, g.start_s));
    } else t.disabled = true;
    ln.appendChild(t);
    const b = document.createElement('div');
    b.className = 'b';
    if (g.speaker && g.speaker !== prevSpk) {
      const w = document.createElement('span');
      w.className = `who ${spkClass(g.speaker)}`;
      w.textContent = speakerName(d, g.speaker);
      b.appendChild(w);
    }
    prevSpk = g.speaker;
    const x = document.createElement('span');
    x.className = 'x';
    x.textContent = g.text;
    b.appendChild(x);
    ln.appendChild(b);
    box.appendChild(ln);
    rows.push({ ln, x, text: g.text, who: g.speaker ? speakerName(d, g.speaker) : '' });
  }
  search.addEventListener('input', () => {
    const q = search.value.trim();
    const lower = q.toLowerCase();
    for (const r of rows) {
      const hit = !q || r.text.toLowerCase().includes(lower) || r.who.includes(q);
      r.ln.hidden = !hit;
      r.x.textContent = r.text;
      if (q && hit) {
        const i = r.text.toLowerCase().indexOf(lower);
        if (i >= 0) {
          r.x.textContent = '';
          const m = document.createElement('mark');
          m.textContent = r.text.slice(i, i + q.length);
          r.x.append(r.text.slice(0, i), m, r.text.slice(i + q.length));
        }
      }
    }
  });
  p.appendChild(downloadRow([
    ['下載文字檔', () => dl(id, 'txt', 'transcript')],
    ['下載 Word', () => dl(id, 'docx', 'transcript')],
  ]));
}

// iOS 要先載入中繼資料才設得了 currentTime，而且 play() 要在點擊當下呼叫
function seekAudio(player, t) {
  const go = () => {
    player.currentTime = t;
    const r = player.play();
    if (r && r.catch) r.catch(() => {});
  };
  if (player.readyState >= 1) { go(); return; }
  player.addEventListener('loadedmetadata', go, { once: true });
  player.load();
}

function downloadRow(pairs) {
  const r = document.createElement('div');
  r.className = 'btns dl-row';
  for (const [label, fn] of pairs) {
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = label;
    b.addEventListener('click', fn);
    r.appendChild(b);
  }
  return r;
}

function renderNotesTab(p, id, hc, fm) {
  const content = document.createElement('div');
  const show = (which) => {
    content.innerHTML = '';
    const md = which === 'handcopy' ? hc : fm;
    const box = document.createElement('div');
    box.className = 'md';
    renderMd(box, md);
    content.appendChild(box);
    content.appendChild(downloadRow([
      ['複製', () => copyText(md)],
      ['下載 Word', () => dl(id, 'docx', which)],
    ]));
  };
  if (hc && fm) {
    const tg = document.createElement('div');
    tg.className = 'seg-toggle';
    const add = (label, which) => {
      const b = document.createElement('button');
      b.textContent = label;
      b.addEventListener('click', () => {
        tg.querySelectorAll('button').forEach((x) => x.classList.remove('on'));
        b.classList.add('on');
        show(which);
      });
      tg.appendChild(b);
      return b;
    };
    add('精簡版', 'handcopy').classList.add('on');
    add('完整版', 'full');
    p.appendChild(tg);
  }
  p.appendChild(content);
  show(hc ? 'handcopy' : 'full');
}

function renderSections(p, sections, align) {
  for (const sc of sections) {
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
    p.appendChild(det);
  }
}

function closeDetail() {
  clearTimeout(detailPoll);
  state.detailId = null;
  const a = el.detailBody.querySelector('audio');
  if (a) a.pause();
  el.detail.classList.remove('show');
  if (el.history.classList.contains('show')) renderHistory();
  else if (!el.overlay.classList.contains('hide')) renderRecent();
}

function dl(id, format, which) {
  openFile(`/api/sessions/${id}/export?format=${format}&which=${which}`);
}

// iPhone 主畫面版開附件會蓋掉整個 App 而且回不來，要開到外部 Safari
function openFile(url) {
  if (navigator.standalone) { window.open(url, '_blank'); return; }
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function copyText(t) {
  try {
    await navigator.clipboard.writeText(t.replace(/<!--[\s\S]*?-->/g, '').trim());
    toast('已複製');
  } catch (e) { toast('複製失敗：' + e.message); }
}
$('sum-toggle').addEventListener('click', toggleAllSections);
$('open-history').addEventListener('click', openHistory);
// 標頭那顆：錄音中也能看紀錄，「離開」按下去是會結束錄音的。
$('hist-top').addEventListener('click', openHistory);
el.notes.addEventListener('click', requestNotes);
$('copy-close').addEventListener('click', () => el.copyView.classList.remove('show'));
el.notesOpen.addEventListener('click', () => {
  if (state.notesMd) showHandcopy(state.notesMd, state.notesSections, state.notesCheck);
});
$('select-cancel').addEventListener('click', exitSelect);
el.selectMerge.addEventListener('click', mergeSelected);
$('copy-refresh').addEventListener('click', requestNotes);
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
$('history-close').addEventListener('click', () => {
  exitSelect();
  el.history.classList.remove('show');
  renderRecent();
});
$('detail-close').addEventListener('click', closeDetail);

// ── 上傳 ──────────────────────────────────────────────────────────────
// 分塊送，每塊成功才往前。iPhone 鎖屏或切走時請求會斷，回來問伺服器收到哪裡接著傳。
const CHUNK_BYTES = 4 * 1024 * 1024;
const UP_KEY = 'nobook.uploads';
const up = { items: [], running: false, cancelled: false };

function fmtSize(n) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(n / 1e6))} MB`;
}

function resumeMap() {
  try { return JSON.parse(localStorage.getItem(UP_KEY) || '{}') || {}; } catch (e) { return {}; }
}
function saveResume(item, id) {
  try {
    const m = resumeMap();
    if (id) m[item.key] = id; else delete m[item.key];
    localStorage.setItem(UP_KEY, JSON.stringify(m));
  } catch (e) { /* 沒有 localStorage 就只是不能跨重新整理續傳 */ }
}

function addFiles(files) {
  for (const f of files) {
    const key = `${f.name}|${f.size}|${f.lastModified}`;
    if (up.items.some((i) => i.key === key && i.status !== 'done')) continue;
    up.items.push({ file: f, key, status: 'ready', sent: 0, msg: '' });
  }
  renderUpload();
}

function renderUpload() {
  const list = $('up-list');
  list.innerHTML = '';
  for (const item of up.items) {
    const c = document.createElement('div');
    c.className = 'card uf';
    c.innerHTML = '<div class="ufhead"><span class="ufname"></span><span class="ufsize"></span></div>';
    c.querySelector('.ufname').textContent = item.file.name;
    c.querySelector('.ufsize').textContent = fmtSize(item.file.size);
    if (!up.running && item.status !== 'done') {
      const x = document.createElement('button');
      x.className = 'ufx';
      x.textContent = '×';
      x.setAttribute('aria-label', `移除 ${item.file.name}`);
      x.addEventListener('click', () => {
        up.items = up.items.filter((i) => i !== item);
        renderUpload();
      });
      c.querySelector('.ufhead').appendChild(x);
    }
    if (item.status === 'sending' || item.status === 'waiting') {
      const t = document.createElement('div');
      t.className = 'bar-track';
      t.innerHTML = '<div class="bar-fill"></div>';
      t.firstChild.style.width = `${Math.max(2, (item.sent / item.file.size) * 100)}%`;
      c.appendChild(t);
    }
    const msg = item.msg || (item.status === 'sending'
      ? `上傳中 ${Math.floor((item.sent / item.file.size) * 100)}%` : '');
    if (msg) {
      const m = document.createElement('div');
      m.className = 'ufmsg' + (item.status === 'failed' ? ' err' : item.status === 'done' ? ' ok' : '');
      m.textContent = msg;
      c.appendChild(m);
    }
    list.appendChild(c);
  }
  const pending = up.items.filter((i) => i.status === 'ready' || i.status === 'failed');
  const startBtn = $('up-start');
  startBtn.disabled = up.running || !pending.length;
  startBtn.textContent = up.running ? '上傳中…'
    : pending.length > 1 ? `開始上傳 ${pending.length} 個檔案` : '開始上傳';
  startBtn.hidden = !up.running && !pending.length && up.items.length > 0;
  $('up-tip').hidden = !up.running;
  $('up-form').hidden = up.running || (!pending.length && up.items.length > 0);
  $('up-done').hidden = up.running || !up.items.length
    || !up.items.every((i) => i.status === 'done');
}

function updateUploadForm() {
  $('up-speakers-field').hidden = $('up-course').value !== 'meeting';
  try {
    const n = localStorage.getItem('nobook.upSpeakerCount');
    if (n !== null) $('up-speakers').value = n;
  } catch (e) { /* 用預設 */ }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 等到畫面回來、網路通了再繼續
async function waitForResume(item, wait) {
  item.status = 'waiting';
  item.msg = document.visibilityState === 'hidden'
    ? '暫停中，回到這個畫面會自動接著傳'
    : '連線中斷，恢復後會自動接著傳';
  renderUpload();
  await sleep(wait);
  while (document.visibilityState === 'hidden' && !up.cancelled) await sleep(500);
  item.status = 'sending';
  item.msg = '';
}

async function uploadOne(item, opts) {
  const f = item.file;
  item.status = 'sending';
  item.msg = '';
  item.sent = 0;
  renderUpload();

  // 重新整理或 Safari 把分頁回收之後，選同一個檔要能接著傳
  let uid = resumeMap()[item.key] || null;
  if (uid) {
    try {
      item.sent = (await fetchJson(`/api/uploads/${uid}`)).received;
    } catch (e) { uid = null; item.sent = 0; }
  }
  if (!uid) {
    const made = await postJson('/api/uploads', {
      filename: f.name, size: f.size, course_id: opts.course,
      speakers: opts.speakers, terms: opts.terms,
      recorded_at: f.lastModified ? new Date(f.lastModified).toISOString() : '',
    });
    uid = made.id;
    saveResume(item, uid);
  }

  let wait = 1000;
  while (item.sent < f.size) {
    if (up.cancelled) throw new Error('已取消');
    try {
      const res = await fetch(`/api/uploads/${uid}?offset=${item.sent}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: f.slice(item.sent, item.sent + CHUNK_BYTES),
      });
      if (res.status >= 400 && res.status < 500) {
        const j = await res.json().catch(() => ({}));
        saveResume(item, null);
        throw Object.assign(new Error(j.detail || `HTTP ${res.status}`), { fatal: true });
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      item.sent = (await res.json()).received;
      wait = 1000;
      renderUpload();
    } catch (e) {
      if (e.fatal) throw e;
      await waitForResume(item, wait);
      wait = Math.min(wait * 2, 8000);
      // 斷線那塊可能只收了一半，問清楚再傳
      try { item.sent = (await fetchJson(`/api/uploads/${uid}`)).received; } catch (e2) { /* 下一輪再試 */ }
      renderUpload();
    }
  }
  await postJson(`/api/uploads/${uid}/complete`);
  saveResume(item, null);
  item.status = 'done';
  item.msg = '已上傳，等待轉文字';
  renderUpload();
}

async function startUpload() {
  if (up.running) return;
  const opts = {
    course: $('up-course').value,
    speakers: Number($('up-speakers').value) || 0,
    terms: '',
  };
  try {
    localStorage.setItem('nobook.upCourse', opts.course);
    localStorage.setItem('nobook.upSpeakerCount', String(opts.speakers));
  } catch (e) { /* 不影響上傳 */ }
  up.running = true;
  up.cancelled = false;
  acquireWakeLock(true);
  for (const item of up.items) {
    if (item.status !== 'ready' && item.status !== 'failed') continue;
    try {
      await uploadOne(item, opts);
    } catch (e) {
      item.status = 'failed';
      item.msg = '上傳失敗：' + e.message;
    }
  }
  up.running = false;
  if (state.wakeLock && !state.running) { state.wakeLock.release(); state.wakeLock = null; }
  renderUpload();
}

async function loadUploadCourses() {
  try {
    const list = await (await fetch('/api/courses')).json();
    const sel = $('up-course');
    const meeting = list.filter((c) => c.id === 'meeting');
    const rest = list.filter((c) => c.id !== 'meeting');
    sel.innerHTML = '';
    for (const c of [...meeting, ...rest]) sel.add(new Option(c.name, c.id));
    let last = null;
    try { last = localStorage.getItem('nobook.upCourse'); } catch (e) { /* 用預設 */ }
    if (last && list.some((c) => c.id === last)) sel.value = last;
    updateUploadForm();
  } catch (e) { toast('連不上伺服器'); }
}

function openUpload() {
  if (!up.running && up.items.length && up.items.every((i) => i.status === 'done')) up.items = [];
  el.upload.classList.add('show');
  $('up-file').value = '';
  renderUpload();
  loadUploadCourses();
}

async function closeUpload() {
  if (up.running && !await ask({
    title: '還在上傳，要離開嗎？',
    message: '離開這個畫面不會中斷上傳，但請不要關閉 App。',
    ok: '離開', cancel: '留在這裡',
  })) return;
  el.upload.classList.remove('show');
  renderRecent();
}

$('open-upload').addEventListener('click', openUpload);
$('upload-close').addEventListener('click', closeUpload);
$('up-done-home').addEventListener('click', () => {
  up.items = [];
  el.upload.classList.remove('show');
  renderRecent();
});
$('up-file').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
$('up-course').addEventListener('change', () => {
  $('up-speakers-field').hidden = $('up-course').value !== 'meeting';
});
$('up-start').addEventListener('click', startUpload);
{
  const drop = $('drop');
  for (const ev of ['dragenter', 'dragover']) {
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
  }
  for (const ev of ['dragleave', 'drop']) {
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); });
  }
  drop.addEventListener('drop', (e) => { if (e.dataTransfer) addFiles(e.dataTransfer.files); });
}
// 非 iPhone 不用看語音備忘錄的說明
if (!/iPhone|iPad|iPod/.test(navigator.userAgent)
    && !(navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)) {
  $('ios-howto').hidden = true;
}

// ── 現場錄音入口 ──────────────────────────────────────────────────────
$('open-live').addEventListener('click', () => {
  const noMic = !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia;
  const err = $('mic-error');
  err.hidden = !noMic;
  err.textContent = noMic
    ? '這個網址不能使用麥克風。請改用 https 開頭的網址（Tailscale 提供的那個），上傳錄音檔則不受影響。'
    : '';
  el.start.disabled = noMic || !el.courseSelect.value;
  $('live-setup').classList.add('show');
});
$('live-close').addEventListener('click', () => $('live-setup').classList.remove('show'));
$('note-toggle').addEventListener('click', () => {
  const on = el.note.classList.toggle('show');
  if (on) el.note.focus();
});
$('select-start').addEventListener('click', () => {
  if (state.selecting) exitSelect(); else startSelect();
});
$('select-delete').addEventListener('click', deleteSelected);
{
  let timer = null;
  $('history-search').addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(renderHistory, 200);
  });
}

// ── 封面與外觀 ────────────────────────────────────────────────────────
const WAVE = [14, 26, 40, 30, 58, 82, 60, 96, 128, 92, 150, 118, 76, 132, 100, 62, 88, 50, 72, 36, 48, 24, 32, 14];
$('c-wave').innerHTML = WAVE.map((h, i) => `<i style="--h:${h}px;--d:${(i * 0.09).toFixed(2)}s"></i>`).join('');
$('a-deco').innerHTML = [18, 36, 58, 30, 66, 44, 24, 12].map((h) => `<i style="--h:${h}px"></i>`).join('');

// 重新載入（換版、接回錄音）不用再看一次封面，只有真的重新打開才顯示
function showCover() {
  const cover = $('cover');
  let seen = false;
  try { seen = sessionStorage.getItem('nobook.cover') === '1' || !!recallSession(); } catch (e) { /* 當作沒看過 */ }
  if (seen) return;
  cover.hidden = false;
  const enter = () => {
    if (cover.classList.contains('leaving')) return;
    try { sessionStorage.setItem('nobook.cover', '1'); } catch (e) { /* 下次再看一次而已 */ }
    cover.classList.add('leaving');
    setTimeout(() => { cover.hidden = true; }, 400);
  };
  cover.addEventListener('click', enter);
  cover.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') enter(); });
}

const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4"/></svg>';
const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z"/></svg>';
function paintTheme() {
  const light = document.documentElement.dataset.theme === 'light';
  $('theme-toggle').innerHTML = light ? MOON : SUN;
  document.querySelector('meta[name="theme-color"]').content = light ? '#f6f3ec' : '#0b0d10';
}
$('theme-toggle').addEventListener('click', () => {
  const light = document.documentElement.dataset.theme !== 'light';
  if (light) document.documentElement.dataset.theme = 'light';
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem('nobook.theme', light ? 'light' : 'dark'); } catch (e) { /* 這次有效就好 */ }
  paintTheme();
});
paintTheme();
showCover();

// ── 啟動 ──────────────────────────────────────────────────────────────
async function loadCourses() {
  // 連不上時**不要**默默塞一個假的課程進去。原本的寫法會顯示「範例課程」，
  // 看起來像是課程檔案不見了，實際上只是伺服器沒開——誤導比報錯更糟。
  try {
    const r = await fetchSoon('/api/courses');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const list = await r.json();
    if (!list.length) throw new Error('伺服器沒有回報任何課程');
    el.courseSelect.innerHTML = '';
    for (const c of list) {
      el.courseSelect.add(new Option(c.instructor ? `${c.name} — ${c.instructor}` : c.name, c.id));
    }
    let last = null;
    try { last = localStorage.getItem('nobook.liveCourse'); } catch (e) { /* 用預設 */ }
    if (last && list.some((c) => c.id === last)) el.courseSelect.value = last;
    state.courseId = el.courseSelect.value;
    el.start.disabled = false;
    $('load-error').textContent = '';
    $('conn-error').hidden = true;
  } catch (e) {
    el.courseSelect.innerHTML = '<option value="">（讀不到課程）</option>';
    el.start.disabled = true;
    $('conn-error').hidden = false;
    $('load-error').textContent =
      '請確認手機的 Tailscale App 已開啟連線，以及伺服器那台電腦開著。';
  }
}

$('retry-load').addEventListener('click', () => {
  $('load-error').textContent = '重新連線中…';
  loadCourses();
  renderRecent();
});
el.courseSelect.addEventListener('change', (e) => {
  state.courseId = e.target.value;
  try { localStorage.setItem('nobook.liveCourse', e.target.value); } catch (err) { /* 不影響錄音 */ }
});

function micHelp(e) {
  if (e && e.name === 'NotAllowedError') {
    return '瀏覽器沒有麥克風權限。iPhone 請到「設定 → Safari → 麥克風」改成允許，再回來重新開始。';
  }
  return '這台裝置目前無法錄音（' + (e && e.message) + '）。可以先用手機內建的錄音 App 錄好再上傳。';
}

el.start.addEventListener('click', async () => {
  el.start.disabled = true;
  try {
    await startAudio();
  } catch (e) {
    el.start.disabled = false;
    $('mic-error').textContent = micHelp(e);
    $('mic-error').hidden = false;
    return;
  }
  await acquireWakeLock();
  state.running = true;
  state.startedAt = Date.now();
  el.course.textContent = el.courseSelect.selectedOptions[0]
    ? el.courseSelect.selectedOptions[0].textContent : state.courseId;
  el.overlay.classList.add('hide');
  $('live-setup').classList.remove('show');
  setLight('warn');
  connect();
});

window.addEventListener('beforeunload', (e) => {
  if ((state.running && !state.ended) || up.running) { e.preventDefault(); e.returnValue = ''; }
});

let swReg = null;
if ('serviceWorker' in navigator) {
  const pageVersion = new URL(document.currentScript.src).searchParams.get('v');
  // 網路優先的情況下，打開時拿到的通常已經是新版。只有這一頁真的比較舊才重新載入，
  // 不然每次改版都會多載入一次，封面閃一下。
  navigator.serviceWorker.addEventListener('message', (e) => {
    if (!e.data || e.data.type !== 'version' || e.data.version === pageVersion) return;
    // 錄音中絕對不能自動重載。實測在課堂中途更新版本，平板會直接
    // 重新整理，那堂課就變成孤兒——伺服器端還活著，但前端認不回去。
    // 等這堂結束再換版，課比版本新舊重要。
    const busy = () => (state.running && !state.ended) || up.running;
    if (busy()) {
      toast('有新版本，目前的工作完成後會自動更新', 5000);
      const wait = setInterval(() => {
        if (busy()) return;
        clearInterval(wait);
        location.reload();
      }, 5000);
      return;
    }
    location.reload();
  });
  navigator.serviceWorker.register('sw.js').then((reg) => {
    swReg = reg;
    reg.update().catch(() => { /* 連不上，下次再檢查 */ });
  }).catch(() => { /* 沒有 SW 也能用 */ });
}

loadCourses();
renderRecent();

// 頁面重載後把剛才那堂接回來。沒有這段的話，重載一次就等於把課丟了：
// 伺服器端 session 還在重連佇列裡，但前端不知道自己剛剛在錄哪一堂。
(async function recoverLive() {
  const v = recallSession();
  if (!v) return;
  try {
    const d = await (await fetch(`/api/sessions/${v.id}`)).json();
    if (!d || !d.resumable) { rememberSession(); return; }
    const secs = d.session.duration_s
      || (d.segments.length ? d.segments[d.segments.length - 1].end : 0);
    const name = d.session.note || d.session.course_id;
    if (!await ask({
      title: '要繼續剛才的錄音嗎？',
      message: `「${name}」已錄 ${Math.round(secs / 60)} 分鐘。選「先不要」的話它會留在紀錄裡。`,
      ok: '繼續錄音', cancel: '先不要',
    })) return;
    resumeSession(d);
  } catch (e) { /* 接不回去就算了，資料還在資料庫裡 */ }
})();