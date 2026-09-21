"""單元層自檢：不需要模型、GPU、服務就能跑。

規格 §14 沒有要求這支，但里程碑閘門全都依賴模型與硬體，
在模型下載完成前需要一個能立刻驗證純邏輯正確性的入口。
涵蓋：VAD 切段規則、JSON 防禦性解析、prompt 組裝、Markdown 匯出、
      SQLite CRUD、課程設定檔載入、幻覺過濾、chunk header 編解碼。
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from _common import Gate, header

from server import config, courses, export, storage
from server.audio_pipeline import (FRAME, AudioPipeline, VadSegmenter, is_oom)
from server.summarizer import (Summarizer, extract_json, fmt_ts,
                               repair_truncated_json, strip_fence,
                               strip_think)
from server.ws_session import HEADER, pcm16_to_float32


class ScriptedVad:
    """依 pattern 字串逐 frame 回傳語音/靜音，讓切段規則可精確斷言。

    pattern 的每個字元代表一個 32ms frame：'S' = 語音、'.' = 靜音。
    """

    def __init__(self, pattern):
        self.pattern = pattern
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self, frame):
        ch = self.pattern[self.i] if self.i < len(self.pattern) else "."
        self.i += 1
        return 1.0 if ch == "S" else 0.0


def frames(n):
    """n 個 frame 的音訊（內容不重要，VAD 是腳本化的）。"""
    return np.full(n * FRAME, 0.1, dtype=np.float32)


def ms_to_frames(ms):
    return int(round(ms / config.VAD_FRAME_MS))


# ── VAD 切段（規格 §4.2）──────────────────────────────────────────────
def test_vad(gate):
    fpms = config.VAD_FRAME_MS

    # 1) 靜音超過 MIN_SILENCE_MS → 切段
    speech = ms_to_frames(3000)
    silence = ms_to_frames(config.MIN_SILENCE_MS + 200)
    pat = "." * 5 + "S" * speech + "." * silence + "S" * 10
    seg = VadSegmenter(ScriptedVad(pat))
    out = seg.feed(frames(len(pat)))
    gate.check(len(out) == 1, "靜音超過 MIN_SILENCE_MS 後切段",
               "產出 %d 段" % len(out))
    if out:
        s = out[0]
        expect = (speech + ms_to_frames(config.MIN_SILENCE_MS)) * fpms / 1000.0
        gate.check(abs(s.duration_s - expect) < 0.2 + config.PRE_ROLL_MS / 1000.0,
                   "切出的長度合理（語音 + 靜音門檻 + pre-roll）",
                   "%.2fs（語音 %.2fs）" % (s.duration_s, speech * fpms / 1000.0))
        # 2) pre-roll 往前多帶，起點應早於語音起點
        speech_start = 5 * fpms / 1000.0
        gate.check(s.start_s < speech_start + 1e-6,
                   "PRE_ROLL 往前多帶，不吃掉字首",
                   "段落起點 %.3fs，語音起點 %.3fs" % (s.start_s, speech_start))

    # 3) 短於 MIN_SEGMENT_MS 的片段併入下一段
    short = ms_to_frames(400)
    gap = ms_to_frames(config.MIN_SILENCE_MS + 100)
    long_ = ms_to_frames(2000)
    pat = "S" * short + "." * gap + "S" * long_ + "." * gap
    seg = VadSegmenter(ScriptedVad(pat))
    out = seg.feed(frames(len(pat)))
    gate.check(len(out) == 1, "短於 MIN_SEGMENT_MS 的片段併入下一段，不單獨送 ASR",
               "產出 %d 段（兩段語音合併為一）" % len(out))
    if out:
        gate.check(out[0].duration_s > (short + gap + long_) * fpms / 1000.0 - 0.5,
                   "合併後的段落涵蓋兩段語音", "%.2fs" % out[0].duration_s)

    # 4) MAX_SEGMENT_MS 強制切段
    n = ms_to_frames(config.MAX_SEGMENT_MS + 4000)
    seg = VadSegmenter(ScriptedVad("S" * n))
    out = seg.feed(frames(n))
    gate.check(len(out) >= 1, "連講不停時 MAX_SEGMENT_MS 強制切段",
               "%.0fs 連續語音切出 %d 段" % (n * fpms / 1000.0, len(out)))
    if out:
        gate.check(out[0].duration_s <= config.MAX_SEGMENT_MS / 1000.0 + 0.1,
                   "強制切段不超過 MAX_SEGMENT_MS",
                   "%.2fs（上限 %.1fs）" % (out[0].duration_s,
                                            config.MAX_SEGMENT_MS / 1000.0))

    # 5) flush 把手上的殘段吐出來（不受 MIN_SEGMENT_MS 限制）
    seg = VadSegmenter(ScriptedVad("S" * ms_to_frames(500)))
    seg.feed(frames(ms_to_frames(500)))
    tail = seg.flush()
    gate.check(len(tail) == 1, "flush 吐出未達長度門檻的殘段",
               "%.2fs" % tail[0].duration_s if tail else "無")

    # 6) 逐塊餵入與一次餵入結果一致（residual 處理正確）
    pat = "." * 3 + "S" * ms_to_frames(2500) + "." * ms_to_frames(900)
    audio = frames(len(pat))
    a = VadSegmenter(ScriptedVad(pat)).feed(audio)
    b_seg = VadSegmenter(ScriptedVad(pat))
    b = []
    step = 4000                      # 250ms，非 FRAME 整數倍，測 residual
    for i in range(0, len(audio), step):
        b += b_seg.feed(audio[i:i + step])
    same = (len(a) == len(b)
            and all(abs(x.start_s - y.start_s) < 1e-6 for x, y in zip(a, b)))
    gate.check(same, "250ms 逐塊餵入與整段餵入切點一致（residual 正確）",
               "整段 %d 段 vs 逐塊 %d 段" % (len(a), len(b)))


# ── JSON 防禦性解析（規格 §6.3）───────────────────────────────────────
def test_json(gate):
    cases = [
        ('{"title":"甲","bullets":["a"]}', "純 JSON"),
        ('```json\n{"title":"乙","bullets":["b"]}\n```', "有 ```json 圍欄"),
        ('```\n{"title":"丙","bullets":["c"]}\n```', "有裸圍欄"),
        ('好的，這是結果：{"title":"丁","bullets":["d"]} 以上。', "前後有贅字"),
        ('{"title":"戊{}","bullets":["e\\"f"]}', "字串內含括號與跳脫引號"),
    ]
    ok = 0
    for raw, label in cases:
        obj = extract_json(raw)
        if obj and "title" in obj:
            ok += 1
        else:
            gate.info("解析失敗：%s → %r" % (label, raw[:40]))
    gate.check(ok == len(cases), "JSON 解析涵蓋圍欄／贅字／巢狀括號",
               "%d/%d 種格式都能還原" % (ok, len(cases)))
    gate.check(extract_json("完全不是 JSON") is None,
               "無法解析時回傳 None 讓呼叫端走降級路徑", "回傳 None")
    gate.check(strip_fence("```json\n{}\n```") == "{}", "strip_fence 正確剝除圍欄")

    # 輸出撞到 max_tokens 被砍斷時，前面已經生成完的要點要救得回來，
    # 不要整份丟掉再重跑一次推論（實測 mmdb-2026 那堂切成 6 塊，3 塊撞到）。
    _full = {"title": "類別不平衡",
             "bullets": ["第一則要點。", "第二則要點。", "第三則要點。", "第四則要點。"]}
    _raw = json.dumps(_full, ensure_ascii=False)
    _bad = _with = 0
    for _cut in range(len(_raw)):
        _r = repair_truncated_json(_raw[:_cut])
        if _r is None:
            continue
        if any(k not in _full for k in _r) or any(
                b not in _full["bullets"] for b in _r.get("bullets", [])):
            _bad += 1          # 救出半句，或憑空多出欄位
        if _r.get("bullets"):
            _with += 1
    gate.check(_bad == 0 and _with >= len(_raw) // 3,
               "截斷的輸出能救回完整的要點，不會留半句",
               "%d 個截斷點救回要點、壞掉 %d 個" % (_with, _bad))
    gate.check(repair_truncated_json(_raw) is None,
               "完整的 JSON 不會被誤判成截斷")
    gate.check(repair_truncated_json('{"title": "甲", "bulle') == {"title": "甲"},
               "切在鍵名中間不會留下孤兒鍵")
    _esc = '{"bullets": ["他說\\"會考\\"。", "第二則還沒'
    gate.check(repair_truncated_json(_esc) == {"bullets": ['他說"會考"。']},
               "跳脫引號不會被誤判成字串結尾")

    # Qwen3 hybrid 模型的思考鏈（llama-server 旗標失效時的第二道防線）
    think_cases = [
        ('<think>先想想要寫什麼…</think>{"title":"甲","bullets":["a"]}', "完整 think 標籤"),
        ('<think>\n嗯，這段在講收斂性\n</think>\n```json\n'
         '{"title":"乙","bullets":["b"]}\n```', "think + 圍欄併發"),
    ]
    ok = 0
    for raw, label in think_cases:
        obj = extract_json(raw)
        if obj and obj.get("title"):
            ok += 1
        else:
            gate.info("think 剝除失敗：%s" % label)
    gate.check(ok == len(think_cases), "剝除 <think> 思考鏈後仍能解析 JSON",
               "%d/%d" % (ok, len(think_cases)))
    gate.check(strip_think("<think>沒有收尾就被截斷了").strip() == "",
               "思考鏈被 max_tokens 截斷（無收尾標籤）時整段丟棄")


def test_prompt(gate):
    course = courses.get_course("ml-2026")
    u = Summarizer.build_section_user(course, [], "今天講 gradient descent。")
    gate.check("（本段為課程開頭）" in u and course.name in u,
               "無前文脈絡時標示為課程開頭")
    gate.check(course.glossary[0] in u,
               "glossary 注入摘要 prompt（與 asr_prompt 共用同一份術語來源）")
    prev = [{"title": "前一段", "bullets": ["要點甲", "要點乙"]}]
    u2 = Summarizer.build_section_user(course, prev, "接續上一段。")
    gate.check("前一段" in u2 and "要點甲" in u2, "前文脈絡帶入前一則 section 的標題與要點")

    f = Summarizer.build_final_user(course, [
        {"start_s": 980.0, "title": "梯度下降", "bullets": ["會收斂"],
         "user_note": "這題會考"}])
    gate.check("[00:16:20]" in f, "總結 prompt 帶時間戳", "00:16:20")
    gate.check("這題會考" in f, "總結 prompt 帶使用者標註")
    gate.check("逐字稿" not in f,
               "總結輸入是段落摘要而非逐字稿（規格 §1.2、§6.4）")


# ── Markdown 匯出（規格 §6.4）─────────────────────────────────────────
def test_export(gate):
    course = courses.get_course("ml-2026")
    sections = [
        {"start_s": 980.0, "title": "梯度下降的收斂性",
         "bullets": ["凸函數必收斂", "深度學習只能到局部最小值"],
         "user_note": "這題會考"},
        {"start_s": 2400.0, "title": "regularization",
         "bullets": ["L2 等同 weight decay"], "user_note": None},
    ]
    md = export.build_markdown(
        course, {"id": "x", "started_at": "2026-09-09T09:00:00+08:00"}, sections,
        {"overview": ["這堂課講最佳化", "重點在收斂性"],
         "open_questions": ["鞍點怎麼逃離"]})
    gate.check(md.startswith("# 機器學習 — 2026-09-09"), "標題含課名與日期",
               md.splitlines()[0])
    gate.check("### 梯度下降的收斂性 `[00:16:20]`" in md, "段落標題含時間戳")

    # 段落筆記要有層次：導言 + 子標題 + 要點。二十幾個同一層級的句子
    # 不是筆記，那只是把逐字稿剁碎。
    structured = [{
        "start_s": 0.0, "title": "供應鏈的制度設計",
        "summary": "以品牌商的供應商責任報告為例，說明準則到稽核的流程。",
        "groups": [
            {"heading": "移工保障", "points": ["認證 470 家仲介。", "退還仲介費。"]},
            {"heading": "行為準則", "points": ["842 項準則，35 國執行。"]},
        ],
        "bullets": [], "user_note": None}]
    smd = export.build_markdown(course, {"started_at": "2026-09-21T09:00:00+08:00"},
                                structured, {"overview": ["概述。"],
                                             "open_questions": []})
    gate.check("#### 移工保障" in smd and "#### 行為準則" in smd,
               "子題輸出成第四層標題")
    gate.check(smd.index("以品牌商的供應商責任報告") < smd.index("#### 移工保障"),
               "導言排在子題前面")
    # 舊紀錄沒有 groups，只有平鋪的 bullets，不能整段消失
    legacy = [{"start_s": 0.0, "title": "舊格式", "bullets": ["甲。", "乙。"],
               "user_note": None}]
    lmd = export.build_markdown(course, {"started_at": "2026-09-21T09:00:00+08:00"},
                                legacy, {"overview": ["概述。"],
                                         "open_questions": []})
    gate.check("- 甲。" in lmd and "- 乙。" in lmd and "####" not in lmd,
               "沒有分組的舊紀錄照樣輸出，不會憑空多出子標題")
    gate.check(md.count("> 使用者標註：") == 1, "使用者標註只在有 note 的段落輸出")
    gate.check("## 待釐清" in md and "鞍點怎麼逃離" in md, "待釐清區塊")
    try:
        from markdown_it import MarkdownIt
        MarkdownIt().parse(md)
        gate.check(True, "匯出的 Markdown 可被 markdown-it-py 解析")
    except ImportError:
        gate.skip("匯出的 Markdown 可被 markdown-it-py 解析", "未安裝 markdown-it-py")

    md2 = export.build_markdown(course, {"id": "x", "started_at": ""}, sections,
                                {"overview": ["無"], "open_questions": []})
    gate.check("## 待釐清" not in md2, "沒有待釐清項目時不輸出該區塊")
    gate.check(fmt_ts(3661) == "01:01:01", "時間戳格式 HH:MM:SS", fmt_ts(3661))

    # 本堂重點必須是清單。寫成純文字每行一句的話，Markdown 會把連續幾行
    # 當成同一段的軟換行黏成一大段，轉 Word 也只剩一堆內文段落。
    ov = md.split("## 本堂重點", 1)[1].split("##", 1)[0].strip().splitlines()
    gate.check(ov and all(x.startswith("- ") for x in ov),
               "本堂重點輸出成 Markdown 清單", " / ".join(ov)[:60])

    # docx：要轉成 Word 的原生樣式，不是把 # 和 - 原樣塞進去
    try:
        import io

        from docx import Document

        from server import docx_export
        doc = Document(io.BytesIO(docx_export.markdown_to_docx_bytes(md)))
        styles = {}
        for par in doc.paragraphs:
            styles[par.style.name] = styles.get(par.style.name, 0) + 1
        leftover = [par.text for par in doc.paragraphs
                    if par.text.lstrip().startswith(("#", "- ", "> ")) or "**" in par.text]
        gate.check(styles.get("Heading 1") == 1 and styles.get("Heading 2", 0) >= 2,
                   "docx 的 # / ## 轉成 Heading 樣式", str(styles))
        gate.check(styles.get("List Bullet", 0) >= 5, "docx 的 - 轉成 List Bullet",
                   str(styles.get("List Bullet")))
        gate.check(styles.get("Intense Quote", 0) == 1, "docx 的 > 轉成引用樣式")
        gate.check(not leftover, "docx 裡沒有殘留的 Markdown 字元",
                   "; ".join(leftover)[:60])
    except ImportError:
        gate.skip("匯出的 docx 使用 Word 原生樣式", "未安裝 python-docx")


# ── 學習單模式的手抄版 ────────────────────────────────────────────────
def test_worksheet(gate):
    """有些課要邊上課邊填固定欄位的學習單，下課就交。"""
    import yaml

    from server import courses as courses_mod
    from server.summarizer import (build_worksheet_fields,
                                   build_worksheet_schema, worksheet_counts,
                                   worksheet_groups)

    cfg = {
        "id": "ws-test", "name": "測試課",
        "handcopy": {
            "target_chars": 1500,
            "sections": [
                {"id": "content", "title": "今天上課內容", "hint": "講了什麼"},
                {"id": "impressive", "title": "印象深刻的部分", "pick": 2},
                {"id": "reflection", "title": "今日反思", "pick": 1},
            ],
        },
    }
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "ws-test.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        course = courses_mod.load_course("ws-test", d)
        plain = courses_mod.load_course("ml-2026", str(Path(__file__).resolve()
                                                      .parent.parent / "courses"))

    gate.check(course.is_worksheet and len(course.handcopy_sections) == 3,
               "課程設定檔能定義學習單欄位",
               "%d 欄" % len(course.handcopy_sections))
    gate.check(not plain.is_worksheet,
               "沒設定 handcopy 的課維持原本的主題式手抄版")
    gate.check(course.handcopy_target_chars == 1500,
               "學習單的目標字數可以逐課覆寫")

    # 沒有 maxItems 的話模型會把整份學習單寫完再從頭寫一遍，撞爆 max_tokens
    schema = build_worksheet_schema(course.handcopy_sections)
    fields = schema["properties"]["fields"]
    gate.check(sorted(fields["required"]) == ["content", "impressive", "reflection"],
               "schema 要求每一個欄位都要有內容")
    caps = {k: (v["minItems"], v["maxItems"]) for k, v in fields["properties"].items()}
    gate.check(all(lo >= 1 and hi > lo for lo, hi in caps.values())
               and caps["impressive"][1] > 2,
               "每個欄位都有則數上下限，且備選比實際要抄的多", str(caps))
    gate.check(worksheet_counts({"pick": 0})[1] > worksheet_counts({"pick": 1})[1],
               "沒指定 pick 的欄位（整堂內容）給得比反思類的多")
    gate.check(worksheet_counts({"pick": 2, "count": [18, 34]}) == (18, 34),
               "設定檔的 count 可以蓋過由 pick 推出來的則數")
    # 素材只有 11 則卻硬要 18 則，模型就會開始編規格數字
    lo, hi = worksheet_counts({"pick": 0, "count": [18, 34]}, 11)
    gate.check(hi <= 14 and lo < 18,
               "素材不夠時則數跟著縮，不要逼模型湊數量", "(%d, %d)" % (lo, hi))
    gate.check(worksheet_counts({"pick": 0, "count": [18, 34]}, 105) == (18, 34),
               "素材夠多時維持設定檔的則數")

    # 接地過濾：編出來的規格數字不能抄到紙上，但名詞解釋要留著
    from server.summarizer import drop_ungrounded, to_traditional
    hay = ("IaaS 提供基礎設施服務，包含 CPU、RAM。SaaS 透過瀏覽器使用，"
           "如 Google Meet。PaaS 提供開發與部署環境。")
    keep, dropped = drop_ungrounded([
        "IaaS（Infrastructure as a Service）：提供 CPU、RAM 等資源。",
        "設定為 4 vCPU 與 8 GB RAM。",
        "Google Meet 最多支援 100 人同時線上會議。",
    ], hay, ["IaaS", "SaaS", "PaaS"])
    gate.check(len(keep) == 1 and keep[0].startswith("IaaS"),
               "縮寫的英文全稱不會被當成幻覺刪掉", "留下 %d 則" % len(keep))
    gate.check(len(dropped) == 2,
               "素材裡查無依據的數字規格會被擋下來",
               "；".join(c[:16] for c, _ in dropped))
    # 中文詞後面掛英文譯名也是名詞解釋；掰出來的產品名（有斜線／版本號）不是
    k2, d2 = drop_ungrounded([
        "虛擬機器（Virtual Machine）：可安裝作業系統。",
        "AI 工具在前端網頁（React/Vue.js）上自動部署。",
    ], "虛擬機器可安裝作業系統。AI 工具可自動部署前端網頁。", [])
    gate.check(len(k2) == 1 and k2[0].startswith("虛擬機器")
               and len(d2) == 1,
               "中文術語的英文譯名留著，掰出來的產品名擋掉",
               "留 %d 丟 %d" % (len(k2), len(d2)))

    gate.check(to_traditional("尤其適閤中小企業") == "尤其適合中小企業"
               and to_traditional("閤家歡樂") == "閤家歡樂",
               "罕用異體字修正不會誤傷真的要用閤的詞")
    # OpenCC 的台灣用語層有「导出→匯出」，會把正確的「推導出」改壞
    # OpenCC 的台灣用語層有三條規則會把正確的繁體改壞。掃過 80 個常用詞，
    # 只有這三條：导出→匯出、程序→程式、权限→許可權。
    opencc_cases = [
        ("这不是直觉能推导出的", "這不是直覺能推導出的"),
        ("把筆記匯出成 Word", "把筆記匯出成 Word"),   # 真的要講匯出的不能動
        ("将进入调查程序", "將進入調查程序"),
        ("這支程式跑得很快", "這支程式跑得很快"),     # 真的要講程式的不能動
        ("人权限制", "人權限制"),
        ("存取权限", "存取權限"),
        ("授权限制", "授權限制"),
        ("信息安全", "資訊安全"),
    ]
    bad = [(a, to_traditional(a), b) for a, b in opencc_cases
           if to_traditional(a) != b]
    gate.check(not bad, "OpenCC 台灣用語層的誤傷都修掉了，且沒有誤傷正確的詞",
               "；".join("%s→%s 應為 %s" % x for x in bad)[:70])

    # 學習單的品質門檻：同一個心得句型反覆出現、句子長到抄不完、
    # 以及替使用者編造他沒說過的個人經歷
    from server.summarizer import fabricated_past, worksheet_quality_issues
    templated = ["我原本以為 A，這讓我發現 B。" * 3,
                 "我原本以為 C，這讓我覺得 D。" * 3,
                 "我原本以為 E，我才發現 F。" * 3]
    varied = ["虛擬化宣稱的隔離，前提是有設定資源配額，否則單一 VM 會拖垮同機其他 VM。",
              "判斷要不要用 SaaS，之後會先問資料落在誰的邊界內，而不是只看部署快不快。",
              "課堂把 PaaS 的價值放在免管伺服器，但跨環境的變數一致性沒有被解決。"]
    gate.check(len(worksheet_quality_issues(templated)) >= 2
               and worksheet_quality_issues(varied) == [],
               "抓得出模板化與過長的欄位，寫得好的不會被誤判",
               "；".join(worksheet_quality_issues(templated))[:50])
    # 整欄都是問句讀起來像在出考題，不像在想事情
    all_q = ["如果 A 成立，是否會導致 B？", "若 C 未設定，是不是就會 D？",
             "當 E 發生時，F 會不會失效？", "如果 G，H 是否仍然成立？"]
    mixed = ["虛擬化宣稱的隔離，前提是有設定資源配額，否則同機的 VM 會互相拖累。",
             "課堂把 PaaS 的價值放在免管伺服器，但跨環境變數一致性沒有被解決。",
             "如果 GPU 只以 IaaS 出租，驅動與最佳化仍落在客戶身上，這是隱性成本。",
             "SaaS 的即時可用與資料主權互相牽制，前者越強後者越弱。"]
    # 開頭都不同、但骨架一樣的欄位（「A 若未 B，可能導致 C」×N）
    from server.summarizer import ensure_sentence
    mono = ["IaaS 的虛擬機器若未配置網路隔離，可能導致不同租戶資料串流。",
            "PaaS 部署流程若缺乏版本控制，可能無法追蹤程式變更歷史。",
            "企業租用虛擬機器時，若未界定存取權限，可能違反資料主權法規。",
            "SaaS 的即時存取若未加密，可能使資料在傳輸中遭竊取。",
            "CSP 的伺服器若未定期更新修補，可能成為攻擊入口。"]
    gate.check(any("骨架" in x for x in worksheet_quality_issues(mono)),
               "抓得出開頭都不同、但句型骨架一樣的欄位",
               "；".join(worksheet_quality_issues(mono))[:44])
    gate.check(ensure_sentence("CSP 與 ISP 的區別在於前者專注雲端資源")
               .endswith("。")
               and ensure_sentence("這樣對嗎？") == "這樣對嗎？",
               "缺句號直接補上，本來就有收尾的不動")

    frag = ["IaaS 的虛擬機器租用模式，其資源分配依賴於電算中心的負載平衡系統",
            "企業租用 GPU 的彈性，建立在 CSP 擁有異質計算資源池的整合能力之上",
            "資料集中於雲端後的同步效率，取決於網路延遲與壓縮演算法的最佳化"]
    gate.check(any("沒有句號" in x for x in worksheet_quality_issues(frag)),
               "抓得出寫成沒講完的名詞片語的欄位")
    # keep_min 不可以把跟別欄幾乎一樣的句子放回來
    from server.summarizer import dedupe_against
    dup_src = ["雲端服務從產品銷售轉為服務訂購，隱含對長期客戶關係管理的重新設計。"]
    dup_new = ["雲端服務從產品銷售轉為服務訂購，隱含對長期客戶關係管理的重新設計，"
               "這與我過去僅關注技術層面的思維不同。"]
    k3, _ = dedupe_against(dup_new, dup_src, keep_min=2)
    gate.check(k3 == [],
               "欄位太短時寧可留白，也不把別欄那句話原樣放回來")

    qi = worksheet_quality_issues(all_q)
    gate.check(any("問句" in x for x in qi) and any("條件句" in x for x in qi)
               and worksheet_quality_issues(mixed) == [],
               "抓得出整欄都是問句／都用條件句開場，句式混合的不會被誤判",
               "；".join(qi)[:60])
    gate.check(len(fabricated_past(["我去年做學生專案時因資源不足導致訓練中斷。"])) == 1
               and fabricated_past(["下次遇到 AI 訓練需求會先查 GPU 支援。"]) == [],
               "替使用者編造的個人經歷會被丟掉")

    # 截斷救回來的內容尾巴會黏 JSON 標點，那些字元會被照抄到紙上
    from server.summarizer import strip_json_artifacts
    keep_mid = "正常的一句話，含逗號在中間，沒問題。"
    gate.check(strip_json_artifacts("此為人類不可取代之處。'," ) == "此為人類不可取代之處。"
               and strip_json_artifacts('"]}正常內容。') == "正常內容。"
               and strip_json_artifacts(keep_mid) == keep_mid,
               "漏進文字的 JSON 標點會被清掉，句中的標點不受影響")

    # 記錄型與感受型分開跑，上課內容那一欄才拿得到完整的輸出預算
    rec, ref = worksheet_groups(course)
    gate.check([x["id"] for x in rec] == ["content"]
               and [x["id"] for x in ref] == ["impressive", "reflection"],
               "欄位分成記錄型與感受型兩組",
               "記錄型 %d 欄、感受型 %d 欄" % (len(rec), len(ref)))
    gate.check("exam" in build_worksheet_schema(rec, with_exam=True)["properties"]
               and "exam" not in build_worksheet_schema(ref, with_exam=False)["properties"],
               "會考的重點只在記錄型那一次問，不要問兩遍")

    prompt_fields = build_worksheet_fields(course.handcopy_sections)
    gate.check(all(x["title"] in prompt_fields for x in course.handcopy_sections)
               and "講了什麼" in prompt_fields,
               "欄位標題與提示都有寫進 prompt")

    # 渲染：欄位要編號、備選要標「挑幾則」，缺的欄位不能無聲消失
    notes = {"worksheet": True, "exam": ["這題會考"],
             "topics": [
                 {"title": "今天上課內容", "points": ["甲", "乙"], "pick": 0},
                 {"title": "印象深刻的部分",
                  "points": ["丙", "丁", "戊", "己"], "pick": 2},
                 {"title": "今日反思", "points": [], "pick": 1}]}
    md = export.build_handcopy(course, {"started_at": "2026-09-16T14:00:00+08:00"},
                               notes)
    gate.check("## 一、今天上課內容" in md and "## 三、今日反思" in md,
               "學習單欄位照設定檔的順序編號輸出")
    gate.check("挑 2 則抄" in md and "共 4 則可選" in md,
               "備選欄位標出要抄幾則、有幾則可選")
    gate.check("這欄沒產出內容" in md,
               "欄位空白時明講，不要讓使用者到紙本前才發現少一欄")
    gate.check(md.index("★ 老師說會考") < md.index("一、今天上課內容"),
               "會考的重點仍排在最前面")

    plain_md = export.build_handcopy(
        plain, {"started_at": "2026-09-16T14:00:00+08:00"},
        {"topics": [{"title": "主題", "points": ["甲"]}], "exam": []})
    gate.check("## 主題" in plain_md and "一、" not in plain_md,
               "一般手抄版不受學習單的編號影響")


# ── 儲存層（規格 §10）─────────────────────────────────────────────────
def test_storage(gate):
    with tempfile.TemporaryDirectory() as d:
        storage.init(Path(d) / "t.db")
        storage.create_session("s1", "ml-2026", "2026-09-09T09:00:00+08:00")
        storage.insert_segment("s1", 0.0, 3.0, "第一句", -0.2)
        storage.insert_segment("s1", 3.0, 6.0, "第二句", -0.4)
        sid = storage.insert_section("s1", 1, 0.0, 6.0, "標題", ["a", "b"], "註解")
        gate.check(len(storage.list_segments("s1")) == 2, "segments 寫入與查詢")
        secs = storage.list_sections("s1")
        gate.check(secs and secs[0]["bullets"] == ["a", "b"],
                   "sections 的 bullets 以 JSON array 存取")
        storage.update_section(sid, 0.0, 9.0, "新標題", ["c"], "新註解")
        secs = storage.list_sections("s1")
        gate.check(secs[0]["title"] == "新標題" and secs[0]["bullets"] == ["c"],
                   "section 更新（短 buffer 合併時會用到）")
        gate.check(len(storage.segments_between("s1", 0.0, 3.0)) == 1
                   and len(storage.segments_between("s1", 0.0, 9.0)) == 2,
                   "依時間範圍取逐字稿（合併重生成時用）")
        storage.finish_session("s1", "2026-09-09T11:00:00+08:00", 7200.0, "# md")
        gate.check(storage.get_session("s1")["final_md"] == "# md", "session 收尾寫入")
        storage.close()


# ── 課程設定檔（規格 §9.3）────────────────────────────────────────────
def test_courses(gate):
    c = courses.get_course("ml-2026")
    gate.check(c.name == "機器學習" and "overfitting" in c.glossary,
               "課程 YAML 載入（name / glossary）")
    gate.check("施皇嘉" in c.asr_prompt,
               "asr_prompt 從 YAML 讀取，供 Whisper 當 initial_prompt")
    gate.check(courses._estimate_tokens(c.asr_prompt) <= courses.ASR_PROMPT_MAX_TOKENS,
               "asr_prompt 在 Whisper 的 200 token 安全上限內",
               "估計 %d token" % courses._estimate_tokens(c.asr_prompt))
    long_prompt = "測試" * 400
    gate.check(len(courses._truncate_prompt(long_prompt, "t")) < len(long_prompt)
               or courses._estimate_tokens(long_prompt) <= 200,
               "超長 asr_prompt 會被截斷而不是整段丟給 Whisper")
    ids = [c["id"] for c in courses.list_courses()]
    gate.check("ml-2026" in ids and "general" in ids, "列出所有課程", str(ids))


# ── ASR 後處理（規格 §4.4）────────────────────────────────────────────
def test_postprocess(gate):
    from server.asr import clean_text
    kept = clean_text("這個模型的 loss 會 converge", None)
    gate.check(kept == "這個模型的 loss 會 converge", "正常文字保留")
    drops = ["請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
             "字幕由 Amara.org 社群提供", "嗯嗯嗯", "。", "字"]
    bad = [t for t in drops if clean_text(t, None)]
    gate.check(not bad, "幻覺尾巴與過短輸出被丟棄", "未被過濾：%s" % bad if bad else "全數過濾")
    gate.check(clean_text("  多  餘   空白  ", None) == "多 餘 空白", "空白正規化")


def test_oom_detect(gate):
    gate.check(is_oom(RuntimeError("CUDA failed with error out of memory")),
               "CUDA OOM 訊息可被辨識（CTranslate2 只拋普通 RuntimeError）")
    gate.check(not is_oom(ValueError("bad shape")), "一般例外不會被誤判為 OOM")


def test_header(gate):
    pcm = (np.arange(4000, dtype=np.int16) % 1000)
    frame = HEADER.pack(12345, len(pcm)) + pcm.tobytes()
    seq, cnt = HEADER.unpack_from(frame, 0)
    out = pcm16_to_float32(frame[HEADER.size:])
    gate.check(seq == 12345 and cnt == 4000 and len(out) == 4000,
               "chunk header 編解碼（uint32 seq + uint32 sampleCount, LE）")
    gate.check(abs(out[999] - 999 / 32768.0) < 1e-6, "PCM16 → float32 換算正確")


# ── pipeline 佇列行為 ─────────────────────────────────────────────────
async def test_pipeline(gate):
    class Eng:
        def __init__(self):
            self.n = 0
            self.concurrent = 0
            self.max_concurrent = 0

        def transcribe_joined(self, audio, prompt="", *a, **k):
            import time
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            time.sleep(0.02)
            self.n += 1
            self.concurrent -= 1
            return {"text": "第%d段" % self.n, "avg_logprob": -0.3, "rtf": 0.1}

    got = []
    eng = Eng()
    pat = ("S" * ms_to_frames(1500) + "." * ms_to_frames(800)) * 6
    pipe = AudioPipeline(eng, on_result=lambda s, r: got.append(r["text"]),
                         vad=ScriptedVad(pat))
    pipe.start()
    audio = frames(len(pat))
    for i in range(0, len(audio), 4000):
        await pipe.feed(audio[i:i + 4000])
    await pipe.flush()
    await asyncio.wait_for(pipe.drain(), timeout=30)
    await pipe.stop()
    gate.check(len(got) >= 5, "pipeline 產出多段逐字稿", "%d 段" % len(got))
    gate.check(got == sorted(got, key=lambda t: int(t[1:-1])),
               "單一 worker 序列化推論，逐字稿順序不亂", " / ".join(got[:4]))
    gate.check(eng.max_concurrent == 1,
               "不會並行推論（8GB VRAM 承受不住，規格 §4.2）",
               "最大並行數 %d" % eng.max_concurrent)


def main() -> int:
    header("selftest.py — 純邏輯自檢（不需模型／GPU／服務）")
    gate = Gate("SELFTEST")
    print("\n[VAD 切段規則 §4.2]")
    test_vad(gate)
    print("\n[JSON 防禦性解析 §6.3]")
    test_json(gate)
    print("\n[Prompt 組裝 §6.3 §6.4]")
    test_prompt(gate)
    print("\n[Markdown 匯出 §6.4]")
    test_export(gate)
    print("\n[SQLite 儲存 §10]")
    test_worksheet(gate)
    test_storage(gate)
    print("\n[課程設定檔 §9.3]")
    test_courses(gate)
    print("\n[ASR 後處理 §4.4]")
    test_postprocess(gate)
    print("\n[OOM 偵測 §7.4]")
    test_oom_detect(gate)
    print("\n[音訊 chunk 協定 §4.1]")
    test_header(gate)
    print("\n[ASR 佇列行為 §4.2]")
    asyncio.run(test_pipeline(gate))
    print("\n[WebSocket 端到端 §5 §6]")
    try:
        import _wstest
    except ImportError as e:
        gate.skip("WebSocket 端到端流程", "缺少測試依賴：%s" % e)
    else:
        _wstest.run(gate)
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
