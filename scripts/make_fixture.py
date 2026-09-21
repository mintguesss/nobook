"""用 Windows SAPI（zh-TW）合成一段課堂講稿當測試音檔。

**這不是規格 §14.3 要的東西。** 規格要的是真實的長篇中文演講錄音，
因為 TTS 音訊乾淨得不真實，量出來的 RTF 與字錯率都會過度樂觀。

它的用途只有兩個：
  1. 在拿到真實錄音之前，先把 M1 的管線接通、跑得起來
  2. 講稿內容是已知的，所以術語命中率有明確的 ground truth 可對照

拿到真實課堂錄音後，放成 tests/fixtures/lecture_3h.wav，
verify_m1.py 會優先用它，這支腳本的產物就退居備位。

用法：
    python scripts/make_fixture.py                 # 產生約 3 分鐘的 m0_sample.wav
    python scripts/make_fixture.py --minutes 20 --out lecture_synth.wav
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import ROOT, header

FIXTURES = ROOT / "tests" / "fixtures"

# 每段之間會插入停頓，讓 VAD 切得出段落（規格 §4.2 的 MIN_SILENCE_MS = 600ms）
SCRIPT = [
    "好，我們今天繼續上禮拜的內容，先講 gradient descent 的收斂性。",
    "在凸函數的情況下，只要 learning rate 選得夠小，"
    "gradient descent 一定會收斂到全域最小值。",
    "可是深度學習的 loss surface 並不是凸的，"
    "所以我們只能保證收斂到局部最小值，或者是鞍點。",
    "接下來講 regularization。L2 regularization 其實就是在 loss 後面"
    "加一個 weight decay 項，它可以有效抑制 overfitting。",
    "那 overfitting 要怎麼判斷呢？最直接的做法就是 cross validation，"
    "把資料切成幾份輪流當驗證集。",
    "講到模型評估，就一定要提混淆矩陣。"
    "從混淆矩陣可以算出精確率跟召回率。",
    "召回率在醫療診斷這種情境特別重要，因為漏掉一個病人的代價非常高。",
    "再來我們看 transformer。transformer 最核心的機制是 attention，"
    "它讓每個位置都可以直接看到序列裡的其他位置。",
    "attention 的輸入是 embedding，embedding 就是把離散的 token"
    "映射到連續向量空間。",
    "最後講一下特徵工程。雖然深度學習號稱可以自動學特徵，"
    "但在表格資料上，好的特徵工程還是常常打敗複雜的模型。",
    "好，那今天就到這邊，下禮拜小考範圍就是 regularization 跟 cross validation。",
    "我們再回頭補充一點 gradient descent 的變形。"
    "batch gradient descent 每次要看完整個資料集，計算量太大。",
    "所以實務上都用 mini-batch，batch size 通常取 32 到 256 之間。",
    "stochastic gradient descent 就是 batch size 等於一的極端情況，"
    "它的梯度估計雜訊很大，但有時候反而幫助跳出局部最小值。",
    "講到 attention 的計算複雜度，它對序列長度是平方級的，"
    "這也是為什麼長文本的 transformer 這麼吃記憶體。",
    "有一些改進方法像 sparse attention 或 linear attention，"
    "想把複雜度降到接近線性，但品質通常會有一些損失。",
    "embedding 的維度也是一個超參數，維度太低表達能力不足，"
    "太高又容易 overfitting，而且訓練變慢。",
    "至於 cross validation 的折數，常見是五折或十折。"
    "折數越多估計越準，但要訓練的次數也越多。",
    "混淆矩陣裡的四個格子分別是真陽性、假陽性、真陰性、假陰性。"
    "精確率看的是分母是預測為陽性的數量，召回率的分母是實際為陽性的數量。",
    "如果資料極度不平衡，準確率這個指標會非常誤導，"
    "這時候要看 F1 分數或是 PR 曲線底下的面積。",
    "特徵工程常用的手法包括分箱、交叉特徵、目標編碼，"
    "還有針對時間序列的滯後特徵。",
    "做目標編碼的時候要特別小心資料洩漏，"
    "一定要在 cross validation 的每一折內部各自計算。",
    "regularization 除了 L1 跟 L2，dropout 也算廣義的正則化，"
    "它在訓練時隨機把一部分神經元關掉。",
    "early stopping 也是，當驗證集的 loss 開始上升就停下來，"
    "這其實是在限制模型的有效容量。",
]

# 術語出現次數直接從講稿數出來，避免手寫的數字跟內容走鐘
GLOSSARY = ["gradient descent", "learning rate", "regularization", "overfitting",
            "cross validation", "混淆矩陣", "召回率", "transformer", "attention",
            "embedding", "特徵工程"]


def count_terms(lines):
    body = "".join(lines).lower()
    return {t: body.count(t.lower()) for t in GLOSSARY}


GROUND_TRUTH = count_terms(SCRIPT)

PS_TEMPLATE = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'zh-*' }} | Select-Object -First 1
if ($null -eq $voice) {{ Write-Error 'no zh voice'; exit 2 }}
$s.SelectVoice($voice.VoiceInfo.Name)
$s.Rate = {rate}
$s.SetOutputToWaveFile('{out}')
$text = Get-Content -Path '{txt}' -Encoding UTF8 -Raw
foreach ($line in ($text -split "`n")) {{
    $line = $line.Trim()
    if ($line.Length -eq 0) {{ continue }}
    $s.Speak($line)
    $s.Speak([System.Speech.Synthesis.PromptBuilder]::new())
    Start-Sleep -Milliseconds 1
}}
$s.SetOutputToNull()
$s.Dispose()
Write-Output ('OK ' + $voice.VoiceInfo.Name)
"""


def synth(lines, out_wav: Path, rate: int = 0) -> str:
    """用 SAPI 合成到 wav。回傳所選語音名稱。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        # 段落之間補標點停頓，讓 VAD 有明確的切點
        f.write("\n".join(lines))
        txt = f.name
    ps = PS_TEMPLATE.format(rate=rate, out=str(out_wav).replace("'", "''"),
                            txt=txt.replace("'", "''"))
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", ps],
                           capture_output=True, text=True, timeout=900)
    finally:
        Path(txt).unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError("SAPI 合成失敗：%s%s" % (r.stdout, r.stderr))
    return r.stdout.strip()


def to_16k_mono(src: Path, dst: Path) -> float:
    """SAPI 產出的是 22kHz/16bit；轉成規格要求的 16kHz 單聲道。"""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype("float32")
    sf.write(str(dst), data, 16000, subtype="PCM_16")
    return len(data) / 16000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="m0_sample.wav")
    ap.add_argument("--minutes", type=float, default=0,
                    help="重複講稿直到達到這個長度（0 = 只講一遍）")
    ap.add_argument("--rate", type=int, default=0, help="SAPI 語速 -10~10")
    args = ap.parse_args()

    header("make_fixture.py — 合成測試音檔（非規格要求，真實錄音的暫代品）")
    if sys.platform != "win32":
        print("  只支援 Windows SAPI")
        return 1

    FIXTURES.mkdir(parents=True, exist_ok=True)
    lines = list(SCRIPT)
    if args.minutes:
        # 依實測：SAPI 唸一句約 8 秒
        one_pass_min = len(SCRIPT) * 8 / 60.0
        rounds = max(1, int(args.minutes / one_pass_min + 0.5))
        lines = lines * rounds
        print("  重複講稿 %d 輪以逼近 %.1f 分鐘" % (rounds, args.minutes))

    raw = FIXTURES / "_sapi_raw.wav"
    out = FIXTURES / args.out
    print("  合成中…（%d 句）" % len(lines), flush=True)
    voice = synth(lines, raw, args.rate)
    dur = to_16k_mono(raw, out)
    raw.unlink(missing_ok=True)

    print("  %s" % voice)
    print("  已產生 %s（%.1f 秒，16kHz 單聲道 PCM16）" % (out, dur))

    gt = FIXTURES / (out.stem + ".groundtruth.json")
    import json
    counts = dict(GROUND_TRUTH)
    if args.minutes:
        rounds = len(lines) // len(SCRIPT)
        counts = {k: v * rounds for k, v in counts.items()}
    gt.write_text(json.dumps(
        {"source": "Windows SAPI TTS（非真實錄音）",
         "voice": voice, "duration_s": round(dur, 1),
         "term_counts": counts,
         "note": "TTS 音訊過於乾淨，RTF 與字錯率會比真實課堂樂觀，"
                 "僅供管線接通用；真實驗收見規格 §11 M5"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("  ground truth 已寫到 %s" % gt.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
