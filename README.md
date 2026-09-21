# lecture-scribe — 課堂即時轉錄與筆記系統

手機或平板當錄音端，自己的筆電當推論伺服器，兩邊用 Tailscale 連。
逐字稿、段落摘要、下課要交的手抄筆記全部在本地 GPU 上算完，
**不呼叫任何雲端 API，錄音與逐字稿不離開自己的機器**。

寫來解決一個具體問題：上課要邊聽邊在紙上寫筆記，一下課就得交。
所以重點不是「課後產一份漂亮的整理」，而是**課還在上的時候就要有
一份能直接照著抄的東西**，而且抄得完。

**硬體門檻**：一張 8GB 的 NVIDIA 顯示卡。開發與實測都在 RTX 4060 Laptop
上進行，ASR 與 LLM 要輪流共用這 8GB，模型選型與切換邏輯都是繞著這個限制設計的。

## 能做什麼

- **即時逐字稿**：說完一句約 1 秒內出現（實測中位數 0.88s、P95 1.66s）
- **分段整理**：上課中按一次，把「上次按到現在」整理成一則段落筆記
- **手抄版筆記**：隨時按，把目前為止的段落彙整成一份能直接抄到紙上的筆記，
  錄音不中斷
- **學習單模式**：有些課要填固定欄位的學習單（今天上課內容／印象深刻的部分／
  想法／反思），可以在課程設定裡定義欄位，系統照欄位產出，每欄多給幾則讓你挑
- **課後完整筆記**：下課後切到較大的模型重新整理，同時產出手抄版與完整版
- **匯出**：Markdown 與 Word（`.docx` 用 Word 原生樣式，不是把 `#` 原樣塞進去）
- **同一天多段合併**：一堂課分好幾次錄的話，可以把錄音接起來重新產生總筆記
- **近似音修正**：用課程術語表把逐字稿裡讀音相近的錯字改回來
- **接地檢查**：標出筆記裡「逐字稿找不到依據」的數字與專有名詞，
  避免把模型自己補的東西抄到紙上交出去

## 這個專案不做什麼

- 不支援 CPU 推論。ASR 在 CPU 上跑不到即時。
- 不是會議記錄工具。整個流程是為「一個人、一門課、要手抄」設計的。
- 沒有多使用者、沒有帳號、沒有雲端同步。它預設只有你自己會連。
- PWA 沒有 foreground service，**錄音時切到其他 App 或關螢幕就會斷線**。
  斷線時前端會繼續錄音並在本地累積最多 60 秒，重連後補送。

---

## 實測結果（RTX 4060 Laptop 8GB）

規格書上的 VRAM／延遲／RTF 都是估計值，以下是這台機器上實際量到的；
完整內容在 [data/bench.json](data/bench.json)。

```
GPU 基準      桌面佔用 608 MB → 可用 7580 MB
ASR           Breeze-ASR-25 int8_float16
              穩態 1601 MB、推論峰值 2104 MB、RTF 0.077
課中模型      Qwen3-4B-Instruct-2507-Q4_K_M  ctx 8192  峰值 5669 MB  P95 10.7s
課後模型      Qwen3-8B-Q4_K_M                ctx 8192  峰值 5830 MB  總結 5.7s
ASR 卸載      釋放 1499 MB（佔用的 94%）
模型切換      課中→課後 4.6s、切回 3.1s
```

三小時連續運轉：1701 段、RTF 平均 0.236／P95 0.381、端到端延遲
中位數 0.88s／P95 1.66s、private commit 在 2.8 小時穩態下 +1 MB。

### 閘門狀態

| 閘門 | 狀態 |
|---|---|
| G0 環境與模型可用性 | 8/8 |
| G1 即時逐字稿（真實錄音） | 8/8 |
| G2 分段摘要 | 12/12 |
| G3 模型切換與課後總結 | 8/8 |
| G4 韌性（混沌測試） | 11/11 |
| selftest（純邏輯自檢） | 109/109 |
| M5 實機使用 | 已在多堂真實課堂使用 |

`initial_prompt`（課程術語表）對辨識正確率的影響，在真實錄音上實測
**+46.4%**。同一份比較在合成 TTS 音檔上只有 +6.2%——差別完全來自
量測路徑，合成音檔乾淨到出現天花板效應，對照組失去意義。
會特別寫這一段，是因為前者才是這個專案裡最值得花時間的地方。

---

## 安裝

一次把所有外部資產抓下來：

```bash
pip install -r requirements.txt
python scripts/fetch_assets.py     # ASR 模型 + llama.cpp + Qwen3 GGUF（約 13 GB）
python scripts/make_fixture.py     # 產生暫代的測試音檔（見第 5 節）
python scripts/bench_vram.py       # 量測硬體能力，決定摘要模型選型
python scripts/verify_m0.py        # 閘門 G0
```

`verify_m0.py` 會逐項印出還缺什麼與具體的修復指令，隨時可以重跑。
以下是各項的細節。

### 1. Python 環境

規格 §2 要求 Python 3.11。這台機器上跑的是 3.9，實測所有依賴都能裝、
CTranslate2 也認得到 CUDA，所以沒有另建 3.11 環境——
若之後遇到套件相容問題，先懷疑這一點。

```bash
pip install -r requirements.txt
```

`torch` 是 `silero-vad` 的依賴。這台機器上裝的是 CUDA 版（`2.6.0+cu124`），
順帶解決了下一節的 cuDNN 問題。VAD 本身跑 CPU，不佔 VRAM；
ASR 走 CTranslate2，不經 torch。

### 2. cuDNN 9

`faster-whisper` 依賴 cuDNN 9（規格 §2），這是最常見的失敗點，
`verify_m0.py` 會單獨驗這一項。

**這台機器不需要另外裝**：cuDNN 9.1 與 cuBLAS 的 dll 已隨 `torch` 的 CUDA
wheel 附在 `site-packages/torch/lib` 下。[server/cuda_dlls.py](server/cuda_dlls.py)
會在匯入 `server` 套件時處理路徑問題。

它做了兩件事，而**第二件才是關鍵**：

1. `os.add_dll_directory` 掛上目錄（Python 3.8+ 的 Windows 不再吃 PATH 找
   extension module 的相依 dll）
2. 用**絕對路徑**把 `cudart` → `cublasLt` → `cublas` → `cudnn_*` 依序預載進行程

只做第 1 件不夠：模型載得起來、`verify_m0` 的 cuDNN 檢查也會過，但一開始
推論就會炸 `Library cublas64_12.dll is not found or cannot be loaded`。
原因是 CTranslate2 用**裸檔名**呼叫 `LoadLibrary` 去延遲載入 cuBLAS/cuDNN，
那條路徑不吃我們加的目錄。而同名模組只要已經在行程裡，後續的 `LoadLibrary`
就會直接拿到它，不再去檔案系統找——所以預載有效。

若換了環境找不到，就補上其中一種：

```bash
pip install nvidia-cudnn-cu12
# 或設環境變數指向自己的 cuDNN
set LS_CUDA_DLL_DIR=C:\path\to\cudnn\bin
```

### 3. ASR 模型（規格 §3.3）

Breeze-ASR-25 的原始權重是 HuggingFace transformers 格式，
**不能**直接被 faster-whisper 載入，需要 CTranslate2 格式。
`fetch_assets.py` 走規格的第一選項——社群已轉換好的
`phate334/Breeze-ASR-25-int8-CT2` → `models/breeze-asr-25-ct2`。

若該版本失效，自行轉換：

```bash
pip install ctranslate2 transformers
ct2-transformers-converter \
  --model MediaTek-Research/Breeze-ASR-25 \
  --output_dir models/breeze-asr-25-ct2 \
  --quantization int8_float16 \
  --copy_files tokenizer.json preprocessor_config.json
```

### 4. 摘要模型與 llama-server

`fetch_assets.py` 會抓 llama.cpp 的 Windows CUDA 12.4 預編譯版到
`tools/llama.cpp/`（驅動支援到 CUDA 12.7，所以不能用 13.x 版），
[server/config.py](server/config.py) 會自動找到它，不必手動設 PATH。

GGUF 檔放在 `models/` 下。**檔名與規格 §7.2 有出入**：規格寫的是
`Qwen3-8B-Instruct` / `Qwen3-4B-Instruct` / `Qwen3-1.7B-Instruct`，
但 Qwen 官方 GGUF 只有 `Qwen3-8B` / `Qwen3-4B` / `Qwen3-1.7B`
（hybrid thinking 模型，沒有 `-Instruct` 後綴），真正的 instruct-only
版本只出到 4B。所以實際採用：

```
models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf   # 課中首選，真 instruct 版
models/Qwen3-8B-Q4_K_M.gguf                 # 課後總結候選，hybrid
models/Qwen3-1.7B-Q4_K_M.gguf               # 保底
models/Qwen3-4B-Instruct-2507-Q4_0.gguf     # 更省的備案
```

hybrid 模型會輸出 `<think>…</think>`，那會吃掉 8 秒的延遲預算並污染 JSON。
[llm_manager.py](server/llm_manager.py) 啟動時帶 `--reasoning-budget 0` 關掉它，
旗標不被認得就退掉重試；[summarizer.py](server/summarizer.py) 另外再剝一層。

找不到檔案的候選會被標記 `MISSING` 並跳過，不會讓量測失敗——
所以可以只先下載 4B 跑起來，之後再補 8B 重跑一次 bench。

### 5. 測試音檔

規格 §14.3 要的是**真實**的長篇中文演講錄音：

```
tests/fixtures/lecture_3h.wav   # G1 的 3 小時模擬串流
tests/fixtures/m0_sample.wav    # G0 的快速健檢（一兩分鐘即可）
```

在拿到真實錄音之前，`make_fixture.py` 會用 Windows SAPI（zh-TW）合成一段
含已知術語的講稿當暫代品，並輸出 `*.groundtruth.json` 記錄每個術語的出現次數。

**這只夠把管線接通。** TTS 音訊乾淨得不真實，量出來的 RTF 與字錯率都會過度樂觀，
術語命中率的對照也失去意義（乾淨音訊本來就不太需要 `initial_prompt` 幫忙）。
真實驗收看規格 §11 M5。

---

## 量測與驗證（規格 §13、§14.7）

**先量測，再實作。** 規格書上所有 VRAM / 延遲 / RTF 數字都是估計值，
config 參數一律從 `data/bench.json` 推導，不 hard-code。
服務啟動時會讀這個檔，檔案不存在或缺少 `asr.model` 就拒絕啟動。

```bash
python scripts/bench_vram.py          # 決定摘要模型選型 → data/bench.json
python scripts/verify_m0.py           # G0 環境與模型可用性
python scripts/verify_m1.py           # G1 即時逐字稿（完整 3 小時模擬）
python scripts/verify_m2.py           # G2 按鈕摘要（需服務已啟動）
python scripts/verify_m3.py           # G3 期末總結與模型切換
python scripts/verify_m4.py           # G4 混沌測試
python scripts/verify_all.py          # 全跑一次，回歸測試用
```

閘門未通過就不要開始下一個里程碑（規格 §11）。`verify_all.py` 會在第一個
失敗的閘門停下。

常用參數：

```bash
python scripts/verify_m1.py --hours 0.25   # 縮短模擬時長（記憶體洩漏項改為參考值）
python scripts/verify_m1.py --fast         # 不做真實節奏，只驗正確性
python scripts/verify_m2.py --start-server # 自己拉起 uvicorn
python scripts/verify_all.py --skip m2     # 跳過需要服務的閘門
```

`scripts/selftest.py` 不在規格的腳本清單裡，是額外補的純邏輯自檢
（VAD 切段規則、JSON 防禦性解析、prompt 組裝、Markdown 匯出、SQLite、
WebSocket 狀態機），不需要模型／GPU／服務，用來在模型下載完成前先確認邏輯正確。

---

### 量測時踩到的坑

這幾個都是「數字看起來合理、實際上量錯對象」的錯誤，全都是跑出結果後
回頭核對計算邏輯才發現的。留在這裡免得重蹈。

**1. 殘留的 llama-server 會讓整份 bench 靜靜地失效。**
一個沒關乾淨的 llama-server 同時汙染兩件事：吃掉 VRAM 讓 `baseline_used_mb`
虛高（實測 3926 MB vs 實際 478 MB），以及佔著 port 8080 讓每個候選的
`/health` 都打到它——於是「啟動成功」了，但量的自始至終是同一個陌生行程。
產出的結論（不同大小的模型佔用幾乎一樣、課中只能用純 CPU）看起來完全合理。
現在 `bench_vram.py` 開跑前會掃殘留行程，`LLMManager` 啟動前檢查 port、
啟動後查 `/v1/models` 確認模型身分不符就拋 `LLMPortBusy`。

**2. VRAM 峰值與可用量的基準要一致。**
`nvidia-smi` 的用量是絕對值（含桌面的約 480 MB），而 `available_mb` 已經
扣掉基準了。兩者直接相比會讓門檻無故緊縮約 1.2 GB，在 8 GB 的機器上足以
把塞得下的模型誤判成 OOM。現在 `peak_mb` 一律扣掉基準，另存 `peak_abs_mb`
供對照。

**3. 能量的就不要推估——尤其推估值落在門檻附近時。**
課後總結原本用 `FINAL_MAX_TOKENS(3000) ÷ tok/s` 推估耗時，但 3000 是
max_tokens **上限**不是預期輸出長度（實際約 600 token）。推估 65 秒、
實測 5.4 秒，差 12 倍，而且剛好卡在 63 秒門檻上把 8B 刷掉。現在改成直接
量真實形狀的總結呼叫（15 則 section 的輸入），對齊 G3 的驗收條件。

**4. 兩個階段的速度標準不同。**
課中是「一則 400 token 的段落摘要 8 秒內」（§7.2，使用者按了按鈕在等）；
課後是「15 則 section 的總結 < 90 秒」（§11 G3，一次性、看著進度條等）。
把課中的 8 秒套到課後，會把 8B 誤判為太慢而退回小模型，正好違背 §7.3
「不要偷懶直接沿用」。

**5. 記憶體判準的基準點。**
CUDA context、cuDNN workspace、CTranslate2 buffer 會在前約 11 分鐘的音訊內
配置完畢（961→997 MB）之後完全持平，穩態成長率 1.5 MB/小時。這段一次性
成本約 430 MB，若從「模型載入後」起算，規格 §11 的 200 MB 門檻會讓一個
毫無洩漏的系統被判為洩漏。基準因此取在穩態之後（音訊時間 900 秒後），
量的才是規格真正想問的「有沒有持續成長」；報告會同時列出兩個數字。

**6. 合成音檔會讓某些判定失去鑑別力。**
循環播放短音檔會讓「重複輸出迴圈」偵測誤判（內容本來就重複）；乾淨的 TTS
音訊讓 ASR 不靠 `initial_prompt` 就全對，術語命中率提升自然是 0%。這兩項
現在會偵測到素材限制並標為無法判定，而不是報 FAIL——FAIL 會讓人以為
實作有缺陷。

---

## 啟動

```bash
python -m server.main                                  # http://127.0.0.1:8000
tailscale serve --bg --https=443 http://localhost:8000 # 對外（規格 §9.1）
```

平板瀏覽器開 `https://<機器名>.<tailnet>.ts.net`。

**必須走 HTTPS。** `getUserMedia` 只在 secure context 下可用，
HTTPS 頁面連 `ws://` 會被 mixed content 擋掉。不要用自簽憑證——
Android Chrome 對自簽憑證的 WebSocket 支援很差（規格 §8.3）。

### 環境變數

| 變數 | 預設 | 用途 |
|---|---|---|
| `LS_HOST` / `LS_PORT` | `127.0.0.1` / `8000` | 服務位址 |
| `LS_LLAMA_SERVER_BIN` | `llama-server` | llama.cpp 執行檔路徑 |
| `LS_LLAMA_PORT` | `8080` | llama-server 埠號 |
| `LS_LLAMA_VERBOSE` | — | 設任意值以顯示 llama-server 日誌 |
| `LS_ASR_MODEL` | 由 bench.json 決定 | 覆寫 ASR 模型路徑 |
| `LS_NVIDIA_SMI` | 自動尋找 | nvidia-smi.exe 路徑 |
| `LS_SAVE_AUDIO` | `0` | 設 `1` 保存原始音訊（規格 §10） |
| `LS_VAD_THRESHOLD` 等 | 見 `server/config.py` | 覆寫切段參數 |

---

## 使用

1. 手機／平板開啟頁面 → 選課程 → **開始錄音**（會要求麥克風權限並取得 wake lock）
2. 逐字稿即時滾動。手動往上捲會停止自動捲動，右下出現「回到最新」
3. 老師講完一段就按 **即時整理**
   - 把「上次按到現在」整理成一則段落筆記
   - 逐字稿很長時會自動切成幾段，同一次按鈕產生的會收在同一個容器裡
   - 可以先在輸入框寫註解，註解會跟著那一段進到最終筆記
   - 連按無效：前一則完成前的重複按鈕會被忽略
   - 30 秒內二次按鈕不新開段落，而是併回前一則重新生成
4. 想看能抄的版本就按 **產生筆記**：把目前為止的段落彙整成一份手抄版，
   **錄音不會停**，隨時可以再按一次刷新
5. 下課按 **結束課程** → 卸載 ASR → 切到較大的模型 → 同時產出手抄版與完整版

沒有正常結束的課（手機睡著、網路斷掉）會留在紀錄裡標成「未結束」，
點進去有「結束並產生筆記」可以補收尾；逐字稿都還在，不會掉。

---

## 課程設定檔

`courses/<id>.yaml`，一門課一個檔，放進去就會出現在前端的下拉選單。

不想設定的話直接用內建的 `general`（一般錄音）就能跑。為某一門課調過的
設定跟它的差別只有詞彙相關的兩項：

```yaml
id: ml-2026
name: 機器學習
instructor: 王小明

# 給 Whisper 當 initial_prompt。這是整個專案裡影響辨識正確率最大的一項
# （真實錄音實測 +46.4%）。長度自動限制在 200 token 內（Whisper 上限 224）。
asr_prompt: |
  以下是資訊管理研究所的機器學習課程錄音。
  常見術語：overfitting、regularization、gradient descent、交叉驗證。

# 兩個用途：注入摘要模型的 system prompt（讓它照原樣寫這些詞），
# 以及近似音修正（把逐字稿裡讀音接近術語的錯字改回來）。
glossary:
  - overfitting
  - regularization
  - gradient descent
  - 交叉驗證
```

詞怎麼挑：**先錄一堂，看逐字稿裡哪些詞被聽錯，把正確寫法填進去。**
憑空想像會出現什麼詞，通常猜不中老師實際的用語。

### 學習單模式（選用）

有些課要邊上課邊填固定欄位的學習單。定義欄位之後，手抄版就照欄位產出：

```yaml
handcopy:
  target_chars: 5200
  sections:
    - id: content
      title: 今天上課內容
      count: [20, 60]        # 這一欄要產幾則
      style: detail          # 要寫細節與名詞解釋，不是條列標題
      hint: 照時間順序涵蓋整堂課；專有名詞、縮寫、數字規格都要有一則解釋
    - id: reflection
      title: 今日反思
      pick: 1                # 實際只會抄 1 則，其餘是備選
      count: [5, 8]
      hint: 這一欄的主角是「我」：原本用什麼標準判斷、哪裡太粗糙、下次會怎麼做
```

幾個實作上的判斷，都是踩過才加的：

- **則數會跟著素材量縮**。設定要 20 則但逐字稿只有 11 條素材時，模型會
  開始編規格數字湊數量（實測編出「4 vCPU 8GB RAM」「Google Meet 最多 100 人」
  這種逐字稿裡沒有的東西）。寧可短，不可以編。
- **每一欄分開產生**。擠在同一次呼叫時輸出會互相同化，「想法」和「反思」
  會變成同一批內容換句話說。
- **產出後會量句型**。開頭重複率、心得句型佔比、問句比例、是否整欄都同一個
  句型骨架——超標就帶著具體理由重寫一次。光在 prompt 裡寫「不要重複」沒有用。
- **不會替使用者編造個人經歷**。模型寫過「我去年做學生專案時…」這種它
  不可能知道的事，這份是要交出去的，一律丟掉。

---

## 專案結構

```
lecture-scribe/
├── server/
│   ├── main.py              FastAPI app、路由註冊
│   ├── ws_session.py        WebSocket 連線處理與 session 狀態機
│   ├── audio_pipeline.py    ring buffer → VAD → 切段 → ASR 佇列
│   ├── vad.py               silero-vad 封裝（可注入，方便測試）
│   ├── asr.py               faster-whisper 封裝、模型載入/卸載、後處理
│   ├── summarizer.py        llama-server 客戶端、prompt 組裝、JSON 防禦解析
│   ├── llm_manager.py       llama-server 子行程生命週期、模型切換
│   ├── storage.py           SQLite schema 與 CRUD
│   ├── export.py            Markdown / TXT / JSON 匯出
│   ├── docx_export.py       Markdown → Word 原生樣式
│   ├── merge.py             同一天多段錄音合併、重新產生總筆記
│   ├── term_fix.py          用術語表做近似音修正（pypinyin）
│   ├── verify_notes.py      接地檢查：標出逐字稿找不到依據的內容
│   ├── audio_store.py       整堂錄音存成 FLAC（選用）
│   ├── courses.py           課程設定檔載入
│   ├── gpu.py               nvidia-smi 封裝（VRAM 查詢）
│   ├── cuda_dlls.py         Windows 上預載 cuDNN/cuBLAS
│   ├── winjob.py            Job Object，確保子行程跟著主行程結束
│   └── config.py            環境變數、常數、bench.json 載入
├── courses/                 課程設定檔（general.yaml 是不需設定的預設）
├── web/                     PWA 靜態檔（FastAPI StaticFiles 掛載）
├── scripts/                 量測與自驗證腳本（verify_m0~m4、selftest）
├── models/                  模型檔（不進版控，用 fetch_assets.py 下載）
└── data/                    bench.json 進版控；lecture.db 與錄音不進
```

---

## 設計要點

- **摘要用按鈕觸發，不做自動定時摘要。** 使用者的按鈕代表人為判斷的段落邊界。
- **期末總結由段落摘要合成，不餵原始逐字稿。** ASR 逐字稿含錯字與口語贅詞。
- **上課用小模型、課後換大模型。** VRAM 只有 8GB，靠模型生命週期管理換品質。
- **逐字稿是核心，摘要是加值。** 任何資源衝突下都犧牲摘要保逐字稿（規格 §7.4）：
  - ASR 推論前檢查 VRAM 餘量，低於 400MB 推入延遲佇列而非直接 OOM
  - 攔截 CUDA OOM → 卸載摘要模型 → 重試 ASR → 回送降級 error 事件
  - llama-server 掛掉、磁碟寫入失敗都不會中斷逐字稿
- **單一 ASR worker，不並行推論。** 8GB VRAM 承受不住，而且會打亂順序。
- **`condition_on_previous_text=False` 是必須的**，否則長音訊必然進入重複輸出迴圈。

---

## 已知風險（規格 §12）

收音品質是**對準確度影響最大的單一變因，超過任何模型選擇**。
強烈建議搭配指向性或領夾式麥克風。

M5（實機驗證）無法腳本化，必須真的去上一堂課。前面的閘門只是確保
不會帶著已知缺陷走進教室。

**筆記是模型寫的，不保證正確。** 系統會標出逐字稿裡找不到依據的數字與
專有名詞，但它抓不到「ASR 把某個詞聽成另一個同樣合理的詞」——那種錯誤
在逐字稿裡看起來完全正常。要交出去的東西，自己看過一遍。

---

## 授權

個人專案，沒有選授權條款。要拿去用的話自己斟酌。
