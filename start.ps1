# 一鍵啟動：檢查前置條件 → 起服務 → 經 Tailscale 對外 → 印出手機要開的網址
#
# 用法：
#   .\start.ps1              啟動並保持在前景（Ctrl+C 停止）
#   .\start.ps1 -CheckOnly   只檢查前置條件，不啟動

param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$tailscale = "C:\Program Files\Tailscale\tailscale.exe"
$port = 8000

function Fail($msg, $fix) {
    Write-Host "  [X] $msg" -ForegroundColor Red
    if ($fix) { Write-Host "      $fix" -ForegroundColor Yellow }
    return $false
}
function Ok($msg, $detail) {
    Write-Host "  [v] $msg" -ForegroundColor Green
    if ($detail) { Write-Host "      $detail" -ForegroundColor DarkGray }
    return $true
}

Write-Host "`n=== 前置檢查 ===" -ForegroundColor Cyan
$allOk = $true

# 1. bench.json：服務啟動時會讀，缺了會拒絕啟動（規格 §14.1）
$bench = Join-Path $root "data\bench.json"
if (Test-Path $bench) {
    $b = Get-Content $bench -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($b.asr.model) {
        Ok "bench.json 已定案" ("ASR " + (Split-Path $b.asr.model -Leaf) +
            " / 課中 " + $b.summary_model_inclass.model) | Out-Null
    } else {
        $allOk = (Fail "bench.json 缺少 asr.model" "先跑 python scripts\bench_vram.py")
    }
} else {
    $allOk = (Fail "找不到 data\bench.json" "先跑 python scripts\bench_vram.py")
}

# 2. 殘留的 llama-server 會佔住 VRAM 與 port
$stray = @(Get-Process llama-server -ErrorAction SilentlyContinue)
if ($stray.Count -eq 0) {
    Ok "沒有殘留的 llama-server" | Out-Null
} else {
    Write-Host "  [!] 發現 $($stray.Count) 個殘留的 llama-server，正在清除" -ForegroundColor Yellow
    $stray | Stop-Process -Force
    Start-Sleep -Seconds 2
}

# 3. GPU
$smi = "C:\Windows\System32\nvidia-smi.exe"
if (Test-Path $smi) {
    $used = (& $smi --query-gpu=memory.used --format=csv,noheader) -replace '\D',''
    Ok "GPU 可用" "目前已佔用 $used MiB"  | Out-Null
} else {
    $allOk = (Fail "找不到 nvidia-smi" "確認 NVIDIA 驅動已安裝")
}

# 4. Tailscale：手機要連進來一定要走它的 HTTPS（規格 §8.3、§9.1）
$tsOk = $false
if (Test-Path $tailscale) {
    $status = & $tailscale status 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dns = (& $tailscale status --json 2>$null | ConvertFrom-Json).Self.DNSName
        if ($dns) { $dns = $dns.TrimEnd('.') }
        $tsOk = $true
        Ok "Tailscale 已登入" $dns | Out-Null
    } else {
        $allOk = (Fail "Tailscale 未登入" "執行：tailscale up  （會開瀏覽器要你登入）")
    }
} else {
    $allOk = (Fail "Tailscale 未安裝" "winget install tailscale.tailscale")
}

if (-not $allOk) {
    Write-Host "`n前置條件未滿足，請先處理上列項目。`n" -ForegroundColor Red
    exit 1
}
if ($CheckOnly) { Write-Host "`n前置檢查全部通過。`n" -ForegroundColor Green; exit 0 }

# ── 啟動 ────────────────────────────────────────────────────────────────
Write-Host "`n=== 啟動服務 ===" -ForegroundColor Cyan
Push-Location $root
try {
    $server = Start-Process python -ArgumentList "-m","server.main" -PassThru -NoNewWindow

    # 等 /api/health 回應
    $ready = $false
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-WebRequest "http://127.0.0.1:$port/api/health" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        if ($server.HasExited) { break }
    }
    if (-not $ready) {
        Write-Host "  [X] 服務在 120 秒內未就緒" -ForegroundColor Red
        if (-not $server.HasExited) { $server | Stop-Process -Force }
        exit 1
    }
    Ok "服務已就緒" "http://127.0.0.1:$port" | Out-Null

    # Tailscale Serve：自動申請並續期 Let's Encrypt 憑證，不需自簽（規格 §9.1）
    & $tailscale serve --bg --https=443 "http://localhost:$port" 2>&1 | Out-Null
    $dns = (& $tailscale status --json 2>$null | ConvertFrom-Json).Self.DNSName
    if ($dns) { $dns = $dns.TrimEnd('.') }

    Write-Host "`n=== 手機上開這個網址 ===" -ForegroundColor Cyan
    Write-Host "    https://$dns`n" -ForegroundColor White
    Write-Host "  手機要裝 Tailscale 並登入同一帳號。" -ForegroundColor DarkGray
    Write-Host "  必須走 https —— getUserMedia 只在 secure context 下可用。" -ForegroundColor DarkGray
    Write-Host "`n  Ctrl+C 停止服務`n" -ForegroundColor DarkGray

    Wait-Process -Id $server.Id
} finally {
    Pop-Location
    & $tailscale serve --https=443 off 2>&1 | Out-Null
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "`n已停止服務並收掉 Tailscale Serve。" -ForegroundColor DarkGray
}
