# 로컬 전체 스택 기동. 저장소 루트에서:  powershell -File scripts\dev_up.ps1
#   -Front <경로>   lipread-connect 위치를 주면 프론트 dev 서버까지 띄운다
#
# 하는 일: 도커 인프라 → DB 마이그레이션 → API 창 → Worker 창 (→ 프론트 창)
# 끝낼 때: scripts\dev_down.ps1  (각 창은 Ctrl+C 또는 창 닫기로도 됨)

param(
    [string]$Front = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# ── 사전 점검 ──────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Write-Host ".env가 없습니다. cp .env.example .env 후 값을 확인하세요." -ForegroundColor Red
    exit 1
}
try { docker version --format "{{.Server.Version}}" | Out-Null }
catch {
    Write-Host "Docker Desktop이 꺼져 있습니다. 켜고 다시 실행하세요." -ForegroundColor Red
    exit 1
}
$ckpt = @(Get-ChildItem "checkpoints\release192_seed*.pt" -ErrorAction SilentlyContinue)
if ($ckpt.Count -lt 3) {
    Write-Host "checkpoints\release192_seed*.pt 3개가 필요합니다 (현재 $($ckpt.Count)개)." -ForegroundColor Red
    Write-Host "INFERENCE_BACKEND=local이면 Worker가 기동을 거부합니다."
    exit 1
}

# ── 인프라 ─────────────────────────────────────────────────────────
Write-Host "[1/4] 도커 인프라 (postgres·minio·redis)" -ForegroundColor Cyan
docker compose up -d postgres minio minio-init redis | Out-Null

Write-Host "[2/4] DB 대기 후 마이그레이션" -ForegroundColor Cyan
$ok = $false
foreach ($i in 1..20) {
    docker exec hanium-lipreading-postgres-1 pg_isready -U postgres 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Start-Sleep 1
}
if (-not $ok) { Write-Host "postgres가 20초 안에 준비되지 않았습니다." -ForegroundColor Red; exit 1 }
& "$root\.venv\Scripts\python.exe" -m alembic upgrade head

# ── 앱 프로세스 (창 분리 — 로그를 따로 보고 따로 끌 수 있게) ─────────
Write-Host "[3/4] API 서버 창 + 추론 Worker 창" -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory $root -ArgumentList @(
    "-NoExit", "-Command",
    "`$host.UI.RawUI.WindowTitle='lipreading API :8000'; .\.venv\Scripts\python.exe -m uvicorn src.backend.main:app --port 8000"
)
Start-Process powershell -WorkingDirectory $root -ArgumentList @(
    "-NoExit", "-Command",
    "`$host.UI.RawUI.WindowTitle='lipreading Worker'; .\.venv\Scripts\python.exe -m src.backend.recognition.worker_main"
)

# ── 준비 확인 ──────────────────────────────────────────────────────
Write-Host "[4/4] /health/ready 대기" -ForegroundColor Cyan
$ready = $false
foreach ($i in 1..60) {
    try {
        $r = Invoke-RestMethod "http://localhost:8000/health/ready" -TimeoutSec 2
        if ($r.status -eq "ready") { $ready = $true; break }
    } catch {}
    Start-Sleep 2
}
if ($ready) { Write-Host "준비 완료: API·DB·추론 모두 ready" -ForegroundColor Green }
else { Write-Host "120초 안에 ready가 되지 않았습니다. API 창의 로그를 확인하세요." -ForegroundColor Yellow }

# ── 프론트 (선택) ──────────────────────────────────────────────────
if ($Front -ne "") {
    if (Test-Path (Join-Path $Front "package.json")) {
        Write-Host "프론트 dev 서버 창 (http://localhost:5173)" -ForegroundColor Cyan
        Start-Process powershell -WorkingDirectory $Front -ArgumentList @(
            "-NoExit", "-Command",
            "`$host.UI.RawUI.WindowTitle='lipread-connect :5173'; npm run dev"
        )
        Start-Process "http://localhost:5173"
    } else {
        Write-Host "-Front 경로에 package.json이 없습니다: $Front" -ForegroundColor Yellow
    }
} else {
    Write-Host "프론트까지 띄우려면:  powershell -File scripts\dev_up.ps1 -Front <lipread-connect 경로>"
}
