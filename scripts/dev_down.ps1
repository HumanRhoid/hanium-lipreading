# 로컬 스택 종료:  powershell -File scripts\dev_down.ps1
#   -Purge   도커 볼륨(DB 기록·저장 영상)까지 삭제. 평소에는 쓰지 말 것

param(
    [switch]$Purge
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# dev_up이 띄운 uvicorn·worker 창의 파이썬 프로세스를 찾아 내린다.
# 이 저장소 venv의 python만 대상 — 다른 파이썬은 건드리지 않는다.
$venvExe = (Resolve-Path ".\.venv\Scripts\python.exe" -ErrorAction SilentlyContinue).Path
if ($venvExe) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.ExecutablePath -eq $venvExe -and $_.CommandLine -match "uvicorn|worker_main" } |
        ForEach-Object {
            Write-Host "종료: PID $($_.ProcessId)  $(($_.CommandLine -split ' ')[-1])"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

if ($Purge) {
    Write-Host "도커 컨테이너 + 볼륨 삭제 (DB·영상 데이터 소멸)" -ForegroundColor Yellow
    docker compose down -v
} else {
    Write-Host "도커 컨테이너 정지 (데이터 보존)"
    docker compose stop
}
Write-Host "종료 완료" -ForegroundColor Green
