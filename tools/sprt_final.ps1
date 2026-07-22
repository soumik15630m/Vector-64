# Final merge gate: the ENTIRE search-elo branch vs current main, native
# PowerShell. If this is a clear positive, the branch merges to main.
#
#   powershell -ExecutionPolicy Bypass -File tools\sprt_final.ps1               # 25k nodes
#   powershell -ExecutionPolicy Bypass -File tools\sprt_final.ps1 -Nodes 50000  # deeper
#
# Binaries in sprt\ (gitignored):
#   sprt\base.exe      current main   (futility + IIR + SEE, +109 Elo)
#   sprt\conthist.exe  search-elo tip (base + singular + capture history
#                                      + continuation history / history-LMR)
#
# The features are depth-sensitive (continuation history especially), so this
# runs deep. SPRT: H0 elo<=0, H1 elo>=5, alpha=beta=0.05, stops at LLR +/-2.94.

param(
  [int]$Nodes = 25000,
  [int]$Games = 6000,
  [int]$Conc  = 10
)

$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $PSScriptRoot
$net    = Join-Path $root "runs\v2\stk_halfka_1024.nnue"
$bin    = Join-Path $root "sprt"
$logdir = Join-Path $bin  "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

if (-not (Test-Path $net)) { Write-Error "net not found: $net"; exit 1 }
foreach ($b in @("base","conthist")) {
  if (-not (Test-Path (Join-Path $bin "$b.exe"))) {
    Write-Error "missing binary: $bin\$b.exe (see header)"; exit 1
  }
}

$log = Join-Path $logdir ("final_{0}n.log" -f $Nodes)
$sw  = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host ""
Write-Host "===================================================================="
Write-Host "  STK-Vector-64  --  FINAL merge gate (whole branch vs main)"
Write-Host "===================================================================="
Write-Host ("  new  : conthist.exe  (singular + capture + continuation history)")
Write-Host ("  base : base.exe      (current main, +109 Elo)")
Write-Host ("  net  : {0}" -f (Split-Path $net -Leaf))
Write-Host ("  test : {0} nodes/move  |  up to {1} games  |  {2} workers" -f $Nodes, $Games, $Conc)
Write-Host ("  gate : H0 elo<=0  H1 elo>=5   (accept at LLR +/-2.94)")
Write-Host ("  log  : {0}" -f $log)
Write-Host "===================================================================="
Write-Host "  Clear positive => merge search-elo to main. Ctrl-C to stop."
Write-Host "--------------------------------------------------------------------"

python (Join-Path $root "tools\nnue\match.py") `
  --engine (Join-Path $bin "conthist.exe") --base-engine (Join-Path $bin "base.exe") `
  --net $net --base-net $net `
  --sprt 0 5 --games $Games --nodes $Nodes `
  --concurrency $Conc --seed 909090 2>&1 | Tee-Object -FilePath $log

$sw.Stop()
Write-Host "--------------------------------------------------------------------"
Write-Host ("  finished in {0:n1} min. Full log: {1}" -f $sw.Elapsed.TotalMinutes, $log)
Write-Host "  Paste the final: line -- clear positive and I merge to main."
Write-Host "===================================================================="
