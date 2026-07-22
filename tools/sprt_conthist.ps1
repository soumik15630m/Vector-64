# Validate continuation history + history-based LMR, native PowerShell.
# Live, detailed progress on screen AND saved to a log.
#
#   powershell -ExecutionPolicy Bypass -File tools\sprt_conthist.ps1              # 25k nodes
#   powershell -ExecutionPolicy Bypass -File tools\sprt_conthist.ps1 -Nodes 50000 # deeper
#
# Binaries in sprt\ (gitignored):
#   sprt\allfeat.exe    base for THIS test (singular + capture + lazy + SMP),
#                       no continuation history  (bench 3158101 @ d12/1thr)
#   sprt\conthist.exe   allfeat + continuation history + history-based LMR
#                       (bench 3200474 @ d12/1thr)
#
# Continuation history is an ORDERING feature: it only expresses in a deep
# enough tree. At 8k nodes the tree is too shallow and fixed-node undervalues
# it (that is why the earlier attempts read negative). This runs at 25k+.
# SPRT: H0 elo<=0, H1 elo>=5, alpha=beta=0.05, stops at LLR +/-2.94.

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
foreach ($b in @("allfeat","conthist")) {
  if (-not (Test-Path (Join-Path $bin "$b.exe"))) {
    Write-Error "missing binary: $bin\$b.exe (see header)"; exit 1
  }
}

$log = Join-Path $logdir ("conthist_{0}n.log" -f $Nodes)
$sw  = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host ""
Write-Host "===================================================================="
Write-Host "  STK-Vector-64  --  continuation history + history-LMR SPRT"
Write-Host "===================================================================="
Write-Host ("  new  : conthist.exe   (allfeat + continuation history + hist-LMR)")
Write-Host ("  base : allfeat.exe    (singular + capture + lazy + SMP)")
Write-Host ("  net  : {0}" -f (Split-Path $net -Leaf))
Write-Host ("  test : {0} nodes/move  |  up to {1} games  |  {2} workers" -f $Nodes, $Games, $Conc)
Write-Host ("  gate : H0 elo<=0  H1 elo>=5   (accept at LLR +/-2.94)")
Write-Host ("  log  : {0}" -f $log)
Write-Host "===================================================================="
Write-Host "  Ordering features need depth -- if borderline, rerun with -Nodes 50000."
Write-Host "--------------------------------------------------------------------"

python (Join-Path $root "tools\nnue\match.py") `
  --engine (Join-Path $bin "conthist.exe") --base-engine (Join-Path $bin "allfeat.exe") `
  --net $net --base-net $net `
  --sprt 0 5 --games $Games --nodes $Nodes `
  --concurrency $Conc --seed 606060 2>&1 | Tee-Object -FilePath $log

$sw.Stop()
Write-Host "--------------------------------------------------------------------"
Write-Host ("  finished in {0:n1} min. Full log: {1}" -f $sw.Elapsed.TotalMinutes, $log)
Write-Host "  Paste the final: line back to me and I'll merge or revert."
Write-Host "===================================================================="
