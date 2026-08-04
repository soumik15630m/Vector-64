# Does keeping move-ordering history warm across a game's searches gain Elo?
#
#   powershell -ExecutionPolicy Bypass -File tools\sprt_warmhistory.ps1
#   powershell -ExecutionPolicy Bypass -File tools\sprt_warmhistory.ps1 -Nodes 50000
#
# Background: the search clears its ordering tables at the start of every search.
# Datagen already opts out (set_persist_ordering) because a game is many short
# fixed-node searches and re-zeroing ~857 KB each time is pure overhead there.
# Whether it also *plays* better has never been measured -- this measures it.
#
# Both sides are the SAME binary. Only the PersistOrdering UCI option differs,
# so nothing else can account for the result: no separate build, no other diff.
# ucinewgame still resets between games, so history only ever warms within one
# game, which is the behaviour under test.
#
# Ordering effects need depth to express (a shallow fixed-node tree undervalues
# them, which is what sank earlier ordering tests at 8k). 25k nodes is the floor;
# rerun at 50k if the result lands borderline.
#
# SPRT: H0 elo<=0, H1 elo>=5, alpha=beta=0.05, stops at LLR +/-2.94.

param(
  [int]$Nodes = 25000,
  [int]$Games = 6000,
  [int]$Conc  = 10,
  [string]$Engine = "build-bench\bin\ChessEngine-nnue.exe"
)

$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $PSScriptRoot
$exe    = Join-Path $root $Engine
$logdir = Join-Path $root "sprt\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

if (-not (Test-Path $exe)) {
  Write-Error "engine not found: $exe`nBuild it first: cmake --build build-bench --target ChessEngine-nnue -j"
  exit 1
}

$log = Join-Path $logdir ("warmhistory_{0}n.log" -f $Nodes)
$sw  = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host ""
Write-Host "===================================================================="
Write-Host "  STK-Vector-64  --  warm move-ordering history SPRT"
Write-Host "===================================================================="
Write-Host ("  new  : PersistOrdering=true   (history kept warm within a game)")
Write-Host ("  base : PersistOrdering=false  (current default: cleared per search)")
Write-Host ("  same binary both sides: {0}" -f (Split-Path $exe -Leaf))
Write-Host ("  net  : embedded (stk-vector-64)")
Write-Host ("  test : {0} nodes/move  |  up to {1} games  |  {2} workers" -f $Nodes, $Games, $Conc)
Write-Host ("  gate : H0 elo<=0  H1 elo>=5   (accept at LLR +/-2.94)")
Write-Host ("  log  : {0}" -f $log)
Write-Host "===================================================================="
Write-Host "  Ordering features need depth -- if borderline, rerun with -Nodes 50000."
Write-Host "--------------------------------------------------------------------"

python (Join-Path $root "tools\nnue\match.py") `
  --engine $exe --base-engine $exe `
  --uci-new "PersistOrdering=true" --uci-base "PersistOrdering=false" `
  --sprt 0 5 --games $Games --nodes $Nodes `
  --concurrency $Conc --seed 707070 2>&1 | Tee-Object -FilePath $log

$sw.Stop()
Write-Host "--------------------------------------------------------------------"
Write-Host ("  finished in {0:n1} min. Full log: {1}" -f $sw.Elapsed.TotalMinutes, $log)
Write-Host "  Paste the final: line back and the default gets flipped or dropped."
Write-Host "===================================================================="
