# Test lazy-eval + dual-net (tiered cheap eval) vs the big-net-only baseline,
# native PowerShell, live + logged. Both engines are allfeat.exe; only the
# UCI eval knobs differ, so this isolates the eval-shortcut Elo cost.
#
#   powershell -ExecutionPolicy Bypass -File tools\sprt_lazydual.ps1
#   powershell -ExecutionPolicy Bypass -File tools\sprt_lazydual.ps1 -LazyMargin 400 -SmallThreshold 250
#
# Tiering (both gate on the O(1) material+psqt estimate):
#   |est| > LazyMargin      -> return material+psqt, skip the NNUE entirely
#   SmallThreshold..Lazy    -> the distilled 128 small net
#   < SmallThreshold        -> full 1024 big net
#
# Fixed nodes measures the Elo COST of the shortcut (it is pure cost at fixed
# nodes, pure speed at fixed time). Gate: H0 elo<=-4 (a real loss), H1 elo>=0
# (neutral/gain). If it accepts H1, the shortcut is free -> keep the NPS at
# real time control.

param(
  [int]$LazyMargin     = 500,
  [int]$SmallThreshold = 300,
  [int]$Nodes          = 8000,
  [int]$Games          = 3000,
  [int]$Conc           = 10
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$big  = Join-Path $root "runs\v2\stk_halfka_1024.nnue"
$sml  = Join-Path $root "runs\small2\stk_halfka_128.nnue"
$bin  = Join-Path $root "sprt"
$eng  = Join-Path $bin "allfeat.exe"
$logdir = Join-Path $bin "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

foreach ($f in @($big, $sml, $eng)) {
  if (-not (Test-Path $f)) { Write-Error "missing: $f"; exit 1 }
}

$log = Join-Path $logdir ("lazydual_L{0}_S{1}_{2}n.log" -f $LazyMargin, $SmallThreshold, $Nodes)
$uci = "LazyEvalMargin={0};SmallNetThreshold={1}" -f $LazyMargin, $SmallThreshold
$sw  = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host ""
Write-Host "===================================================================="
Write-Host "  STK-Vector-64  --  lazy-eval + dual-net eval-shortcut test"
Write-Host "===================================================================="
Write-Host ("  new  : big + small + lazy   ({0})" -f $uci)
Write-Host ("  base : big-net only")
Write-Host ("  test : {0} nodes/move  |  up to {1} games  |  {2} workers" -f $Nodes, $Games, $Conc)
Write-Host ("  gate : H0 elo<=-4  H1 elo>=0   (accept H1 = shortcut is free)")
Write-Host ("  log  : {0}" -f $log)
Write-Host "===================================================================="
Write-Host "  Watch elo (want it near 0) and LLR march to +/-2.94. Ctrl-C stops."
Write-Host "--------------------------------------------------------------------"

python (Join-Path $root "tools\nnue\match.py") `
  --engine $eng --base-engine $eng `
  --net $big --net-small $sml --uci-new $uci `
  --base-net $big `
  --sprt -4 0 --games $Games --nodes $Nodes `
  --concurrency $Conc --seed 515151 2>&1 | Tee-Object -FilePath $log

$sw.Stop()
Write-Host "--------------------------------------------------------------------"
Write-Host ("  done in {0:n1} min. Log: {1}" -f $sw.Elapsed.TotalMinutes, $log)
Write-Host "  Paste the final: line. elo near 0 => shortcut is free NPS at real TC."
Write-Host "===================================================================="
