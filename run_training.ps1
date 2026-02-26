# ==========================================
# CONFIGURATION
# ==========================================
$epochs = 14
$checkpoint = "artifacts/vector64_80m_final.pt"
$stateFile = "artifacts/training_state.txt"

# 8 slices of 20 shards (Low RAM mode ~3GB per load)
$slices = @(
    "0:20", "20:40",
    "40:60", "60:80",
    "80:100", "100:120",
    "120:140"
)

# Ensure artifacts folder exists
New-Item -ItemType Directory -Force -Path artifacts | Out-Null

# ==========================================
# RESUME LOGIC (The "Brain")
# ==========================================
$startEpoch = 1
$startSliceIdx = 0
$resume_flag = ""

# Check if we have a checkpoint AND a state file to tell us where we are
if ((Test-Path $checkpoint) -and (Test-Path $stateFile)) {
    Write-Host "Found previous training state..." -ForegroundColor Yellow

    # Read the file. Format is expected to be "EPOCH,SLICE_INDEX"
    $savedState = Get-Content $stateFile
    $parts = $savedState -split ","

    if ($parts.Length -eq 2) {
        $startEpoch = [int]$parts[0]
        $startSliceIdx = [int]$parts[1]
        $resume_flag = "--resume $checkpoint"

        Write-Host "Resuming from EPOCH $startEpoch, SLICE INDEX $startSliceIdx (" $slices[$startSliceIdx] ")" -ForegroundColor Yellow
    }
} elseif (Test-Path $checkpoint) {
    # Checkpoint exists but no state file (rare, but maybe manual delete?)
    Write-Host "Found checkpoint but no state file. Resuming weights, but starting Epoch logic from 1." -ForegroundColor Magenta
    $resume_flag = "--resume $checkpoint"
}

# ==========================================
# THE GRAND LOOP (FIXED)
# ==========================================
for ($e = 1; $e -le $epochs; $e++) {

    # SKIP LOGIC
    if ($e -lt $startEpoch) { continue }

    Write-Host "========================================" -ForegroundColor Green
    Write-Host "STARTING EPOCH $e / $epochs" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    for ($i = 0; $i -lt $slices.Count; $i++) {

        # SKIP LOGIC
        if ($e -eq $startEpoch -and $i -lt $startSliceIdx) {
            Write-Host "  [Skipping completed slice " $slices[$i] "]" -ForegroundColor Gray
            continue
        }

        $slice = $slices[$i]
        Write-Host "  > Training on shards $slice..." -ForegroundColor Cyan

        # --- NEW: Build arguments as a list to prevent "Unrecognized Argument" errors ---
        $cmdArgs = @(
            "tools/nnue/train_vector64.py",
            "--dataset", "my_dataset_strict_80m/manifest.json",
            "--out-checkpoint", $checkpoint,
            "--epochs", "1",
            "--batch-size", "4096",
            "--lr", "1e-6",
            "--num-workers", "0",
            "--shard-slice", $slice
        )

        # Add resume flag safely if needed
        if ($resume_flag -ne "") {
            $cmdArgs += "--resume"
            $cmdArgs += $checkpoint
        }
        # -----------------------------------------------------------------------------

        # Run Python using the clean list of arguments
        python @cmdArgs

        # CRASH DETECTION
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CRITICAL ERROR: Training crashed on Epoch $e, Slice $slice" -ForegroundColor Red
            Write-Host "Fix the error and re-run this script. It will resume right here." -ForegroundColor Red
            exit
        }

        # SAVE STATE
        $nextSliceIdx = $i + 1
        $nextEpoch = $e
        if ($nextSliceIdx -ge $slices.Count) {
            $nextSliceIdx = 0
            $nextEpoch = $e + 1
        }
        "$nextEpoch,$nextSliceIdx" | Set-Content $stateFile

        Write-Host "  [Cooling down GPU for 60 seconds...]" -ForegroundColor Gray
        Start-Sleep -Seconds 60

        # Set resume flag for the next loop
        $resume_flag = "yes"
    }

    $startSliceIdx = 0
}

Write-Host "Training Complete!" -ForegroundColor Green

Write-Host "Training Complete!" -ForegroundColor Green