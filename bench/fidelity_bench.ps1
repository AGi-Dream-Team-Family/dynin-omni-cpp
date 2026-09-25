<#
.SYNOPSIS
  Exact-vs-windowed (block-local KV) fidelity bench for dynin-t2s-kvblock.

.DESCRIPTION
  Runs every sentence in -Sentences through dynin-t2s-kvblock.exe --compare at
  three schedule arms (block 128 / steps 48, block 128 / steps 96, block 16 /
  steps 48, all at kv-margin 32, cfg 2.5, a fixed seed), each arm producing an
  "exact" (kv-margin 0) and a "margin" (windowed) run on the same seed. Each
  run writes match.json (unit-level exact-match fraction between the two
  decodes) and timing.json (wall-clock ms, forward-pass counts) via the
  tool's own --compare mode, plus a whisper transcript of each wav when
  -WhisperCli/-WhisperModel are given. Reads those back, takes each run's
  word overlap from the value the tool prints on its [DYNIN-T2S-CHECK] line,
  and writes one summary CSV plus a raw per-run JSON log.

  No absolute path is hard-coded here -- every model/tool path is a required
  parameter.

.PARAMETER Exe
  Path to dynin-t2s-kvblock.exe.

.PARAMETER Model
  Path to the Dynin-Omni diffusion checkpoint GGUF.

.PARAMETER U2s
  Path to the EMOVA U2S vocoder GGUF.

.PARAMETER WhisperCli
  Path to whisper-cli.exe. Omit (with -WhisperModel also omitted) to skip
  transcription -- fidelity (unit-match, timing) is still measured.

.PARAMETER WhisperModel
  Path to a ggml whisper model .bin.

.PARAMETER Sentences
  Path to a text file, one sentence per line (see sentences.txt next to this
  script for the ten sentences used in receipts/fidelity_2026-09-22.md).

.PARAMETER OutDir
  Directory to write per-run outputs (wavs, match.json, timing.json, logs)
  and the summary CSV into. Created if missing.

.PARAMETER Seed
  RNG seed shared by every arm and every sentence (default 42, the seed used
  in receipts/fidelity_2026-09-22.md).

.EXAMPLE
  pwsh bench/fidelity_bench.ps1 `
    -Exe ./build/bin/dynin-t2s-kvblock.exe `
    -Model ./models/dynin-omni.gguf `
    -U2s ./models/emova_u2s.gguf `
    -WhisperCli ./whisper/bin/whisper-cli.exe `
    -WhisperModel ./whisper/ggml-base.en.bin `
    -Sentences bench/sentences.txt `
    -OutDir out/fidelity_bench
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Exe,
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $U2s,
    [string] $WhisperCli = "",
    [string] $WhisperModel = "",
    [Parameter(Mandatory = $true)] [string] $Sentences,
    [Parameter(Mandatory = $true)] [string] $OutDir,
    [int] $Seed = 42
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Exe))       { throw "Exe not found: $Exe" }
if (-not (Test-Path $Model))     { throw "Model not found: $Model" }
if (-not (Test-Path $U2s))       { throw "U2S gguf not found: $U2s" }
if (-not (Test-Path $Sentences)) { throw "Sentences file not found: $Sentences" }

$haveWhisper = ($WhisperCli -ne "") -and ($WhisperModel -ne "")
if ($haveWhisper) {
    if (-not (Test-Path $WhisperCli))   { throw "whisper-cli not found: $WhisperCli" }
    if (-not (Test-Path $WhisperModel)) { throw "whisper model not found: $WhisperModel" }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$sentenceList = Get-Content -Path $Sentences | Where-Object { $_.Trim() -ne "" }

# arm name -> (block, steps, kv-margin, cfg)
$arms = [ordered]@{
    "block128_steps48_margin32" = @{ block = 128; steps = 48; margin = 32; cfg = 2.5 }
    "block128_steps96_margin32" = @{ block = 128; steps = 96; margin = 32; cfg = 2.5 }
    "block16_steps48_margin32"  = @{ block = 16;  steps = 48; margin = 32; cfg = 2.5 }
}

function Get-WordOverlap {
    param([string] $Reference, [string] $Hyp)
    $refWords = @(($Reference.ToLowerInvariant() -replace '[^a-z0-9]+', ' ').Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) | Select-Object -Unique)
    if ($refWords.Count -eq 0) { return 0.0 }
    $hypWords = @(($Hyp.ToLowerInvariant() -replace '[^a-z0-9]+', ' ').Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries))
    $hypSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$hypWords)
    $overlap = 0
    foreach ($w in $refWords) { if ($hypSet.Contains($w)) { $overlap++ } }
    return [double]$overlap / [double]$refWords.Count
}

$allRuns = New-Object System.Collections.Generic.List[object]

$sIdx = 0
foreach ($sentence in $sentenceList) {
    $sIdx++
    foreach ($armName in $arms.Keys) {
        $a = $arms[$armName]
        $runDir = Join-Path $OutDir "$armName/s$sIdx"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $logPath = Join-Path $runDir "run.log"

        $cliArgs = @(
            "-m", $Model,
            "--u2s", $U2s,
            "--text", $sentence,
            "--steps", $a.steps,
            "--block", $a.block,
            "--kv-margin", $a.margin,
            "--cfg", $a.cfg,
            "--seed", $Seed,
            "--out", $runDir,
            "--wav", "--compare"
        )
        if ($haveWhisper) {
            $cliArgs += @("--whisper-cli", $WhisperCli, "--whisper-model", $WhisperModel)
        }

        Write-Host "[$armName][s$sIdx] $sentence"
        $stdout = & $Exe @cliArgs 2>&1
        $exitCode = $LASTEXITCODE
        $stdout | Out-File -FilePath $logPath -Encoding utf8

        $result = [ordered]@{
            arm            = $armName
            sentence_index = $sIdx
            sentence       = $sentence
            exit_code      = $exitCode
            block          = $a.block
            steps          = $a.steps
            kv_margin      = $a.margin
            cfg            = $a.cfg
            seed           = $Seed
        }

        if ($exitCode -ne 0) {
            $result.failed = $true
            $result.failure_line = ($stdout | Where-Object { $_ -match "DYNIN-T2S-ERROR|error:" } | Select-Object -First 1)
            $allRuns.Add([pscustomobject]$result)
            Write-Warning "  FAILED: $($result.failure_line)"
            continue
        }
        $result.failed = $false

        $matchPath  = Join-Path $runDir "match.json"
        $timingPath = Join-Path $runDir "timing.json"
        if (Test-Path $matchPath)  { $match  = Get-Content $matchPath  -Raw | ConvertFrom-Json } else { $match  = $null }
        if (Test-Path $timingPath) { $timing = Get-Content $timingPath -Raw | ConvertFrom-Json } else { $timing = $null }

        $result.unit_match_fraction   = if ($match)  { $match.exact_match_fraction } else { $null }
        $result.first_divergence      = if ($match)  { $match.first_divergence_index } else { $null }
        $result.exact_ms              = if ($timing) { $timing.exact.total_ms }  else { $null }
        $result.margin_ms             = if ($timing) { $timing.margin.total_ms } else { $null }
        $result.exact_forward_passes  = if ($timing) { $timing.exact.n_forward_passes }  else { $null }
        $result.margin_forward_passes = if ($timing) { $timing.margin.n_forward_passes } else { $null }

        $exactLine  = $stdout | Where-Object { $_ -match "DYNIN-T2S-CHECK run=exact" }  | Select-Object -First 1
        $marginLine = $stdout | Where-Object { $_ -match "DYNIN-T2S-CHECK run=margin" } | Select-Object -First 1

        # The tool prints the transcript and its own word-overlap on one line:
        #   [DYNIN-T2S-CHECK run=exact  text='...' whisper='...' word_overlap=0.8750]
        # Transcripts can contain apostrophes, so the transcript is everything
        # between whisper=' and the closing ' that precedes word_overlap=, and the
        # overlap is the value the tool printed (Get-WordOverlap is kept only as a
        # fallback for a line without the word_overlap field).
        function Get-CheckFields($line) {
            if ($line -and ($line -match "whisper='(.*)' word_overlap=([0-9.]+)\]")) {
                return @{ transcript = $Matches[1]; overlap = [double]$Matches[2] }
            }
            if ($line -and ($line -match "whisper='(.*)'")) {
                return @{ transcript = $Matches[1]; overlap = $null }
            }
            return $null
        }

        $exactFields  = Get-CheckFields $exactLine
        $marginFields = Get-CheckFields $marginLine
        $exactTranscript  = if ($exactFields)  { $exactFields.transcript }  else { $null }
        $marginTranscript = if ($marginFields) { $marginFields.transcript } else { $null }

        $result.exact_transcript  = $exactTranscript
        $result.margin_transcript = $marginTranscript
        $result.exact_word_overlap  = if ($exactFields -and $null -ne $exactFields.overlap)   { $exactFields.overlap }  elseif ($exactTranscript)  { Get-WordOverlap -Reference $sentence -Hyp $exactTranscript }  else { $null }
        $result.margin_word_overlap = if ($marginFields -and $null -ne $marginFields.overlap) { $marginFields.overlap } elseif ($marginTranscript) { Get-WordOverlap -Reference $sentence -Hyp $marginTranscript } else { $null }

        $allRuns.Add([pscustomobject]$result)
    }
}

$rawJsonPath = Join-Path $OutDir "raw_runs.json"
$allRuns | ConvertTo-Json -Depth 6 | Out-File -FilePath $rawJsonPath -Encoding utf8

# ---- per-arm summary -------------------------------------------------------
$summaryRows = New-Object System.Collections.Generic.List[object]
foreach ($armName in $arms.Keys) {
    $rows = $allRuns | Where-Object { $_.arm -eq $armName -and -not $_.failed }
    $n = @($rows).Count
    if ($n -eq 0) {
        $summaryRows.Add([pscustomobject]@{
            arm = $armName; n_ok = 0; mean_exact_ms = $null; mean_margin_ms = $null
            speedup = $null; mean_unit_match = $null
            mean_word_overlap_exact = $null; mean_word_overlap_margin = $null
        })
        continue
    }
    $meanExactMs  = ($rows | Measure-Object -Property exact_ms  -Average).Average
    $meanMarginMs = ($rows | Measure-Object -Property margin_ms -Average).Average
    $meanMatch    = ($rows | Measure-Object -Property unit_match_fraction -Average).Average
    $owExact  = $rows | Where-Object { $null -ne $_.exact_word_overlap }
    $owMargin = $rows | Where-Object { $null -ne $_.margin_word_overlap }
    $meanOwExact  = if (@($owExact).Count  -gt 0) { ($owExact  | Measure-Object -Property exact_word_overlap  -Average).Average } else { $null }
    $meanOwMargin = if (@($owMargin).Count -gt 0) { ($owMargin | Measure-Object -Property margin_word_overlap -Average).Average } else { $null }

    $summaryRows.Add([pscustomobject]@{
        arm                      = $armName
        n_ok                     = $n
        mean_exact_ms            = [math]::Round($meanExactMs, 1)
        mean_margin_ms           = [math]::Round($meanMarginMs, 1)
        speedup                  = if ($meanMarginMs -gt 0) { [math]::Round($meanExactMs / $meanMarginMs, 3) } else { $null }
        mean_unit_match          = [math]::Round($meanMatch, 4)
        mean_word_overlap_exact  = if ($null -ne $meanOwExact)  { [math]::Round($meanOwExact, 4) }  else { $null }
        mean_word_overlap_margin = if ($null -ne $meanOwMargin) { [math]::Round($meanOwMargin, 4) } else { $null }
    })
}

$summaryCsvPath = Join-Path $OutDir "summary.csv"
$summaryRows | Export-Csv -Path $summaryCsvPath -NoTypeInformation

Write-Host ""
Write-Host "=== SUMMARY ==="
$summaryRows | Format-Table -AutoSize

Write-Host ""
Write-Host "Raw per-run data: $rawJsonPath"
Write-Host "Summary CSV:      $summaryCsvPath"
