param(
    [string]$Python = "python",
    [string]$RootPath = "./dataset/weather/",
    [string]$DataPath = "weather.csv",
    [string]$Data = "custom",
    [string]$Model = "TimeMixer_ME",
    [int[]]$PredLens = @(96, 192, 336, 720),
    [int]$SeqLen = 96,
    [int]$LabelLen = 0,
    [int]$EncIn = 21,
    [int]$DecIn = 21,
    [int]$COut = 21,
    [int]$ELayers = 3,
    [int]$DLayers = 1,
    [int]$Factor = 3,
    [int]$DModel = 16,
    [int]$DFF = 32,
    [int]$BatchSize = 128,
    [double]$LearningRate = 0.01,
    [int]$TrainEpochs = 20,
    [int]$Patience = 10,
    [int]$DownSamplingLayers = 3,
    [string]$DownSamplingMethod = "avg",
    [int]$DownSamplingWindow = 2,
    [string]$Features = "M",
    [string]$BaseModelIdPrefix = "weather",
    [string]$BaseComment = "robust_ablation",
    [string]$Des = "RobustAblation",
    [int]$Itr = 1,
    [double]$MissingRate = 0.1,
    [double]$NoiseStd = 0.03,
    [string]$MissingFill = "mean",
    [int]$BlockLen = 12,
    [double]$ChannelDropoutRate = 0.05,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$modes = @(
    @{
        name = "baseline"
        robust = 0
        missing_mode = "random"
        missing_rate = 0.0
        noise_std = 0.0
        block_len = 1
        channel_dropout = 0.0
    },
    @{
        name = "random"
        robust = 1
        missing_mode = "random"
        missing_rate = $MissingRate
        noise_std = $NoiseStd
        block_len = $BlockLen
        channel_dropout = 0.0
    },
    @{
        name = "block"
        robust = 1
        missing_mode = "block"
        missing_rate = $MissingRate
        noise_std = $NoiseStd
        block_len = $BlockLen
        channel_dropout = 0.0
    },
    @{
        name = "mixed"
        robust = 1
        missing_mode = "mixed"
        missing_rate = $MissingRate
        noise_std = $NoiseStd
        block_len = $BlockLen
        channel_dropout = $ChannelDropoutRate
    }
)

foreach ($predLen in $PredLens) {
    foreach ($mode in $modes) {
        $modeName = $mode.name
        $modelId = "{0}_{1}_{2}_{3}" -f $BaseModelIdPrefix, $SeqLen, $predLen, $modeName
        $comment = "{0}_{1}" -f $BaseComment, $modeName

        $args = @(
            "-u", "run.py",
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--root_path", $RootPath,
            "--data_path", $DataPath,
            "--model_id", $modelId,
            "--model", $Model,
            "--data", $Data,
            "--features", $Features,
            "--seq_len", "$SeqLen",
            "--label_len", "$LabelLen",
            "--pred_len", "$predLen",
            "--e_layers", "$ELayers",
            "--d_layers", "$DLayers",
            "--factor", "$Factor",
            "--enc_in", "$EncIn",
            "--dec_in", "$DecIn",
            "--c_out", "$COut",
            "--des", $Des,
            "--comment", $comment,
            "--itr", "$Itr",
            "--d_model", "$DModel",
            "--d_ff", "$DFF",
            "--batch_size", "$BatchSize",
            "--learning_rate", "$LearningRate",
            "--train_epochs", "$TrainEpochs",
            "--patience", "$Patience",
            "--down_sampling_layers", "$DownSamplingLayers",
            "--down_sampling_method", $DownSamplingMethod,
            "--down_sampling_window", "$DownSamplingWindow",
            "--robust_train_aug", "$($mode.robust)",
            "--aug_missing_rate", "$($mode.missing_rate)",
            "--aug_noise_std", "$($mode.noise_std)",
            "--aug_missing_fill", $MissingFill,
            "--aug_missing_mode", "$($mode.missing_mode)",
            "--aug_block_len", "$($mode.block_len)",
            "--aug_channel_dropout_rate", "$($mode.channel_dropout)"
        )

        Write-Host "============================================================"
        Write-Host "PredLen=$predLen | Mode=$modeName | ModelID=$modelId"
        Write-Host "Command: $Python $($args -join ' ')"

        if ($DryRun) {
            continue
        }

        & $Python @args
        if ($LASTEXITCODE -ne 0) {
            throw "Run failed with exit code $LASTEXITCODE for PredLen=$predLen Mode=$modeName"
        }
    }
}

Write-Host "All robustness ablation runs finished."
