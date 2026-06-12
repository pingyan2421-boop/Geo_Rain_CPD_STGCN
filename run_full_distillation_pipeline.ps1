param (
    [string]$Python = "python"
)

$COMMON_ARGS = "--rainfall_features dataset/rainfall/chirps_daily.csv --forecast_features dataset/forecast/chirps_gefs_15day_cp180_climfallback.csv --rain_susceptibility output/rain_susceptibility/rain_susceptibility.csv --dynamic_rain_graph --save_fold_predictions"

$FOLDS = @("cp_160", "cp_180")
$SEEDS = @(0, 7, 23)

Write-Host "=========================================================="
Write-Host "STARTING PHASE 4 OVERNIGHT MULTI-SEED PIPELINE"
Write-Host "=========================================================="

foreach ($seed in $SEEDS) {
    Write-Host "=========================================================="
    Write-Host "PROCESSING SEED $seed"
    Write-Host "=========================================================="
    
    $A_DIR = "output/pipeline_gefs15_nosar_seed${seed}"
    $B_DIR = "output/pipeline_gefs15_seed${seed}"
    $C_DIR = "output/pipeline_distilled_noweight_seed${seed}"
    $D_DIR = "output/pipeline_distilled_nosar_seed${seed}"
    $E_DIR = "output/pipeline_distilled_seed${seed}"
    $FALL_DIR = "output/pipeline_event_seed${seed}"
    $TEACHER_DIR = "output/pipeline_teacher_seed${seed}"
    
    foreach ($fold in $FOLDS) {
        Write-Host "-> [Seed $seed] Training Model A: GEFS15 Baseline (No SAR) on fold $fold..."
        Invoke-Expression "& $PYTHON -m scripts.cpd_split_validate_impl --seed $seed --fold_filter $fold --output_dir $A_DIR $COMMON_ARGS"

        Write-Host "-> [Seed $seed] Training Model B: GEFS15 + SAR on fold $fold..."
        Invoke-Expression "& $PYTHON -m scripts.cpd_split_validate_impl --seed $seed --fold_filter $fold --source_aware_residual --source_delta_scale 0.35 --forecast_source_col forecast_is_fallback --output_dir $B_DIR $COMMON_ARGS"
        
        Write-Host "-> [Seed $seed] Training Fallback Base (Event Context) on fold $fold..."
        Invoke-Expression "& $PYTHON -m scripts.cpd_split_validate_impl --seed $seed --fold_filter $fold --event_context_features output/rainfall_event_catalog/rainfall_event_catalog.csv --event_context_mode input --output_dir $FALL_DIR $COMMON_ARGS"
    }

    Write-Host "-> [Seed $seed] Generating Continuous Gate Teacher Targets..."
    Invoke-Expression "& $PYTHON -m scripts.generate_continuous_gate_teachers --base_dir_official $B_DIR --base_dir_fallback $FALL_DIR --output_dir $TEACHER_DIR --enable_sample_weight --enable_horizon_weight --enable_node_weight"

    foreach ($fold in $FOLDS) {
        $T_DIR = "${TEACHER_DIR}/${fold}/predictions"

        Write-Host "-> [Seed $seed] Training Model C: GEFS15 + Distillation Only (No SAR, No 3D Weight) on fold $fold..."
        $env:IGNORE_DISTILL_WEIGHTS="1"
        Invoke-Expression "& $PYTHON -m scripts.cpd_split_validate_impl --seed $seed --fold_filter $fold --distill_targets_dir $T_DIR --distill_weight 1.0 --distill_horizon_start 3 --output_dir $C_DIR $COMMON_ARGS"
        $env:IGNORE_DISTILL_WEIGHTS="0"

        Write-Host "-> [Seed $seed] Training Model D: GEFS15 + Distillation + 3D Weight (No SAR) on fold $fold..."
        Invoke-Expression "& $PYTHON -m scripts.cpd_split_validate_impl --seed $seed --fold_filter $fold --distill_targets_dir $T_DIR --distill_weight 1.0 --distill_horizon_start 3 --output_dir $D_DIR $COMMON_ARGS"

        Write-Host "-> [Seed $seed] Training Model E: GEFS15 + Distillation + 3D Weight + SAR on fold $fold..."
        Invoke-Expression "& $PYTHON -m scripts.cpd_split_validate_impl --seed $seed --fold_filter $fold --source_aware_residual --source_delta_scale 0.35 --forecast_source_col forecast_is_fallback --distill_targets_dir $T_DIR --distill_weight 1.0 --distill_horizon_start 3 --output_dir $E_DIR $COMMON_ARGS"
    }
}

Write-Host "Pipeline Execution Complete! Now aggregating metrics..."
Invoke-Expression "& $PYTHON aggregate_predictions.py"
Write-Host "All done!"
