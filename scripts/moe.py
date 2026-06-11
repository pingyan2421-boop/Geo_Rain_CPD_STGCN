if __package__:
    from scripts.common.paths import ensure_project_on_path
else:
    from common.bootstrap import ensure_project_on_path

ensure_project_on_path()

from scripts.common.runner import dispatch


COMMANDS = {
    "anchors": "scripts.summarize_mechanism_anchors_impl",
    "analog-correct": "scripts.analog_residual_correction_impl",
    "analog-correct-summary": "scripts.analog_residual_correction_summary_impl",
    "causal-lag": "scripts.rainfall_causal_lag_analysis_impl",
    "distill-targets": "scripts.source_switch_distillation_impl",
    "distill-summary": "scripts.source_switch_distillation_summary_impl",
    "teacher-grid": "scripts.distillation_teacher_grid_impl",
    "stack-experts": "scripts.expert_stacking_impl",
    "multiscale": "scripts.multiscale_residual_features_impl",
    "optimization-audit": "scripts.optimization_audit_impl",
    "posthoc": "scripts.posthoc_mechanism_moe_impl",
    "project-displacement": "scripts.displacement_projection_impl",
    "quantile-map": "scripts.quantile_mapping_calibration_impl",
    "source-calibrate": "scripts.source_residual_calibration_impl",
    "event-source-switch": "scripts.event_source_switch_posthoc_moe_impl",
    "event-source-switch-grid": "scripts.event_source_switch_grid_impl",
    "event-source-switch-summary": "scripts.event_source_switch_summary_impl",
    "strict-validate": "scripts.strict_validation_impl",
    "source-switch": "scripts.source_switch_posthoc_moe_impl",
    "source-switch-grid": "scripts.source_switch_grid_impl",
    "summarize": "scripts.summarize_posthoc_moe_grid_impl",
    "train-calibrator": "scripts.train_event_calibrator_impl",
}


if __name__ == "__main__":
    dispatch("Mechanism MoE feature, fusion, and audit workflows.", COMMANDS)
