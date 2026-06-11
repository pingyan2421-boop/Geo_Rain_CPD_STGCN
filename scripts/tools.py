if __package__:
    from scripts.common.paths import ensure_project_on_path
else:
    from common.bootstrap import ensure_project_on_path

ensure_project_on_path()

from scripts.common.runner import dispatch


COMMANDS = {
    "adjacency-probe": "scripts.adjacency_probe_impl",
    "check-display-metrics": "scripts.check_display_metrics_impl",
    "check-prediction-arrays": "scripts.check_prediction_arrays_impl",
    "get-common-bbox": "scripts.get_common_bbox_impl",
    "prepare-data": "scripts.prepare_data_impl",
    "tester-legacy": "scripts.tester_legacy_impl",
}


if __name__ == "__main__":
    dispatch("Small development and inspection tools.", COMMANDS)
