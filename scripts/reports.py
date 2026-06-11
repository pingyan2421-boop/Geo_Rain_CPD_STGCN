if __package__:
    from scripts.common.paths import ensure_project_on_path
else:
    from common.bootstrap import ensure_project_on_path

ensure_project_on_path()

from scripts.common.runner import dispatch


COMMANDS = {
    "evaluator": "scripts.evaluator_impl",
    "future-directions": "scripts.make_future_directions_impl",
    "horizon-error-figures": "scripts.make_horizon_error_figures_impl",
    "paper-comparison": "scripts.make_paper_comparison_table_impl",
    "plotter": "scripts.plotter_impl",
    "prediction-arrays-csv": "scripts.fig_impl",
    "prediction-figures": "scripts.make_prediction_figures_impl",
    "regional-roadmap": "scripts.make_regional_roadmap_impl",
    "report-figures": "scripts.make_report_figures_impl",
    "workflow-figure": "scripts.make_workflow_figure_impl",
}


if __name__ == "__main__":
    dispatch("Report, paper table, and figure generation workflows.", COMMANDS)
