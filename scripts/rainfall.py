if __package__:
    from scripts.common.paths import ensure_project_on_path
else:
    from common.bootstrap import ensure_project_on_path

ensure_project_on_path()

from scripts.common.runner import dispatch


COMMANDS = {
    "build-susceptibility": "scripts.build_rain_susceptibility_impl",
    "chirps-cog-probe": "scripts.chirps_cog_probe_impl",
    "chirps-gefs-forecast": "scripts.chirps_gefs_forecast_impl",
    "chirps-progress": "scripts.chirps_progress_impl",
    "chirps-validate-cache": "scripts.chirps_validate_cache_impl",
    "cpd-response": "scripts.rainfall_cpd_response_impl",
    "forecast-climatology-fallback": "scripts.forecast_climatology_fallback_impl",
    "forecast-diagnostics": "scripts.forecast_diagnostics_impl",
    "forecast-subset-eval": "scripts.forecast_subset_eval_impl",
    "focus-eval": "scripts.evaluate_rainfall_focus_impl",
    "hotspots": "scripts.node_rainfall_hotspots_impl",
    "event-catalog": "scripts.rainfall_event_catalog_impl",
    "mechanism-analysis": "scripts.landslide_mechanism_analysis_impl",
    "mechanism-figure": "scripts.plot_rainfall_mechanism_figure_impl",
    "response-figures": "scripts.plot_chirps_response_figures_impl",
}


if __name__ == "__main__":
    dispatch("Rainfall response and rain-susceptible-node workflows.", COMMANDS)
