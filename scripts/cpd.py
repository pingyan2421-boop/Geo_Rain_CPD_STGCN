if __package__:
    from scripts.common.paths import ensure_project_on_path
else:
    from common.bootstrap import ensure_project_on_path

ensure_project_on_path()

from scripts.common.runner import dispatch


COMMANDS = {
    "validate": "scripts.cpd_split_validate_impl",
    "method-sensitivity": "scripts.cpd_method_sensitivity_impl",
    "rolling-param": "scripts.rolling_param_validate_impl",
}


if __name__ == "__main__":
    dispatch("CPD stage validation and sensitivity workflows.", COMMANDS)
