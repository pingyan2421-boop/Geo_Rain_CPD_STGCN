if __package__:
    from scripts.common.paths import ensure_project_on_path
else:
    from common.bootstrap import ensure_project_on_path

ensure_project_on_path()

from scripts.common.runner import dispatch


COMMANDS = {
    "preprocess": "scripts.preprocess_impl",
    "train-fixed": "scripts.train_fixed_impl",
    "test-fixed": "models.tester",
}


if __name__ == "__main__":
    dispatch("Fixed-split CPD-STGCN preprocess, train, and test workflows.", COMMANDS)
