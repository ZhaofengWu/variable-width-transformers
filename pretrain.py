"""
Pretrain entrypoint with auto-resume: you can always leave load_args in your config
with load_path equal to save_path. On first run (no checkpoint at load_path) we
strip load_args so training starts from scratch; on later runs we keep it and
lm_engine resumes from the latest checkpoint.
"""
import os
import sys

import yaml
from lm_engine.lm_engine.pretrain import main as dolomite_main
from lm_engine.lm_engine.utils.yaml import load_yaml

from custom_yaml_utils import update_custom_yaml

import width_varying_model

# Same filename lm_engine uses to detect a valid checkpoint dir
_LATEST_ITERATION_FILE = "latest_checkpointed_iteration.json"


def _strip_load_args_if_no_checkpoint(config_path: str) -> None:
    """
    If the config has load_args with a load_path but that path has no checkpoint yet
    (first run), remove load_args from the config so lm_engine starts from scratch.
    This allows the same config to be used for both first run and resume.
    """
    config = load_yaml(config_path)
    if config is None:
        return
    load_args = config.get("load_args")
    if not load_args:
        return
    load_path = load_args.get("load_path") if isinstance(load_args, dict) else None
    if not load_path:
        return
    latest_file = os.path.join(load_path, _LATEST_ITERATION_FILE)
    if os.path.isfile(latest_file):
        return
    # No checkpoint yet: strip load_args so we don't try to load
    del config["load_args"]
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    assert len(sys.argv) == 3 and sys.argv[1] == "--config" and sys.argv[2].endswith(".yml")
    resolved_path = update_custom_yaml(sys.argv[2])
    _strip_load_args_if_no_checkpoint(resolved_path)
    sys.argv[2] = resolved_path

    dolomite_main()


if __name__ == "__main__":
    main()
