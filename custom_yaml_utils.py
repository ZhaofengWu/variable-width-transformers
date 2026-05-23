import datetime
import os

from lm_engine.lm_engine.utils.yaml import load_yaml


def update_custom_yaml(input_path: str) -> str:
    assert input_path.endswith(".yml")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"

    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)

    generated_dir = os.path.join(input_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    output_filename = input_filename.replace(".yml", f"_{timestamp}.yml")
    output_path = os.path.join(generated_dir, output_filename)

    assert not os.path.exists(output_path)
    parse_custom_yaml(input_path, output_path)

    return output_path


def parse_custom_yaml(input_path: str, output_path: str):
    """Replace all occurrences of keys in global_args with their values in the yaml file."""

    yaml_dict = load_yaml(input_path)

    # Read the original file as a string
    with open(input_path, "r") as f:
        yaml_str = f.read()

    if "global_args" in yaml_dict:
        global_args = yaml_dict["global_args"]

        # Find and remove the global_args section
        assert yaml_str.startswith("global_args:")
        # Find the first empty line after global_args section
        next_section_start = yaml_str.find("\n\n")
        assert next_section_start != -1
        # Remove the global_args section including the empty line after it
        yaml_str = yaml_str[next_section_start + 2 :]

        # Replace all occurrences of ${key} with value
        for key, value in global_args.items():
            placeholder = "${" + key + "}"
            yaml_str = yaml_str.replace(placeholder, str(value))

    # Write the processed string to output file
    with open(output_path, "w") as f:
        f.write(yaml_str)