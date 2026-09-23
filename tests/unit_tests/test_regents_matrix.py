from mlops import regents_matrix
from pufferlib.pufferl import _parse_regents_cli_args


def test_matrix_output_root_is_forwarded_as_a_generation_output_directory(tmp_path):
    config_path = tmp_path / "regents.yaml"
    config_path.touch()
    arguments = regents_matrix._parse_arguments(
        (
            "local",
            "--config-path",
            str(config_path),
            "--experiment-name",
            "seed_4_regents_n1",
            "--output-root",
            "experiments/regents_2",
        )
    )
    generation = regents_matrix.GenerationMetadata(
        ego_label="rl_nocond",
        generation_name="regents_rl_nocond",
        scenario_count=100,
        map_dir="maps",
        artifact_paths=(),
    )

    command = regents_matrix._local_command(generation, arguments)

    output_directory_idx = command.index("--output-dir") + 1
    assert command[output_directory_idx] == "experiments/regents_2/regents_rl_nocond"


def test_regents_cli_accepts_an_explicit_output_directory():
    arguments = _parse_regents_cli_args(
        ("regents_rl_nocond", "--output-dir", "experiments/regents_2/regents_rl_nocond")
    )

    assert arguments.output_dir == "experiments/regents_2/regents_rl_nocond"
