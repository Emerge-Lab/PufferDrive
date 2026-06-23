"""Composable YAML config (Hydra-style) for PufferDrive.

A *recipe* is a small .yaml that names which component file to use for each
config group, plus a list of eval suites and any inline overrides:

    # config/ocean/drive/recipes/default.yaml
    package: ocean
    env_name: puffer_drive
    vecenv: default          # -> vecenv/default.yaml      ([vec])
    train: default           # -> train/default.yaml       ([train])
    simulator: gigaflow_carla# -> simulator/gigaflow_carla.yaml ([env])
    model: gigaflow          # -> model/gigaflow.yaml      ([base]/[policy]/[rnn])
    eval:                    # list, each -> eval/<name>.yaml
      - validation
      - behaviors
    mine: {...}              # inline section, merged verbatim

Each component file is a mapping of *internal section name* -> {keys}, e.g.
`simulator/gigaflow_carla.yaml` contains a top-level `env:` block. The internal
section names (`vec`/`env`/`policy`/`rnn`/`train`/`base`) are deliberately kept
so the composed dict matches exactly what `load_config` already returns and
nothing downstream has to change.

Composition order (low -> high precedence):
  1. default.ini base layer
  2. the recipe's chosen component files
  3. inline sections written directly in the recipe
  (CLI flags are layered on top later, by load_config's argparse.)
"""

import ast
import configparser
import glob
import os

import yaml

# Recipe keys that select a single component file from the matching directory.
# group name -> sub-directory under the component root.
COMPONENT_GROUPS = {
    "vecenv": "vecenv",
    "train": "train",
    "simulator": "simulator",
    "model": "model",
}

# Recipe keys consumed by the composer itself (never copied verbatim as
# config sections). Everything else in a recipe is treated as an inline
# section (e.g. `mine`, `controlled_exp` for those subcommands).
_RECIPE_RESERVED = set(COMPONENT_GROUPS) | {"package", "env_name", "eval", "overrides"}


# Lowercase boolean spellings ast.literal_eval can't parse. Centralizing the
# coercion here means native bools reach every consumer (e.g. the evaluators),
# so they never have to defensively re-parse string "true"/"false".
_BOOL_LITERALS = {"true": True, "false": False}


def coerce_value(value):
    """Config value coercion shared by the .ini loader and argparse CLI parsing.

    Literal-eval Python values (ints, floats, lists, True/False/None), accept
    lowercase true/false, else keep the raw string ("auto", "none", paths)."""
    if not isinstance(value, str):
        return value
    if value.strip().lower() in _BOOL_LITERALS:
        return _BOOL_LITERALS[value.strip().lower()]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


# Backwards-compatible internal alias.
_coerce = coerce_value


def _ini_to_nested(ini_path):
    """Read an .ini into a nested dict {section: {key: coerced_value}}.

    Section names are kept as-is (including `base`); dotted section names like
    `eval.foo` stay flat here and are expanded by the caller if needed."""
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    parser.read(ini_path)
    nested = {}
    for section in parser.sections():
        nested[section] = {key: _coerce(val) for key, val in parser[section].items()}
    return nested


def _deep_merge(base, overlay):
    """Recursively merge `overlay` into `base` in place; `overlay` wins."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at top level, got {type(data).__name__}")
    return data


def _load_component(component_root, group_dir, name):
    path = os.path.join(component_root, group_dir, f"{name}.yaml")
    if not os.path.exists(path):
        available = (
            sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(os.path.join(component_root, group_dir))
                if f.endswith(".yaml")
            )
            if os.path.isdir(os.path.join(component_root, group_dir))
            else []
        )
        raise FileNotFoundError(f"Config component '{name}' not found at {path}. Available {group_dir}: {available}")
    return _load_yaml(path)


def is_recipe(path):
    """A recipe is a .yaml/.yml file (vs the legacy .ini config path)."""
    return isinstance(path, str) and path.lower().endswith((".yaml", ".yml"))


def find_recipe(env_name, puffer_dir):
    """Locate the default recipe for `env_name` under config/**/recipes/.

    Returns the path to the recipe whose `env_name` matches, preferring a file
    literally named `default.yaml`. Returns None if no recipe matches (caller
    then falls back to the legacy .ini loader)."""
    pattern = os.path.join(puffer_dir, "config", "**", "recipes", "*.yaml")
    matches = []
    for path in glob.glob(pattern, recursive=True):
        try:
            recipe = _load_yaml(path)
        except (ValueError, yaml.YAMLError):
            continue
        if recipe.get("env_name") == env_name:
            matches.append(path)
    if not matches:
        return None
    for path in matches:
        if os.path.basename(path) == "default.yaml":
            return path
    return matches[0]


def compose_recipe(recipe_path, default_ini_path):
    """Build the nested config dict from a recipe + its component files.

    Output matches load_config's shape: top-level identity keys (package,
    env_name, policy_name, rnn_name) hoisted out of `base`, plus section
    sub-dicts (vec, train, env, policy, rnn, eval, ...)."""
    composed = _ini_to_nested(default_ini_path)

    recipe = _load_yaml(recipe_path)
    # Component dirs live alongside the recipes/ dir: <root>/recipes/foo.yaml
    component_root = os.path.dirname(os.path.dirname(os.path.abspath(recipe_path)))

    # Identity fields live at the recipe top level.
    for key in ("package", "env_name"):
        if key in recipe:
            composed.setdefault("base", {})[key] = recipe[key]

    # Single-choice component groups: each contributes one or more sections.
    for group, group_dir in COMPONENT_GROUPS.items():
        name = recipe.get(group)
        if name is None:
            continue
        if not isinstance(name, str):
            raise ValueError(
                f"Recipe key '{group}' must name a component file (a string), got {type(name).__name__}. "
                f"To tweak individual values, use an `overrides:` block, e.g.\n"
                f"  overrides:\n    {group if group not in ('vecenv', 'simulator', 'model') else 'env'}:\n      key: value"
            )
        _deep_merge(composed, _load_component(component_root, group_dir, name))

    # Eval is a list of suites; each file may declare multiple [eval.<name>]
    # sections (templates + concrete suites). They all land under `eval`.
    composed.setdefault("eval", {})
    for eval_name in recipe.get("eval", []) or []:
        suite = _load_component(component_root, "eval", eval_name)
        for section_name, body in suite.items():
            composed["eval"][section_name] = body

    # Standalone inline sections (mine, controlled_exp, ...) for non-training
    # subcommands. These are whole sections, not patches of a component.
    for key, value in recipe.items():
        if key in _RECIPE_RESERVED:
            continue
        if isinstance(value, dict):
            _deep_merge(composed.setdefault(key, {}), value)
        else:
            composed[key] = value

    # `overrides:` patches any section's values on top of its component — the
    # uniform way to tweak a few knobs without forking a component file. Highest
    # precedence (CLI flags still win over everything, later).
    overrides = recipe.get("overrides") or {}
    for section, patch in overrides.items():
        if isinstance(patch, dict):
            _deep_merge(composed.setdefault(section, {}), patch)
        else:
            composed[section] = patch

    # Hoist `base` keys to the top level (load_config registers them as
    # bare --flags, e.g. --policy-name, not --base.policy-name).
    base_section = composed.pop("base", {})
    for key, value in base_section.items():
        composed[key] = value

    return composed


def _read_env_ini(env_name, puffer_dir, default_ini_path):
    """Legacy loader: find the .ini whose [base].env_name matches `env_name`,
    merged on top of default.ini. Returns a ConfigParser."""
    if env_name == "default":
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        parser.read(default_ini_path)
        return parser
    pattern = os.path.join(puffer_dir, "config", "**", "*.ini")
    for path in glob.glob(pattern, recursive=True):
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        parser.read([default_ini_path, path])
        if parser.has_section("base") and env_name in parser["base"].get("env_name", "").split():
            return parser
    raise ValueError(f"No config (recipe or .ini) for env_name {env_name}")


def flatten_ini(env_name, puffer_dir, default_ini_path):
    """Legacy .ini path -> {dotted_key: coerced_value}, matching the historic
    flag scheme: `base` keys are bare, every other section is `section.key`."""
    parser = _read_env_ini(env_name, puffer_dir, default_ini_path)
    flat = {}
    for section in parser.sections():
        for key in parser[section]:
            dotted = key if section == "base" else f"{section}.{key}"
            flat[dotted] = _coerce(parser[section][key])
    return flat


def flatten_config(nested, prefix=""):
    """Nested config dict -> {dotted_key: leaf_value}. Lists are leaves.

    Mirrors the dotted-section flag scheme load_config builds for argparse:
    `eval.validation.env.simulation_mode` etc."""
    flat = {}
    for key, value in nested.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, dotted))
        else:
            flat[dotted] = value
    return flat
