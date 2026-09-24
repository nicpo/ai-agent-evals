"""Load and validate repository-local YAML experiment profiles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import config
from agent.tools import AVAILABLE_TOOLS

_JUDGE_PROMPTS = frozenset({
    "branch_a_false_positive", "branch_b_false_negative", "faithfulness",
    "uncertainty", "error_recovery", "question_alignment",
})


@dataclass(frozen=True)
class JudgePromptBundle:
    path: Path
    templates: dict[str, str]
    sha256: str


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    agent_model: str
    agent_prompt_path: Path
    agent_prompt: str
    agent_prompt_sha256: str
    enabled_tools: tuple[str, ...]
    judge_enabled: bool
    judge_model: str
    judge_prompts: JudgePromptBundle
    eval_glob: str
    max_tool_calls: int

    def metadata(self) -> dict[str, Any]:
        return {
            "profile": self.name,
            "agent_model": self.agent_model,
            "judge_model": self.judge_model if self.judge_enabled else None,
            "agent_prompt": str(self.agent_prompt_path.relative_to(config.REPO_ROOT)),
            "agent_prompt_sha256": self.agent_prompt_sha256,
            "judge_prompt_bundle": str(self.judge_prompts.path.relative_to(config.REPO_ROOT)),
            "judge_prompt_bundle_sha256": self.judge_prompts.sha256,
            "enabled_tools": list(self.enabled_tools),
            "eval_glob": self.eval_glob,
            "max_tool_calls": self.max_tool_calls,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read YAML file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping.")
    return data


def _repo_path(raw: object, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty repository-relative path.")
    root = config.REPO_ROOT.resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"{field} must remain within the repository: {raw!r}") from None
    if not path.is_file():
        raise ValueError(f"{field} does not name a readable file: {raw!r}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _model(value: object, field: str) -> str:
    if not isinstance(value, str) or value not in config.MODELS:
        raise ValueError(f"{field} must name a configured model; available: {', '.join(config.MODELS)}")
    return value


def _load_judge_prompts(path: Path) -> JudgePromptBundle:
    data = _load_yaml(path)
    templates = _mapping(data.get("prompts"), "judge prompt bundle prompts")
    missing = _JUDGE_PROMPTS - set(templates)
    extra = set(templates) - _JUDGE_PROMPTS
    if missing or extra or any(not isinstance(v, str) or not v.strip() for v in templates.values()):
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown {', '.join(sorted(extra))}")
        if any(not isinstance(v, str) or not v.strip() for v in templates.values()):
            details.append("all templates must be non-empty strings")
        raise ValueError("Malformed judge prompt bundle: " + "; ".join(details))
    return JudgePromptBundle(path=path, templates=dict(templates), sha256=_sha256(path))


def available_profiles(catalog_path: Path = config.PROFILE_CATALOG) -> tuple[str, ...]:
    catalog = _load_yaml(catalog_path)
    profiles = _mapping(catalog.get("profiles"), "profiles")
    return tuple(profiles)


def default_profile_name(catalog_path: Path = config.PROFILE_CATALOG) -> str:
    catalog = _load_yaml(catalog_path)
    default = catalog.get("default_profile")
    if not isinstance(default, str) or default not in _mapping(catalog.get("profiles"), "profiles"):
        raise ValueError("default_profile must name a profile in the catalog.")
    return default


def load_profile(name: str | None = None, catalog_path: Path = config.PROFILE_CATALOG) -> ExperimentProfile:
    catalog = _load_yaml(catalog_path)
    profiles = _mapping(catalog.get("profiles"), "profiles")
    name = name or default_profile_name(catalog_path)
    raw = profiles.get(name)
    if raw is None:
        raise ValueError(f"Unknown profile {name!r}. Available: {', '.join(profiles)}")
    profile = _mapping(raw, f"profiles.{name}")
    agent = _mapping(profile.get("agent"), f"profiles.{name}.agent")
    judge = _mapping(profile.get("judge"), f"profiles.{name}.judge")
    harness = _mapping(profile.get("harness"), f"profiles.{name}.harness")

    tools = agent.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise ValueError(f"profiles.{name}.agent.tools must be a list of tool names.")
    if len(tools) != len(set(tools)):
        raise ValueError(f"profiles.{name}.agent.tools contains duplicates.")
    unknown_tools = set(tools) - AVAILABLE_TOOLS
    if unknown_tools:
        raise ValueError(f"profiles.{name}.agent.tools has unknown tools: {', '.join(sorted(unknown_tools))}")
    if not tools:
        raise ValueError(f"profiles.{name}.agent.tools must enable at least one tool.")

    judge_enabled = judge.get("enabled")
    if not isinstance(judge_enabled, bool):
        raise ValueError(f"profiles.{name}.judge.enabled must be true or false.")
    eval_glob = harness.get("eval_glob")
    max_tool_calls = harness.get("max_tool_calls")
    if not isinstance(eval_glob, str) or not eval_glob:
        raise ValueError(f"profiles.{name}.harness.eval_glob must be a non-empty string.")
    if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise ValueError(f"profiles.{name}.harness.max_tool_calls must be a positive integer.")

    agent_prompt_path = _repo_path(agent.get("prompt"), f"profiles.{name}.agent.prompt")
    judge_prompt_path = _repo_path(judge.get("prompts"), f"profiles.{name}.judge.prompts")
    return ExperimentProfile(
        name=name,
        agent_model=_model(agent.get("model"), f"profiles.{name}.agent.model"),
        agent_prompt_path=agent_prompt_path,
        agent_prompt=agent_prompt_path.read_text(encoding="utf-8"),
        agent_prompt_sha256=_sha256(agent_prompt_path),
        enabled_tools=tuple(tools),
        judge_enabled=judge_enabled,
        judge_model=_model(judge.get("model"), f"profiles.{name}.judge.model"),
        judge_prompts=_load_judge_prompts(judge_prompt_path),
        eval_glob=eval_glob,
        max_tool_calls=max_tool_calls,
    )
