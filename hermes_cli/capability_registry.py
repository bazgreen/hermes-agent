"""Fleet capability registry and routing helpers.

This module loads a small YAML/JSON registry that maps profiles to declared
capabilities. It is used by profile routing and the handoff policy layer to
resolve direct profile names, capability aliases, and per-profile capability
metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from hermes_cli import profiles as profiles_mod

try:
    from hermes_cli.config import load_config
except Exception:  # pragma: no cover - import-time fallback for minimal envs
    load_config = None  # type: ignore[assignment]

CAPABILITY_REGISTRY_ENV = "HERMES_CAPABILITY_REGISTRY"
DEFAULT_REGISTRY_CANDIDATES = (
    Path("/opt/hermes/repo/shared/capabilities.yaml"),
    Path("/opt/hermes/repo/shared/capabilities.json"),
    Path("/opt/hermes/repo/bots/shared/capabilities.yaml"),
    Path("/opt/hermes/repo/bots/shared/capabilities.json"),
    Path.home() / ".hermes" / "capabilities.yaml",
    Path.home() / ".hermes" / "capabilities.json",
)


class CapabilityRegistryError(RuntimeError):
    """Base class for registry load / resolve errors."""


class CapabilityRegistryNotFoundError(CapabilityRegistryError):
    """Raised when no registry file could be located."""


class CapabilityLookupError(CapabilityRegistryError):
    """Raised when a capability cannot be resolved."""


class AmbiguousCapabilityError(CapabilityLookupError):
    """Raised when multiple capabilities match a lookup target."""


@dataclass(frozen=True)
class CapabilityRecord:
    name: str
    profile: str
    description: str = ""
    risk: str = "unknown"
    aliases: tuple[str, ...] = field(default_factory=tuple)
    tools: tuple[str, ...] = field(default_factory=tuple)
    routing: Mapping[str, Any] = field(default_factory=dict)
    summon: bool = False
    source_path: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.profile}:{self.name}"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        data["tools"] = list(self.tools)
        data["routing"] = dict(self.routing)
        return data


@dataclass(frozen=True)
class CapabilityResolution:
    target: str
    profile: str
    capability: Optional[str]
    description: Optional[str]
    risk: str
    tools: tuple[str, ...]
    routing: Mapping[str, Any]
    source_path: str
    direct_profile: bool
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tools"] = list(self.tools)
        data["routing"] = dict(self.routing)
        return data


@dataclass(frozen=True)
class CapabilityRegistry:
    version: int
    source_path: Path
    profiles: tuple[str, ...]
    profile_descriptions: Mapping[str, str]
    capabilities_by_profile: Mapping[str, tuple[CapabilityRecord, ...]]
    by_alias: Mapping[str, tuple[CapabilityRecord, ...]]

    def capabilities_for_profile(self, profile: str) -> tuple[CapabilityRecord, ...]:
        return self.capabilities_by_profile.get(_normalize_text(profile, field_name="profile"), ())

    def as_dict(self) -> dict[str, Any]:
        return registry_as_dict(self)


def _normalize_text(value: Any, *, field_name: str, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise CapabilityRegistryError(f"Missing required field {field_name!r}")
    return text.lower()


def _normalize_str_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        raise CapabilityRegistryError(f"Expected a list of strings, got {type(value).__name__}")
    cleaned: list[str] = []
    for item in items:
        text = str(item or "").strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def _read_registry_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CapabilityRegistryNotFoundError(f"Capability registry file not found: {path}")
    if path.is_dir():
        # A directory is a malformed registry location, not a registry. Fail
        # closed with a registry error (never propagate IsADirectoryError so
        # callers can rely on CapabilityRegistryError for policy decisions).
        raise CapabilityRegistryError(f"Capability registry path is a directory, not a file: {path}")
    raw: Any
    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CapabilityRegistryError(f"Capability registry file is malformed: {path}: {exc}") from exc
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - yaml is expected in this repo
            raise CapabilityRegistryError(f"PyYAML is required to read {path}: {exc}") from exc
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise CapabilityRegistryError(f"Capability registry file is malformed: {path}: {exc}") from exc
    loaded = raw or {}
    if not isinstance(loaded, Mapping):
        raise CapabilityRegistryError(
            f"Capability registry file must contain a mapping at the top level, got {type(loaded).__name__}: {path}"
        )
    return dict(loaded)


def _candidate_paths(explicit: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    env_raw = os.environ.get(CAPABILITY_REGISTRY_ENV, "").strip()
    if env_raw:
        env_path = Path(env_raw).expanduser()
        if env_path not in candidates:
            candidates.append(env_path)

    if load_config is not None:
        try:
            cfg = load_config() or {}
            for key in ("capability_registry", "capabilities", "routing"):
                block = cfg.get(key)
                if isinstance(block, Mapping):
                    for nested_key in ("path", "file", "registry"):
                        value = block.get(nested_key)
                        if value:
                            candidate = Path(str(value)).expanduser()
                            if candidate not in candidates:
                                candidates.append(candidate)
                            break
        except Exception:
            pass

    for candidate in DEFAULT_REGISTRY_CANDIDATES:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _resolve_registry_path(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.exists():
            return candidate
        raise CapabilityRegistryNotFoundError(f"Capability registry file not found: {candidate}")

    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate
    raise CapabilityRegistryNotFoundError(
        "No capability registry found. Looked in: "
        + ", ".join(str(candidate) for candidate in _candidate_paths())
    )


def _build_registry(data: Mapping[str, Any], source_path: Path) -> CapabilityRegistry:
    version = int(data.get("version", 1) or 1)
    profiles = data.get("profiles")
    if not isinstance(profiles, Mapping):
        raise CapabilityRegistryError("Capability registry must contain a 'profiles' mapping")

    profile_names: list[str] = []
    profile_descriptions: dict[str, str] = {}
    capabilities_by_profile: dict[str, tuple[CapabilityRecord, ...]] = {}
    alias_index: dict[str, list[CapabilityRecord]] = {}

    for raw_profile, raw_entry in profiles.items():
        profile = _normalize_text(raw_profile, field_name="profile")
        if profile not in profile_names:
            profile_names.append(profile)

        entry = raw_entry or {}
        if not isinstance(entry, Mapping):
            raise CapabilityRegistryError(f"Profile {raw_profile!r} must map to an object")

        description = str(entry.get("description") or "").strip()
        profile_descriptions[profile] = description

        records: list[CapabilityRecord] = []
        capabilities = entry.get("capabilities") or ()
        if not isinstance(capabilities, Iterable) or isinstance(capabilities, (str, bytes, bytearray)):
            raise CapabilityRegistryError(f"Profile {raw_profile!r} capabilities must be a list")

        for raw_capability in capabilities:
            if not isinstance(raw_capability, Mapping):
                raise CapabilityRegistryError(
                    f"Capability entry for profile {raw_profile!r} must be an object"
                )
            name = _normalize_text(raw_capability.get("name"), field_name="name")
            aliases = _normalize_str_list(raw_capability.get("aliases"))
            tools = _normalize_str_list(raw_capability.get("tools"))
            routing = raw_capability.get("routing") or {}
            if not isinstance(routing, Mapping):
                raise CapabilityRegistryError(
                    f"Capability {raw_profile!r}.{name!r} routing must be a mapping"
                )
            risk = _normalize_text(raw_capability.get("risk", "unknown"), field_name="risk", allow_empty=True) or "unknown"
            summon = bool(raw_capability.get("summon", False))
            record = CapabilityRecord(
                name=name,
                profile=profile,
                description=str(raw_capability.get("description") or description or "").strip(),
                risk=risk,
                aliases=aliases,
                tools=tools,
                routing=dict(routing),
                summon=summon,
                source_path=str(source_path),
            )
            records.append(record)
            for alias in {name, *aliases}:
                alias_index.setdefault(alias.lower(), []).append(record)

        capabilities_by_profile[profile] = tuple(records)

    by_alias = {alias: tuple(records) for alias, records in alias_index.items()}
    return CapabilityRegistry(
        version=version,
        source_path=source_path,
        profiles=tuple(profile_names),
        profile_descriptions=profile_descriptions,
        capabilities_by_profile=capabilities_by_profile,
        by_alias=by_alias,
    )


def load_capability_registry(path: str | Path | None = None) -> CapabilityRegistry:
    """Load the capability registry from disk."""
    source_path = _resolve_registry_path(path)
    return _build_registry(_read_registry_file(source_path), source_path)


def validate_registry(path: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        load_capability_registry(path)
    except CapabilityRegistryError as exc:
        errors.append(str(exc))
    return errors


def resolve_capability(
    capability: str,
    *,
    registry: CapabilityRegistry | None = None,
) -> CapabilityResolution:
    name = _normalize_text(capability, field_name="capability")
    reg = registry or load_capability_registry()
    matches = tuple(reg.by_alias.get(name, ()))
    if not matches:
        available = ", ".join(reg.profiles) or "<none>"
        raise CapabilityLookupError(f"Unknown capability {name!r}. Known profiles: {available}")
    if len(matches) > 1:
        owners = ", ".join(sorted({record.display_name for record in matches}))
        raise AmbiguousCapabilityError(f"Capability {name!r} is ambiguous across: {owners}")

    record = matches[0]
    return CapabilityResolution(
        target=name,
        profile=record.profile,
        capability=record.name,
        description=record.description,
        risk=record.risk,
        tools=record.tools,
        routing=dict(record.routing),
        source_path=record.source_path,
        direct_profile=False,
        explanation=f"Capability {name!r} resolves to profile {record.profile!r} via {record.display_name} (risk={record.risk})",
    )


def resolve_target(
    target: str,
    *,
    registry: CapabilityRegistry | None = None,
) -> CapabilityResolution:
    name = _normalize_text(target, field_name="target")
    reg = registry
    if profiles_mod.profile_exists(name):
        if reg is None:
            try:
                reg = load_capability_registry()
            except CapabilityRegistryError:
                reg = None
        profile_desc = ""
        if reg is not None:
            profile_desc = str(reg.profile_descriptions.get(name, "") or "")
        return CapabilityResolution(
            target=name,
            profile=name,
            capability=None,
            description=profile_desc or None,
            risk="unknown",
            tools=(),
            routing={},
            source_path=str(reg.source_path) if reg is not None else "",
            direct_profile=True,
            explanation=f"Direct profile match: {name!r}",
        )
    return resolve_capability(name, registry=reg)


def describe_profile_capabilities(
    profile: str,
    *,
    registry: CapabilityRegistry | None = None,
) -> tuple[CapabilityRecord, ...]:
    name = _normalize_text(profile, field_name="profile")
    reg = registry or load_capability_registry()
    return reg.capabilities_for_profile(name)


def _fmt_csv(values: Iterable[str]) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(items) if items else "—"


def render_registry_summary(
    profile: str | None = None,
    *,
    registry: CapabilityRegistry | None = None,
) -> str:
    reg = registry or load_capability_registry()
    profiles = [profile] if profile else list(reg.profiles)
    lines: list[str] = []
    for prof in profiles:
        desc = reg.profile_descriptions.get(prof, "") or "(no description)"
        lines.append(f"{prof}: {desc}")
        records = reg.capabilities_for_profile(prof)
        if not records:
            lines.append("  - (no declared capabilities)")
            continue
        for record in records:
            aliases = ", ".join(a for a in record.aliases if a.lower() != record.name.lower())
            alias_text = f"; aliases: {aliases}" if aliases else ""
            routing_bits: list[str] = []
            for key, value in sorted(record.routing.items()):
                if isinstance(value, bool):
                    routing_bits.append(f"{key}={str(value).lower()}")
                elif isinstance(value, (list, tuple, set)):
                    routing_bits.append(f"{key}=[{_fmt_csv(value)}]")
                else:
                    routing_bits.append(f"{key}={value}")
            routing_text = f"; routing: {', '.join(routing_bits)}" if routing_bits else ""
            summon_text = "; summon-enabled" if record.summon else ""
            lines.append(
                f"  - {record.name} (risk={record.risk}, tools={_fmt_csv(record.tools)})"
                f"{alias_text}{routing_text}{summon_text}"
            )
    return "\n".join(lines).rstrip()


def registry_as_dict(registry: CapabilityRegistry | None = None) -> dict[str, Any]:
    reg = registry or load_capability_registry()
    profiles: dict[str, Any] = {}
    for profile in reg.profiles:
        profiles[profile] = {
            "description": reg.profile_descriptions.get(profile, "") or "",
            "capabilities": [record.as_dict() for record in reg.capabilities_for_profile(profile)],
        }
    return {
        "version": reg.version,
        "source_path": str(reg.source_path),
        "profiles": profiles,
    }
