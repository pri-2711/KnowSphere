"""Extensible ontology: the core concept and relation types plus per-project domain extensions.

The core types are the extraction vocabulary (``vocabulary.CONCEPT_TYPES``, ``RELATION_TYPES``),
which every model prompt and validation rule uses. A project can add domain types, each a
specialisation of an existing type ("subsystem" is an "artifact", "verifies" is "related_to"). Every
extension resolves to exactly one core type, so extraction, merging and governance keep working on
core types while reviewers, agents and the interface can use the finer domain vocabulary.

Extensions are data, not code: a project's extensions are stored by the repository and can be
loaded from a JSON file (see ``schemas/ontology/``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.contracts.schemas import Contract
from src.contracts.vocabulary import CONCEPT_TYPES, RELATION_TYPES

Category = Literal["concept", "relation"]


class OntologyType(Contract):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[a-z][a-z0-9_ ]*$")
    category: Category
    parent: str | None = None            # None only for core types
    description: str = ""
    domain: str = "core"                 # "core" or the extension's domain name, e.g. "space systems"


CORE_TYPES: tuple[OntologyType, ...] = tuple(
    [OntologyType(name=name, category="concept") for name in CONCEPT_TYPES]
    + [OntologyType(name=name, category="relation") for name in RELATION_TYPES])


class Ontology(Contract):
    """The core types plus a project's extensions; construction fails on an invalid extension."""
    project_id: str
    extensions: list[OntologyType] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self) -> "Ontology":
        known = {(t.category, t.name) for t in CORE_TYPES}
        for extension in self.extensions:     # parents must be defined before their children
            if (extension.category, extension.name) in known:
                raise ValueError(f"type '{extension.name}' is already defined")
            if extension.domain == "core" or extension.parent is None:
                raise ValueError(f"extension '{extension.name}' needs a domain and a parent type")
            if (extension.category, extension.parent) not in known:
                raise ValueError(f"parent '{extension.parent}' of '{extension.name}' is not a known "
                                 f"{extension.category} type")
            known.add((extension.category, extension.name))
        return self

    @property
    def types(self) -> list[OntologyType]:
        return [*CORE_TYPES, *self.extensions]

    def names(self, category: Category) -> list[str]:
        return [t.name for t in self.types if t.category == category]

    def core_of(self, name: str, category: Category = "concept") -> str | None:
        """The core type a type specialises (itself for a core type), or None when unknown."""
        parents = {t.name: t.parent for t in self.types if t.category == category}
        seen = set()
        while name in parents and parents[name] is not None:
            if name in seen:
                return None
            seen.add(name)
            name = parents[name]
        return name if name in parents else None

    def tree(self) -> dict[str, list[str]]:
        """Core type -> its extensions (all levels), for display."""
        result: dict[str, list[str]] = {t.name: [] for t in CORE_TYPES}
        for extension in self.extensions:
            core = self.core_of(extension.name, extension.category)
            if core is not None:
                result.setdefault(core, []).append(extension.name)
        return result
