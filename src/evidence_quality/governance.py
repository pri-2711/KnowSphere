"""Evidence governance: decide which candidate knowledge is accepted, and record why.

Extraction proposes candidates; this module alone decides their ``KnowledgeStatus``.
Each check is an independent function returning ``CheckResult`` objects (or nothing
when it does not apply). A failed *blocking* check rejects an item; a failed
non-blocking check sends it to review. A model's confidence score is never read.

Checks
    schema               the record is a valid canonical model (done when it is built)
    evidence_trace       every quote is found in the stored text of the unit it cites
    source_validity      the cited document is in the item's project, the unit belongs to it, the page exists
    endpoint_validity    relationship endpoints are existing, different concepts of the same project
    duplicate_alias      no other concept of the project claims the same name or alias
    numeric_consistency  numbers stated in a claim occur, with compatible units, in its evidence
    contradiction        claims that negate or disagree in value with each other; inverse is_a / includes / prerequisite edges

Status rule (policy ``POLICY_VERSION``)
    adjudicated   a reviewer's decision exists (it is kept across rebuilds)
    rejected      any blocking check failed
    needs_review  any non-blocking check failed
    corroborated  all checks passed and traced evidence comes from two or more documents
    verified      all checks passed
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence, Union

from src.contracts.ids import stable_id
from src.contracts.interfaces import UnitRecord
from src.contracts.schemas import (Adjudication, CheckResult, Claim, Concept, KnowledgeStatus, Relationship,
                                   ValidationRecord)
from src.evidence_quality.units import contains_quantity, find_quantities, same_value
from src.evidence_quality.validation import locate
from src.textutil import concept_key

POLICY_VERSION = "governance-1"
Item = Union[Concept, Claim, Relationship]
_ASYMMETRIC = {"is_a", "includes", "prerequisite_for", "example_of"}


@dataclass
class GovernanceContext:
    """Everything the checks may consult, for one project only."""
    project_id: str
    page_counts: Mapping[str, int]                       # document id -> page count
    units: Mapping[str, UnitRecord]                        # content unit id -> owner and stored text
    concept_ids: set[str] = field(default_factory=set)   # concepts of the graph being validated
    concepts: Sequence[Concept] = ()
    claims: Sequence[Claim] = ()
    relationships: Sequence[Relationship] = ()
    adjudications: Mapping[str, Adjudication] = field(default_factory=dict)
    _alias_owners: dict[str, set[str]] | None = None
    _contradictions: dict[str, list[str]] | None = None

    def alias_owners(self) -> dict[str, set[str]]:
        if self._alias_owners is None:
            owners: dict[str, set[str]] = defaultdict(set)
            for concept in self.concepts:
                for name in {concept.normalized_name, concept_key(concept.name), *map(concept_key, concept.aliases)}:
                    if name:
                        owners[name].add(concept.id)
            self._alias_owners = owners
        return self._alias_owners

    def contradictions(self) -> dict[str, list[str]]:
        if self._contradictions is None:
            self._contradictions = find_contradictions(self.claims, self.relationships)
        return self._contradictions


Check = Callable[[Item, GovernanceContext], Iterable[CheckResult]]


# ------------------------------------------------------------------ checks
def check_evidence_trace(item: Item, ctx: GovernanceContext) -> Iterable[CheckResult]:
    if not item.evidence:
        yield CheckResult(check="evidence_trace", passed=False, message="no evidence quote")
        return
    traced = 0
    for evidence in item.evidence:
        unit = ctx.units.get(evidence.source.content_unit_id)
        found = False
        if unit is not None:
            source = evidence.source
            if source.char_start is not None and source.char_end is not None:
                window = unit.text[max(0, source.char_start - 2):source.char_end + 2]
                found = locate(window, evidence.quote) is not None
            if not found:
                found = locate(unit.text, evidence.quote) is not None
        evidence.traced = found
        traced += found
    if traced == 0:
        yield CheckResult(check="evidence_trace", passed=False, message="no quote was found in its source text")
    elif traced < len(item.evidence):
        yield CheckResult(check="evidence_trace", passed=True, blocking=False,
                          message=f"{len(item.evidence) - traced} of {len(item.evidence)} quotes not found; ignored")
    else:
        yield CheckResult(check="evidence_trace", passed=True)


def check_source_validity(item: Item, ctx: GovernanceContext) -> Iterable[CheckResult]:
    problems = []
    valid = 0
    for evidence in item.evidence:
        source = evidence.source
        unit = ctx.units.get(source.content_unit_id)
        if source.document_id not in ctx.page_counts:
            problems.append(f"document {source.document_id[:12]} is not in project '{ctx.project_id}'")
        elif unit is None or unit.document_id != source.document_id:
            problems.append(f"unit {source.content_unit_id[:20]} does not belong to its document")
        elif source.page_number is not None and not 1 <= source.page_number <= max(ctx.page_counts[source.document_id], 1):
            problems.append(f"page {source.page_number} is outside the document")
        else:
            valid += 1
    if item.evidence and valid == 0:
        yield CheckResult(check="source_validity", passed=False, message="; ".join(problems[:3]))
    elif problems:
        yield CheckResult(check="source_validity", passed=False, blocking=False, message="; ".join(problems[:3]))
    elif item.evidence:
        yield CheckResult(check="source_validity", passed=True)


def check_endpoints(item: Item, ctx: GovernanceContext) -> Iterable[CheckResult]:
    if isinstance(item, Relationship):
        missing = [end for end in (item.source_id, item.target_id) if end not in ctx.concept_ids]
        yield CheckResult(check="endpoint_validity", passed=not missing,
                          message=f"unknown endpoint(s): {', '.join(m[:24] for m in missing)}" if missing else "")
    elif isinstance(item, Claim):
        linked = [end for end in (item.subject_concept_id, item.object_concept_id) if end]
        missing = [end for end in linked if end not in ctx.concept_ids]
        if linked:
            yield CheckResult(check="endpoint_validity", passed=not missing, blocking=False,
                              message="claim links to an unknown concept" if missing else "")


def check_duplicate_alias(item: Item, ctx: GovernanceContext) -> Iterable[CheckResult]:
    if not isinstance(item, Concept):
        return
    others: set[str] = set()
    owners = ctx.alias_owners()
    for name in {item.normalized_name, concept_key(item.name), *map(concept_key, item.aliases)}:
        others |= owners.get(name, set()) - {item.id}
    yield CheckResult(check="duplicate_alias", passed=not others, blocking=False,
                      message=f"shares a name or alias with {len(others)} other concept(s)" if others else "")


def check_possible_duplicate(item: Item, ctx: GovernanceContext) -> Iterable[CheckResult]:
    """Canonical resolution could not decide whether this concept and another are one (left for review)."""
    if not isinstance(item, Concept) or not item.possible_duplicates:
        return
    yield CheckResult(check="possible_duplicate", passed=False, blocking=False,
                      message=f"may be the same as {len(item.possible_duplicates)} other concept(s); not merged")


def check_numeric_consistency(item: Item, ctx: GovernanceContext) -> Iterable[CheckResult]:
    if not isinstance(item, Claim):
        return
    stated = find_quantities(item.statement)
    if not stated and not item.quantities:
        return
    quoted = [quantity for evidence in item.evidence if evidence.traced is not False
              for quantity in find_quantities(evidence.quote)]
    missing = [quantity.raw for quantity in [*stated, *item.quantities] if not contains_quantity(quoted, quantity)]
    yield CheckResult(check="numeric_consistency", passed=not missing,
                      message=f"not in the evidence: {', '.join(missing)}" if missing else "")


def check_contradiction(item: Item, ctx: GovernanceContext) -> Iterable[CheckResult]:
    if isinstance(item, Concept):
        return
    conflicts = ctx.contradictions().get(item.id, [])
    yield CheckResult(check="contradiction", passed=not conflicts, blocking=False,
                      message="; ".join(conflicts[:3]))


CHECKS: list[Check] = [check_evidence_trace, check_source_validity, check_endpoints, check_duplicate_alias,
                       check_possible_duplicate, check_numeric_consistency, check_contradiction]


def find_contradictions(claims: Sequence[Claim], relationships: Sequence[Relationship]) -> dict[str, list[str]]:
    """Item id -> descriptions of the items it conflicts with (within the given, single-project lists)."""
    conflicts: dict[str, list[str]] = defaultdict(list)
    by_topic: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for claim in claims:
        by_topic[(concept_key(claim.subject), concept_key(claim.predicate))].append(claim)
    for group in by_topic.values():
        for index, a in enumerate(group):
            for b in group[index + 1:]:
                reason = None
                if a.polarity != b.polarity and concept_key(a.object) == concept_key(b.object):
                    reason = "one source affirms what another negates"
                elif a.polarity == b.polarity:
                    for qa in a.quantities:
                        if any(same_value(qa, qb) is False for qb in b.quantities):
                            reason = f"different values ({qa.raw} vs {', '.join(q.raw for q in b.quantities)})"
                            break
                if reason:
                    conflicts[a.id].append(f"{reason}: {b.statement[:80]}")
                    conflicts[b.id].append(f"{reason}: {a.statement[:80]}")
    directed = {(r.source_id, r.relation_type, r.target_id): r for r in relationships if r.relation_type in _ASYMMETRIC}
    for (source, relation, target), edge in directed.items():
        inverse = directed.get((target, relation, source))
        if inverse is not None:
            conflicts[edge.id].append(f"the inverse '{relation}' edge also exists")
    return conflicts


# ------------------------------------------------------------------ decision
def decide(item: Item, checks: list[CheckResult], adjudication: Adjudication | None) -> KnowledgeStatus:
    if adjudication is not None:
        return KnowledgeStatus.ADJUDICATED
    if any(not check.passed and check.blocking for check in checks):
        return KnowledgeStatus.REJECTED
    if any(not check.passed for check in checks):
        return KnowledgeStatus.NEEDS_REVIEW
    traced_documents = {e.source.document_id for e in item.evidence if e.traced}
    return KnowledgeStatus.CORROBORATED if len(traced_documents) >= 2 else KnowledgeStatus.VERIFIED


def govern(items: Iterable[Item], ctx: GovernanceContext,
           checks: Sequence[Check] = CHECKS) -> tuple[list[ValidationRecord], dict[str, KnowledgeStatus]]:
    """Run every check on every item of one project. Returns the records and the new status per item id."""
    records: list[ValidationRecord] = []
    statuses: dict[str, KnowledgeStatus] = {}
    for item in items:
        if item.project_id != ctx.project_id:
            raise ValueError(f"item {item.id} belongs to project '{item.project_id}', not '{ctx.project_id}'")
        results = [CheckResult(check="schema", passed=True)]
        for check in checks:
            results.extend(check(item, ctx))
        adjudication = ctx.adjudications.get(item.id)
        status = decide(item, results, adjudication)
        item_type = "concept" if isinstance(item, Concept) else "claim" if isinstance(item, Claim) else "relationship"
        records.append(ValidationRecord(
            id=stable_id("validation", ctx.project_id, item.id), project_id=ctx.project_id, item_id=item.id,
            item_type=item_type, checks=results, status_before=item.status, status_after=status,
            policy_version=POLICY_VERSION))
        statuses[item.id] = status
    return records, statuses


def schema_rejection(project_id: str, item_id: str, item_type: str, error: str) -> ValidationRecord:
    """The record for a stored item that could not even be read as a canonical model."""
    return ValidationRecord(
        id=stable_id("validation", project_id, item_id), project_id=project_id, item_id=item_id,
        item_type=item_type, checks=[CheckResult(check="schema", passed=False, message=error[:300])],
        status_before=KnowledgeStatus.CANDIDATE, status_after=KnowledgeStatus.REJECTED, policy_version=POLICY_VERSION)
