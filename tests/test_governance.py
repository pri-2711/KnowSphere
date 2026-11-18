"""Evidence governance: every check on its own, the status rule, and units."""

from __future__ import annotations

from src.contracts.interfaces import UnitRecord
from src.contracts.schemas import (Adjudication, Claim, Concept, Evidence, KnowledgeStatus, Provenance, Quantity,
                                   Relationship, SourceRef)
from src.evidence_quality.governance import GovernanceContext, govern
from src.evidence_quality.units import find_quantities, same_value

P = "p1"
PROV = Provenance(component="test", method="llm")
TEXT_A = "The water processor recovers 93 percent of water. It requires 4.5 kg/day of consumables."
TEXT_B = "The water processor recovers 93 percent of water from urine."
UNITS = {"ua": UnitRecord("da", 1, TEXT_A), "ub": UnitRecord("db", 1, TEXT_B)}


def ev(quote: str, unit: str = "ua", document: str = "da", page: int = 1) -> Evidence:
    text = UNITS[unit].text if unit in UNITS else ""
    start = text.find(quote)
    offsets = {"char_start": start, "char_end": start + len(quote)} if start >= 0 else {}
    return Evidence(source=SourceRef(project_id=P, document_id=document, content_unit_id=unit, page_number=page,
                                     **offsets), quote=quote, provenance=PROV)


def concept(cid: str, name: str, evidence: list[Evidence], aliases=(), confidence=None) -> Concept:
    return Concept(id=cid, project_id=P, name=name, normalized_name=name.lower(), aliases=list(aliases),
                   evidence=evidence, confidence=confidence)


def context(**kwargs) -> GovernanceContext:
    return GovernanceContext(project_id=P, page_counts={"da": 3, "db": 2}, units=UNITS, **kwargs)


def run(items, **kwargs):
    records, statuses = govern(items, context(**kwargs))
    return {r.item_id: r for r in records}, statuses


def failed(record) -> set[str]:
    return {check.check for check in record.checks if not check.passed}


def test_traced_evidence_is_verified_and_two_documents_corroborate() -> None:
    one = concept("c1", "water processor", [ev("The water processor recovers 93 percent of water")])
    two = concept("c2", "urine", [ev("The water processor recovers 93 percent of water"),
                                  ev("recovers 93 percent of water from urine", "ub", "db")])
    _, statuses = run([one, two], concepts=[one, two])
    assert statuses == {"c1": KnowledgeStatus.VERIFIED, "c2": KnowledgeStatus.CORROBORATED}
    assert one.evidence[0].traced is True


def test_model_confidence_never_rescues_untraceable_evidence() -> None:
    invented = concept("c1", "plasma recycler", [ev("A plasma recycler is used.")], confidence=1.0)
    records, statuses = run([invented], concepts=[invented])
    assert statuses["c1"] is KnowledgeStatus.REJECTED and "evidence_trace" in failed(records["c1"])
    no_evidence = concept("c2", "anything", [], confidence=0.99)
    assert run([no_evidence], concepts=[no_evidence])[1]["c2"] is KnowledgeStatus.REJECTED


def test_source_validity_rejects_foreign_documents_and_impossible_pages() -> None:
    foreign = concept("c1", "water", [ev("water", unit="ux", document="other-project-doc")])
    bad_page = concept("c2", "water processor", [ev("The water processor", page=99)])
    records, statuses = run([foreign, bad_page], concepts=[foreign, bad_page])
    assert statuses["c1"] is KnowledgeStatus.REJECTED and "source_validity" in failed(records["c1"])
    assert statuses["c2"] is KnowledgeStatus.REJECTED and "page 99" in records["c2"].checks[2].message


def test_relationship_endpoints_and_inverse_contradiction() -> None:
    a = concept("a", "water processor", [ev("The water processor")])
    b = concept("b", "consumables", [ev("consumables")])
    good = Relationship(id="r1", project_id=P, source_id="a", target_id="b", relation_type="uses",
                        evidence=[ev("It requires 4.5 kg/day of consumables")])
    dangling = Relationship(id="r2", project_id=P, source_id="a", target_id="ghost", relation_type="uses",
                            evidence=[ev("It requires 4.5 kg/day of consumables")])
    forward = Relationship(id="r3", project_id=P, source_id="a", target_id="b", relation_type="includes",
                           evidence=[ev("The water processor")])
    backward = Relationship(id="r4", project_id=P, source_id="b", target_id="a", relation_type="includes",
                            evidence=[ev("consumables")])
    edges = [good, dangling, forward, backward]
    records, statuses = run([a, b, *edges], concepts=[a, b], concept_ids={"a", "b"}, relationships=edges)
    assert statuses["r1"] is KnowledgeStatus.VERIFIED
    assert statuses["r2"] is KnowledgeStatus.REJECTED and "endpoint_validity" in failed(records["r2"])
    assert statuses["r3"] is statuses["r4"] is KnowledgeStatus.NEEDS_REVIEW
    assert "contradiction" in failed(records["r3"])


def test_duplicate_alias_sends_both_concepts_to_review() -> None:
    a = concept("a", "Water Processor Assembly", [ev("The water processor")], aliases=["WPA"])
    b = concept("b", "WPA", [ev("water")])
    records, statuses = run([a, b], concepts=[a, b])
    assert statuses == {"a": KnowledgeStatus.NEEDS_REVIEW, "b": KnowledgeStatus.NEEDS_REVIEW}
    assert "duplicate_alias" in failed(records["a"])


def claim(cid: str, statement: str, quote: str, unit="ua", document="da", polarity="affirmed",
          quantities=None) -> Claim:
    return Claim(id=cid, project_id=P, subject="water processor", predicate="recovers", object="water",
                 statement=statement, polarity=polarity, evidence=[ev(quote, unit, document)],
                 quantities=quantities if quantities is not None else find_quantities(quote))


def test_numbers_must_come_from_the_evidence() -> None:
    honest = claim("k1", "The water processor recovers 93 percent of water.",
                   "The water processor recovers 93 percent of water")
    inflated = claim("k2", "The water processor recovers 98 percent of water.",
                     "The water processor recovers 93 percent of water")
    records, statuses = run([honest, inflated])
    assert statuses["k1"] is KnowledgeStatus.VERIFIED
    assert statuses["k2"] is KnowledgeStatus.REJECTED and "98" in records["k2"].checks[-2].message


def test_contradicting_claims_are_contested_not_accepted() -> None:
    a = claim("k1", "The water processor recovers 93 percent of water.",
              "The water processor recovers 93 percent of water")
    b = claim("k2", "The water processor recovers water from urine.", "recovers 93 percent of water from urine",
              "ub", "db", quantities=[Quantity(value=80, unit="%", raw="80%")])
    c = claim("k3", "The water processor does not recover water.", "The water processor recovers 93 percent of water",
              polarity="negated", quantities=[])
    claims = [a, b, c]
    records, statuses = run(claims, claims=claims)
    assert all(statuses[key] is KnowledgeStatus.NEEDS_REVIEW for key in ("k1", "k3"))
    assert "contradiction" in failed(records["k1"]) and "contradiction" in failed(records["k3"])


def test_adjudication_overrides_the_automatic_status() -> None:
    invented = concept("c1", "plasma recycler", [ev("A plasma recycler is used.")])
    decision = Adjudication(reviewer="analyst", decision="accept", note="seen on the figure")
    _, statuses = run([invented], concepts=[invented], adjudications={"c1": decision})
    assert statuses["c1"] is KnowledgeStatus.ADJUDICATED


def test_units_compare_across_scales() -> None:
    found = find_quantities("Crew needs 4.5 kg/day and 3 systems at 101.3 kPa, 93%.")
    assert [(q.value, q.unit) for q in found] == [(4.5, "kg/day"), (3.0, None), (101.3, "kpa"), (93.0, "%")]
    assert same_value(Quantity(value=4500, unit="g/day", raw=""), Quantity(value=4.5, unit="kg/day", raw="")) is True
    assert same_value(Quantity(value=5, unit="kg", raw=""), Quantity(value=5, unit="m", raw="")) is None
