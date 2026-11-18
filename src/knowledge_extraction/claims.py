"""Claim extraction for one semantic chunk (candidates only; governance decides acceptance).

A claim is an assertion the passage makes: subject, predicate, object, the full
statement, and whether it is affirmed or negated. The model must quote its
evidence; the quote is located in the chunk text. Quantities are read from the
located quote by ``evidence_quality.units``, never taken from the model.

Claims are stored on the chunk as ``claims_extraction`` with their own prompt
version, so adding claims never invalidates the stored concept extractions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from src.contracts.ids import DEFAULT_PROJECT, scoped_id
from src.contracts.interfaces import TextModel
from src.contracts.schemas import Claim, Evidence, KnowledgeStatus, Provenance, SourceRef, UnitType
from src.evidence_quality.injection import UNTRUSTED_RULE, fence, injection_markers
from src.evidence_quality.uncertainty import MODALITIES, modality_hint, uncertainty_markers, weaker
from src.evidence_quality.units import contains_quantity, find_quantities
from src.evidence_quality.validation import make_evidence
from src.knowledge_extraction.recovery import run_with_recovery
from src.textutil import concept_key

CLAIM_PROMPT_VERSION = "claims-2"   # 2: kind, modality, untrusted-text rule
CLAIM_KINDS = ("requirement", "specification", "measurement", "definition", "cause", "comparison", "method",
               "limitation", "result", "statement")
EXTRACTED, FAILED = "extracted", "failed"

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"},
            "statement": {"type": "string"}, "polarity": {"type": "string", "enum": ["affirmed", "negated"]},
            "kind": {"type": "string", "enum": list(CLAIM_KINDS)},
            "modality": {"type": "string", "enum": list(MODALITIES)},
            "evidence": {"type": "string"},
        },
        "required": ["subject", "predicate", "object", "statement", "polarity", "kind", "modality", "evidence"],
        "additionalProperties": False,
    }}},
    "required": ["claims"],
    "additionalProperties": False,
}

CLAIM_SYSTEM_PROMPT = """You list the factual claims one passage of a technical document makes.

- A claim is a single assertion the passage states: a requirement, a measured or specified value,
  a definition, a cause and effect, a comparison. Skip opinions, questions, headings and boilerplate.
- "subject": the thing the claim is about, as written in the passage.
- "predicate": the relation in a few words (for example "requires", "is limited to", "consumes", "causes").
- "object": what the subject is related to, including any number and unit exactly as written.
- "statement": the claim as one short sentence using only the passage's information.
- "polarity": "negated" when the passage says something is not, must not or does not happen; else "affirmed".
- "kind": requirement (shall, must), specification (a designed or rated value), measurement (an observed
  value), definition, cause (cause and effect), comparison, method (how something is done or was studied),
  limitation (a stated limit, weakness or open problem of a method or result), result (a finding), or
  statement for anything else.
- "modality": "fact" for a plain assertion, requirement or measured value; "interpretation" when the
  passage infers or hedges (suggests, indicates, likely, may); "hypothesis" when it proposes or speculates.
- "evidence": an exact quote copied character for character from the passage that states the claim.
- Use only this passage. Never add numbers, units or facts that are not in it.
- Return an empty list when the passage makes no factual claims.

""" + UNTRUSTED_RULE


def claim_user_prompt(chunk: Mapping[str, Any], document_title: str | None) -> str:
    return "\n".join([f"Document: {document_title or 'Unknown'}", f"Pages: {chunk['page_number']}",
                      "", "<passage>", fence(chunk["text"]), "</passage>"])


def claim_structure_errors(data: Any) -> list[str]:
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return ["the answer must be a JSON object with a 'claims' list"]
    errors = []
    for index, item in enumerate(data["claims"]):
        if not isinstance(item, dict):
            errors.append(f"claims[{index}] must be an object")
            continue
        for name in ("subject", "predicate", "object", "statement", "polarity", "evidence"):   # kind, modality optional
            if not isinstance(item.get(name), str):
                errors.append(f"claims[{index}].{name} must be a string")
    return errors[:10]


def claims_are_current(chunk: Mapping[str, Any], model: str) -> bool:
    record = chunk.get("claims_extraction")
    return bool(record) and record.get("status") == EXTRACTED and record.get("text_sha256") == chunk["text_sha256"] \
        and record.get("prompt_version") == CLAIM_PROMPT_VERSION and record.get("model") == model


def local_claim(raw: Mapping[str, Any], chunk: Mapping[str, Any]) -> dict[str, Any]:
    """Locate the evidence and read quantities from it; flag what governance will look at."""
    evidence = make_evidence(chunk, str(raw.get("evidence", "")))
    claim = {key: str(raw.get(key, "")).strip() for key in ("subject", "predicate", "object", "statement")}
    claim["polarity"] = raw.get("polarity") if raw.get("polarity") in ("affirmed", "negated") else "affirmed"
    claim["kind"] = raw.get("kind") if raw.get("kind") in CLAIM_KINDS else "statement"
    stated = raw.get("modality") if raw.get("modality") in MODALITIES else "fact"
    wording = evidence["quote"] if evidence else ""
    claim["modality"] = weaker(stated, modality_hint(wording))    # the source's own hedging always counts
    claim["uncertainty"] = uncertainty_markers(wording)
    claim.update(evidence=evidence, status="supported", issues=[])
    quantities = find_quantities(evidence["quote"]) if evidence else []
    claim["quantities"] = [quantity.model_dump(mode="json", exclude={"schema_version"}) for quantity in quantities]
    if not claim["subject"] or not claim["statement"]:
        claim.update(status="rejected", issues=["claim has no subject or statement"])
    elif evidence is None:
        claim.update(status="rejected", issues=["evidence quote not found in the chunk"])
    else:
        unsupported = [q.raw for q in find_quantities(claim["statement"] + " " + claim["object"])
                       if not contains_quantity(quantities, q)]
        if unsupported:
            claim.update(status="needs_review", issues=[f"numbers not in the evidence: {', '.join(unsupported)}"])
    if injection_markers(chunk.get("text", "")):
        claim.update(status="needs_review" if claim["status"] == "supported" else claim["status"],
                     issues=claim["issues"] + ["passage contains instruction-like text"])
    return claim


def extract_claims_chunk(chunk: Mapping[str, Any], client: TextModel, max_retries: int,
                         document_title: str | None = None) -> dict[str, Any]:
    result = run_with_recovery(
        lambda messages: client.complete(CLAIM_SYSTEM_PROMPT, messages, CLAIM_SCHEMA),
        claim_user_prompt(chunk, document_title), max_retries, claim_structure_errors)
    record: dict[str, Any] = {
        "status": EXTRACTED if result.ok else FAILED, "method": "llm", "text_sha256": chunk["text_sha256"],
        "prompt_version": CLAIM_PROMPT_VERSION, "model": client.model, "attempts": result.attempts,
        "retries": result.retries, "error": result.error, "extracted_at": datetime.now(timezone.utc), "claims": [],
    }
    if result.ok and result.data is not None:
        record["claims"] = [local_claim(raw, chunk) for raw in result.data["claims"]]
    return record


def build_claims(chunks: Iterable[Mapping[str, Any]], concepts: Iterable[Mapping[str, Any]],
                 project_id: str = DEFAULT_PROJECT) -> list[dict[str, Any]]:
    """Canonical claim records for ``kg_claims`` from a project's chunk claim extractions."""
    names: dict[str, str] = {}
    for concept in concepts:
        for name in (concept["canonical_name"], *concept.get("aliases", [])):
            names.setdefault(concept_key(name), concept["_id"])
    records: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        extraction = chunk.get("claims_extraction") or {}
        if extraction.get("status") != EXTRACTED:
            continue
        provenance = Provenance(component="knowledge_extraction", method=extraction.get("method") or "llm",
                                model=extraction.get("model"),
                                prompt_version=extraction.get("prompt_version"))
        for raw in extraction.get("claims", []):
            if raw["status"] == "rejected" or not raw.get("evidence"):
                continue
            found = raw["evidence"]
            source = SourceRef(project_id=project_id, document_id=chunk["document_id"],
                               content_unit_id=found.get("content_unit_id") or chunk["_id"],
                               unit_type=UnitType.PAGE if found.get("content_unit_id") else UnitType.CHUNK,
                               page_number=found.get("page_number"),
                               char_start=found.get("page_start", found.get("chunk_start")),
                               char_end=found.get("page_end", found.get("chunk_end")))
            claim_id = scoped_id("kg_claim", project_id, chunk["_id"], concept_key(raw["statement"]))
            claim = Claim(id=claim_id, project_id=project_id, status=KnowledgeStatus.CANDIDATE,
                          evidence=[Evidence(source=source, quote=found["quote"], provenance=provenance)],
                          provenance=[provenance], subject=raw["subject"], predicate=raw["predicate"] or "states",
                          object=raw.get("object", ""), statement=raw["statement"], polarity=raw["polarity"],
                          kind=raw.get("kind", "statement"), modality=raw.get("modality", "fact"),
                          uncertainty=raw.get("uncertainty", []),
                          quantities=raw.get("quantities", []), subject_concept_id=names.get(concept_key(raw["subject"])),
                          object_concept_id=names.get(concept_key(raw.get("object", ""))))
            records[claim_id] = {"_id": claim_id, **claim.model_dump(mode="json", exclude={"id"}),
                                 "document_id": chunk["document_id"], "chunk_id": chunk["_id"]}
    return list(records.values())
