"""Prompts for Pass 1 and for recovery. Only text lives here; no processing logic.

Bump PROMPT_VERSION whenever the wording changes: chunks extracted with an older
version are extracted again on the next run.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts.vocabulary import DOMAIN_ABBREVIATIONS, DOMAIN_SUBJECTS
from src.evidence_quality.injection import UNTRUSTED_RULE, fence
from src.knowledge_extraction.llm_schema import CONCEPT_TYPES, RELATION_TYPES

PROMPT_VERSION = "kg-extract-3"   # 2: domain context and abbreviation rules; 3: untrusted-text rule

_SUBJECTS = "\n".join(f"- {subject}" for subject in DOMAIN_SUBJECTS)
_EXPANSIONS = "\n".join(
    f"- {short} = {full}" + (" (only when the passage supports this meaning)" if needs_context else "")
    for short, (full, needs_context) in DOMAIN_ABBREVIATIONS.items())

SYSTEM_PROMPT = f"""You extract a knowledge graph from one passage of a technical document.

Domain
The documents are technical and scientific documents about:
{_SUBJECTS}
Read terms in that sense (for example "integration" is system integration, "verification" is checking a
system against its requirements).

Concepts
- Extract only genuine concepts: technical ideas, methods, processes, principles, named entities
  (organizations, systems, standards) and defined terms that a reader would want as a node in a
  knowledge map. Do not extract every noun or phrase. Skip generic words (system, data, figure,
  page, chapter, example) unless the passage defines them in a specific technical sense.
- "name": the concept as written in the passage, in its most complete form.
- "type": one of {", ".join(CONCEPT_TYPES)}.
- "description": one or two sentences saying what the concept is, using only what this passage states.
  If the passage does not say what it is, describe the role it plays in the passage.
- "aliases": other names for the same concept that appear in this passage (abbreviations, acronyms,
  spelled-out forms). Empty list if none.
- "evidence": an exact quote copied character for character from the passage that mentions the concept.
  Keep it short: one sentence or less.

Relationships
- Add a relationship only when the passage itself states or clearly shows it. Never add one because
  two concepts are merely mentioned together or are similar.
- "source" and "target" must be the "name" (or one of the "aliases") of concepts in your concepts list.
- "type" is one of:
  is_a (source is a kind of target), includes (source contains target as a part),
  uses (source uses or depends on target), defines (source defines or specifies target),
  prerequisite_for (source must come before target), example_of (source is an example of target),
  related_to (a stated connection that fits none of the others).
- "evidence": an exact quote from the passage that contains both concepts and states the relationship.

Abbreviations and acronyms
- Recognise abbreviations and acronyms in the passage (for example NASA, MBSE, V&V, ICD).
- When you know an abbreviation's full form, extract ONE concept: "name" is the full form and the
  abbreviation goes in "aliases". Never extract the abbreviation and its full form as two concepts.
- Use a full form only when the passage states it (for example "Interface Control Document (ICD)") or
  when it is one of these standard expansions in this domain:
{_EXPANSIONS}
- Never guess. If an abbreviation is ambiguous or its meaning is not supported by the passage or the
  list above, leave it unresolved: use the abbreviation itself as "name" and give no full form.

Rules
- Use only this passage. Do not add facts from your own knowledge.
- Quotes must be copied exactly from the passage, including spelling and hyphenation.
- If the passage has no meaningful concepts (for example an index, a list of references, or boilerplate),
  return empty lists.

""" + UNTRUSTED_RULE


def user_prompt(chunk: Mapping[str, Any], document_title: str | None) -> str:
    """The passage plus the context a reader would see (document, section, pages)."""
    pages = chunk["page_number"] if chunk["page_end"] == chunk["page_number"] else \
        f"{chunk['page_number']}-{chunk['page_end']}"
    lines = [f"Document: {document_title or 'Unknown'}", f"Pages: {pages}"]
    if chunk.get("section"):
        lines.append(f"Section: {chunk['section']}")
    lines += ["", "<passage>", fence(chunk["text"]), "</passage>"]
    return "\n".join(lines)


def repair_prompt(error: str) -> str:
    """Sent after an answer that could not be used. It asks for a corrected answer, never for new content."""
    return (
        "Your previous answer could not be used: " + error + "\n"
        "Return the corrected JSON only, following the same schema. Fix the problem described above. "
        "Do not add concepts, relationships or quotes that are not supported by the passage; "
        "if something cannot be supported, leave it out."
    )
