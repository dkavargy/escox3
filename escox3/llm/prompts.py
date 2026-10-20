"""
Versioned prompt templates for ESCOX 3.0.

The exact strings below are what the LLM receives; they are reproduced in
docs/LLM_COMPONENT.md for the manuscript. Change PROMPT_VERSION whenever a template changes.
"""

from __future__ import annotations

from typing import Dict, List

PROMPT_VERSION = "escox3-prompts-v1.0"

SYSTEM = (
    "You are an expert annotator of European labour-market documents. You classify text using the ESCO "
    "skills and occupations taxonomy, the ISCO-08 occupation classification and the NACE Rev. 2.1 "
    "classification of economic activities. You are strict: you only choose concepts that the text "
    "genuinely supports, you never invent concepts, and you reply with a single JSON object only."
)

_RULES_COMMON = (
    "Rules:\n"
    "- You may ONLY use ids from the CANDIDATES list.\n"
    "- \"explicit\" = the concept is directly stated in the text (same or equivalent wording).\n"
    "- \"implicit\" = the concept is not named but is clearly required, possessed, practised or taught "
    "according to the text.\n"
    "- Do NOT select a candidate that is only loosely related to the topic of the text.\n"
    "- \"evidence\" must be a short quotation copied exactly from the TEXT (max 25 words).\n"
    "- If no candidate is supported, return {\"selected\": []}.\n"
)

TASKS = {
    "green_skills": (
        "TASK: The candidates below are GREEN skills and knowledge concepts from the ESCO green skills "
        "collection. Select every candidate that the TEXT genuinely mentions or clearly implies as a skill or "
        "knowledge that is required, possessed, practised or taught."
    ),
    "green_occupations": (
        "TASK: Select the ESCO OCCUPATIONS that best describe the role the TEXT is about (the job advertised, "
        "the position held or sought by the person, or the occupation a course trains for). Choose at most 3, "
        "best match first. Do not choose an occupation only because it shares some skills with the text."
    ),
    "sectors": (
        "TASK: Select the NACE Rev. 2.1 ECONOMIC ACTIVITIES that describe the business activity of the "
        "employer or organisation, or the economic activity the TEXT is about. A sector is what the organisation "
        "produces or does, NOT the job title: e.g. a software developer working for a hospital belongs to "
        "hospital activities, not to computer programming. Choose at most 3, best match first. If the text "
        "gives no evidence about the economic activity, return an empty list."
    ),
}

CANDIDATE_HEADER = {
    "green_skills": "CANDIDATES (id | green skill | definition):",
    "green_occupations": "CANDIDATES (id | occupation | ISCO-08 unit group | definition):",
    "sectors": "CANDIDATES (id | NACE code | economic activity):",
}

OUTPUT_FORMAT = (
    "Reply with JSON only, in this format:\n"
    "{\"selected\": [{\"id\": \"<candidate id>\", \"support\": \"explicit\" or \"implicit\", "
    "\"evidence\": \"<exact quote from the TEXT>\"}]}"
)


def build_selection_messages(module: str, text: str, candidate_lines: List[str]) -> List[Dict[str, str]]:
    user = (
        f"{TASKS[module]}\n\n{_RULES_COMMON}\n"
        f"TEXT:\n\"\"\"\n{text}\n\"\"\"\n\n"
        f"{CANDIDATE_HEADER[module]}\n" + "\n".join(candidate_lines) + "\n\n" + OUTPUT_FORMAT
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def selection_schema(candidate_ids: List[str], max_items: int, evidence_max_chars: int = 200) -> dict:
    """JSON schema used for grammar-constrained decoding (ids restricted to the candidate list)."""
    return {
        "type": "object",
        "properties": {
            "selected": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": candidate_ids},
                        "support": {"type": "string", "enum": ["explicit", "implicit"]},
                        "evidence": {"type": "string", "maxLength": evidence_max_chars},
                    },
                    "required": ["id", "support", "evidence"],
                },
            }
        },
        "required": ["selected"],
    }


# ---------------------------------------------------------------- Experiment B: LLM-only (no retrieval)
OPEN_TASKS = {
    "green_skills": (
        "TASK: List the GREEN skills and knowledge (environmental sustainability, renewable energy, energy "
        "efficiency, circular economy, waste, pollution, biodiversity, climate...) that the TEXT mentions or "
        "clearly implies. Use the exact preferred labels of the ESCO green skills collection."
    ),
    "green_occupations": (
        "TASK: Name the ESCO occupations (at most 3) that best describe the role the TEXT is about. Use exact "
        "ESCO occupation preferred labels."
    ),
    "sectors": (
        "TASK: Name the NACE Rev. 2.1 economic activities (at most 3) describing the business activity of the "
        "employer/organisation in the TEXT. Give the NACE code in the crosswalk notation (section letter + "
        "code, e.g. D35.11) and its official title. Return an empty list if the text gives no evidence."
    ),
}


def build_open_messages(module: str, text: str) -> List[Dict[str, str]]:
    fmt = ("{\"items\": [{\"label\": \"<exact taxonomy label>\", \"code\": \"<NACE code or empty>\", "
           "\"support\": \"explicit\" or \"implicit\", \"evidence\": \"<exact quote from the TEXT>\"}]}")
    user = (f"{OPEN_TASKS[module]}\n\nRules:\n- Do not invent labels; only use labels that exist in the official "
            f"taxonomy.\n- \"evidence\" must be copied exactly from the TEXT.\n\n"
            f"TEXT:\n\"\"\"\n{text}\n\"\"\"\n\nReply with JSON only, in this format:\n{fmt}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def open_schema(max_items: int, evidence_max_chars: int = 200) -> dict:
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "maxItems": max_items, "items": {
            "type": "object",
            "properties": {"label": {"type": "string", "maxLength": 150},
                           "code": {"type": "string", "maxLength": 10},
                           "support": {"type": "string", "enum": ["explicit", "implicit"]},
                           "evidence": {"type": "string", "maxLength": evidence_max_chars}},
            "required": ["label", "code", "support", "evidence"]}}},
        "required": ["items"],
    }
