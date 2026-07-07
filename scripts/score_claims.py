#!/usr/bin/env python3
"""Compare structured claims against GroundingEval scene ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JSONDict = dict[str, Any]
TAXONOMY_CATEGORIES = (
    "correctness",
    "hallucination",
    "omission",
    "wrong_count",
    "wrong_relation",
    "wrong_state",
    "wrong_temporal_event",
    "unsupported_answer",
    "ambiguous_answer",
)
AMBIGUOUS_MODALITIES = {"uncertain", "ambiguous", "unknown"}
STATUS_SUPPORTED = "SUPPORTED"
STATUS_CONTRADICTED = "CONTRADICTED"
STATUS_NOT_EVALUABLE = "NOT_EVALUABLE"
EVENT_ALIASES = {
    "entered": "entry",
    "entry": "entry",
    "exited": "exit",
    "exit": "exit",
}
DEFAULT_EVENT_TOLERANCE_SECONDS = 1.0


class ScoringError(Exception):
    """Raised when the input files cannot be scored safely."""


@dataclass(frozen=True)
class EvalContext:
    scene_id: str
    objects: list[JSONDict]
    relations: list[JSONDict]
    events: list[JSONDict] | None
    object_index: dict[str, JSONDict]
    objects_by_type: dict[str, list[JSONDict]]


@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    claim_type: str
    natural_language: str
    status: str
    category: str
    error_type: str | None
    evidence: list[str]

    def to_dict(self) -> JSONDict:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "natural_language": self.natural_language,
            "status": self.status,
            "category": self.category,
            "error_type": self.error_type,
            "evidence": self.evidence,
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score structured claims against GroundingEval ground truth."
    )
    parser.add_argument("ground_truth", help="Path to ground_truth.json")
    parser.add_argument("claims", help="Path to claims.json")
    parser.add_argument(
        "--out",
        help="Optional output path for the JSON report. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def load_json_file(path_str: str) -> JSONDict:
    path = Path(path_str)
    if not path.is_file():
        raise ScoringError(f"Input file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ScoringError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ScoringError(f"Could not read {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ScoringError(f"Expected a JSON object in {path}, found {type(payload).__name__}")

    return payload


def ensure_mapping_list(value: Any, field_name: str) -> list[JSONDict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ScoringError(f"Expected '{field_name}' to be a list.")

    items: list[JSONDict] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ScoringError(
                f"Expected '{field_name}[{index}]' to be an object, found {type(item).__name__}."
            )
        items.append(item)
    return items


def require_mapping_list(data: JSONDict, field_name: str, context: str) -> list[JSONDict]:
    if field_name not in data:
        raise ScoringError(f"Missing required list field '{field_name}' in {context}.")
    return ensure_mapping_list(data.get(field_name), field_name)


def build_context(ground_truth: JSONDict, claims_payload: JSONDict) -> EvalContext:
    gt_scene_id = require_str(ground_truth, "scene_id", "ground truth")
    claims_scene_id = require_str(claims_payload, "scene_id", "claims")

    if gt_scene_id != claims_scene_id:
        raise ScoringError(
            f"Scene ID mismatch: ground truth is '{gt_scene_id}' but claims file is '{claims_scene_id}'."
        )

    objects = require_mapping_list(ground_truth, "objects", "ground truth")
    relations = ensure_mapping_list(ground_truth.get("relations"), "relations")
    events = ground_truth.get("events")
    event_items = ensure_mapping_list(events, "events") if events is not None else None

    object_index: dict[str, JSONDict] = {}
    objects_by_type: dict[str, list[JSONDict]] = defaultdict(list)

    for index, obj in enumerate(objects):
        object_id = require_str(obj, "id", f"objects[{index}]")
        object_type = require_str(obj, "type", f"objects[{index}]")
        object_index[object_id] = obj
        objects_by_type[object_type].append(obj)

    return EvalContext(
        scene_id=gt_scene_id,
        objects=objects,
        relations=relations,
        events=event_items,
        object_index=object_index,
        objects_by_type=dict(objects_by_type),
    )


def require_str(data: JSONDict, field_name: str, context: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ScoringError(f"Missing required string field '{field_name}' in {context}.")
    return value


def make_result(
    claim_id: str,
    claim_type: str,
    natural_language: str,
    status: str,
    category: str,
    evidence: list[str],
) -> ClaimResult:
    error_type = None if category == "correctness" else category
    return ClaimResult(
        claim_id=claim_id,
        claim_type=claim_type,
        natural_language=natural_language,
        status=status,
        category=category,
        error_type=error_type,
        evidence=evidence,
    )


def get_claim_identity(claim: JSONDict, claim_index: int) -> tuple[str, str, str]:
    claim_id = require_str(claim, "claim_id", f"claims[{claim_index}]")
    claim_type = require_str(claim, "claim_type", f"claims[{claim_index}]")
    natural_language = claim.get("natural_language")
    if natural_language is None:
        natural_language = ""
    elif not isinstance(natural_language, str):
        raise ScoringError(
            f"Expected 'natural_language' in claims[{claim_index}] to be a string when present."
        )
    return claim_id, claim_type, natural_language


def evaluate_all_claims(context: EvalContext, claims_payload: JSONDict) -> JSONDict:
    claims = require_mapping_list(claims_payload, "claims", "claims payload")
    claim_results: list[ClaimResult] = []

    for index, claim in enumerate(claims):
        claim_id, claim_type, natural_language = get_claim_identity(claim, index)
        claim_results.append(evaluate_claim(claim, context, claim_id, claim_type, natural_language))

    summary = summarize_results(claim_results)
    return {
        "scene_id": context.scene_id,
        "summary": summary,
        "claim_results": [result.to_dict() for result in claim_results],
    }


def summarize_results(claim_results: list[ClaimResult]) -> JSONDict:
    by_category = {category: 0 for category in TAXONOMY_CATEGORIES}
    supported = 0
    contradicted = 0
    not_evaluable = 0

    for result in claim_results:
        by_category[result.category] += 1
        if result.status == STATUS_SUPPORTED:
            supported += 1
        elif result.status == STATUS_CONTRADICTED:
            contradicted += 1
        else:
            not_evaluable += 1

    return {
        "total_claims": len(claim_results),
        "supported": supported,
        "contradicted": contradicted,
        "not_evaluable": not_evaluable,
        "by_category": by_category,
        # Omission depends on question expectations rather than scene state alone.
        # We leave a stable placeholder until qa_pairs.json or required-facts data is wired in.
        "omission_check": "skipped_no_qa_expected_facts",
    }


def evaluate_claim(
    claim: JSONDict,
    context: EvalContext,
    claim_id: str,
    claim_type: str,
    natural_language: str,
) -> ClaimResult:
    modality = str(claim.get("modality", "asserted")).strip().lower()
    if modality in AMBIGUOUS_MODALITIES:
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_NOT_EVALUABLE,
            "ambiguous_answer",
            [f"Claim modality '{modality}' is ambiguous and cannot be scored deterministically."],
        )

    evaluator = {
        "attribute": evaluate_attribute_claim,
        "relation": evaluate_relation_claim,
        "count": evaluate_count_claim,
        "object_presence": evaluate_object_presence_claim,
        "event": evaluate_event_claim,
    }.get(claim_type)

    if evaluator is None:
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_NOT_EVALUABLE,
            "unsupported_answer",
            [f"Claim type '{claim_type}' is not implemented by this scorer."],
        )

    try:
        return evaluator(claim, context, claim_id, claim_type, natural_language)
    except ScoringError as exc:
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_NOT_EVALUABLE,
            "unsupported_answer",
            [str(exc)],
        )


def get_claim_polarity(claim: JSONDict) -> str:
    polarity = str(claim.get("polarity", "positive")).strip().lower()
    if polarity not in {"positive", "negative"}:
        raise ScoringError(f"Unsupported polarity '{polarity}'.")
    return polarity


def require_entity_type(claim: JSONDict, field_name: str) -> str:
    entity = claim.get(field_name)
    if not isinstance(entity, dict):
        raise ScoringError(f"Claim field '{field_name}' must be an object.")
    entity_type = entity.get("type")
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ScoringError(f"Claim field '{field_name}.type' is required.")
    return entity_type


def infer_count_target_type(claim: JSONDict) -> str:
    subject = claim.get("subject")
    if isinstance(subject, dict) and isinstance(subject.get("type"), str) and subject["type"].strip():
        return subject["type"]

    count_spec = claim.get("count")
    if isinstance(count_spec, dict):
        unit = count_spec.get("unit")
        if isinstance(unit, str) and unit.strip():
            return unit

    obj = claim.get("object")
    if isinstance(obj, dict) and isinstance(obj.get("type"), str) and obj["type"].strip():
        return obj["type"]

    raise ScoringError("Could not determine the target type for the count claim.")


def evaluate_attribute_claim(
    claim: JSONDict,
    context: EvalContext,
    claim_id: str,
    claim_type: str,
    natural_language: str,
) -> ClaimResult:
    subject_type = require_entity_type(claim, "subject")
    predicate = require_str(claim, "predicate", f"claim {claim_id}")
    property_value = claim.get("property_value")
    if property_value is None:
        raise ScoringError("Attribute claims require a non-null 'property_value'.")
    polarity = get_claim_polarity(claim)

    matching_objects = context.objects_by_type.get(subject_type, [])
    evidence = [f"Matched subject type {subject_type} to {len(matching_objects)} ground truth object(s)."]

    if not matching_objects:
        status = STATUS_CONTRADICTED if polarity == "positive" else STATUS_SUPPORTED
        category = "hallucination" if polarity == "positive" else "correctness"
        evidence.append(f"No ground truth objects of type {subject_type} were found.")
        return make_result(claim_id, claim_type, natural_language, status, category, evidence)

    checkable_objects = [obj for obj in matching_objects if predicate in obj]
    if not checkable_objects:
        evidence.append(
            f"None of the {subject_type} objects expose direct field '{predicate}' in ground truth."
        )
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_NOT_EVALUABLE,
            "unsupported_answer",
            evidence,
        )

    matching_value_objects = [obj for obj in checkable_objects if obj.get(predicate) == property_value]
    for obj in checkable_objects:
        evidence.append(f"Checked {obj['id']}.{predicate} == {json.dumps(property_value)}.")

    if polarity == "positive":
        if matching_value_objects:
            return make_result(
                claim_id,
                claim_type,
                natural_language,
                STATUS_SUPPORTED,
                "correctness",
                evidence,
            )

        category = "wrong_state" if predicate == "state" else "unsupported_answer"
        evidence.append(
            f"Found {len(checkable_objects)} matching subject(s), but none had {predicate}={json.dumps(property_value)}."
        )
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_CONTRADICTED,
            category,
            evidence,
        )

    if matching_value_objects:
        category = "wrong_state" if predicate == "state" else "unsupported_answer"
        evidence.append(
            f"Negative claim contradicted by {len(matching_value_objects)} object(s) with {predicate}={json.dumps(property_value)}."
        )
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_CONTRADICTED,
            category,
            evidence,
        )

    evidence.append(
        f"No {subject_type} objects had {predicate}={json.dumps(property_value)} among the checkable subjects."
    )
    return make_result(
        claim_id,
        claim_type,
        natural_language,
        STATUS_SUPPORTED,
        "correctness",
        evidence,
    )


def evaluate_relation_claim(
    claim: JSONDict,
    context: EvalContext,
    claim_id: str,
    claim_type: str,
    natural_language: str,
) -> ClaimResult:
    subject_type = require_entity_type(claim, "subject")
    object_type = require_entity_type(claim, "object")
    predicate = require_str(claim, "predicate", f"claim {claim_id}")
    polarity = get_claim_polarity(claim)

    subject_objects = context.objects_by_type.get(subject_type, [])
    object_objects = context.objects_by_type.get(object_type, [])
    evidence = [
        f"Matched subject type {subject_type} to {len(subject_objects)} ground truth object(s).",
        f"Matched object type {object_type} to {len(object_objects)} ground truth object(s).",
    ]

    if not subject_objects or not object_objects:
        status = STATUS_CONTRADICTED if polarity == "positive" else STATUS_SUPPORTED
        category = "hallucination" if polarity == "positive" else "correctness"
        evidence.append(
            f"Relation target is missing because subject type {subject_type} or object type {object_type} does not exist."
        )
        return make_result(claim_id, claim_type, natural_language, status, category, evidence)

    subject_ids = {obj["id"] for obj in subject_objects}
    object_ids = {obj["id"] for obj in object_objects}
    matching_relations = [
        relation
        for relation in context.relations
        if relation.get("predicate") == predicate
        and relation.get("subject") in subject_ids
        and relation.get("object") in object_ids
    ]
    evidence.append(
        f"Checked {len(context.relations)} relation(s) for {subject_type} --{predicate}--> {object_type}."
    )

    if polarity == "positive":
        if matching_relations:
            relation = matching_relations[0]
            evidence.append(
                f"Matched relation {relation.get('subject')} --{predicate}--> {relation.get('object')}."
            )
            return make_result(
                claim_id,
                claim_type,
                natural_language,
                STATUS_SUPPORTED,
                "correctness",
                evidence,
            )

        evidence.append(
            f"No ground truth relation matched predicate '{predicate}' between types {subject_type} and {object_type}."
        )
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_CONTRADICTED,
            "wrong_relation",
            evidence,
        )

    if matching_relations:
        relation = matching_relations[0]
        evidence.append(
            f"Negative claim contradicted by relation {relation.get('subject')} --{predicate}--> {relation.get('object')}."
        )
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_CONTRADICTED,
            "wrong_relation",
            evidence,
        )

    evidence.append(f"No matching {predicate} relation exists between {subject_type} and {object_type}.")
    return make_result(
        claim_id,
        claim_type,
        natural_language,
        STATUS_SUPPORTED,
        "correctness",
        evidence,
    )


def evaluate_count_claim(
    claim: JSONDict,
    context: EvalContext,
    claim_id: str,
    claim_type: str,
    natural_language: str,
) -> ClaimResult:
    count_spec = claim.get("count")
    if not isinstance(count_spec, dict):
        raise ScoringError("Count claims require a 'count' object.")

    target_type = infer_count_target_type(claim)
    operator = require_str(count_spec, "operator", f"claim {claim_id} count")
    value = count_spec.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScoringError("Count claim field 'count.value' must be a number.")
    polarity = get_claim_polarity(claim)

    actual_count = len(context.objects_by_type.get(target_type, []))
    evidence = [f"Counted {actual_count} ground truth object(s) of type {target_type}."]

    comparison = compare_count(actual_count, value, operator)
    evidence.append(f"Applied operator '{operator}' against expected value {value}.")

    if polarity == "negative":
        comparison = not comparison
        evidence.append("Inverted count comparison because claim polarity is negative.")

    if comparison:
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_SUPPORTED,
            "correctness",
            evidence,
        )

    evidence.append(
        f"Ground truth count {actual_count} does not satisfy the claim requirement after polarity handling."
    )
    return make_result(
        claim_id,
        claim_type,
        natural_language,
        STATUS_CONTRADICTED,
        "wrong_count",
        evidence,
    )


def compare_count(actual: int, expected: float, operator: str) -> bool:
    if operator == "exact":
        return actual == expected
    if operator == "at_least":
        return actual >= expected
    if operator == "at_most":
        return actual <= expected
    if operator == "greater_than":
        return actual > expected
    if operator == "less_than":
        return actual < expected
    raise ScoringError(f"Unsupported count operator '{operator}'.")


def evaluate_object_presence_claim(
    claim: JSONDict,
    context: EvalContext,
    claim_id: str,
    claim_type: str,
    natural_language: str,
) -> ClaimResult:
    target_type = infer_presence_target_type(claim)
    polarity = get_claim_polarity(claim)

    exists = len(context.objects_by_type.get(target_type, [])) > 0
    evidence = [f"Checked object presence for type {target_type}: exists={exists}."]

    if polarity == "positive":
        if exists:
            return make_result(
                claim_id,
                claim_type,
                natural_language,
                STATUS_SUPPORTED,
                "correctness",
                evidence,
            )
        evidence.append(f"No ground truth objects of type {target_type} were found.")
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_CONTRADICTED,
            "hallucination",
            evidence,
        )

    if exists:
        evidence.append(
            f"Negative presence claim contradicted because ground truth includes at least one {target_type}."
        )
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_CONTRADICTED,
            "unsupported_answer",
            evidence,
        )

    evidence.append(f"Ground truth contains no {target_type}, so the negative claim is satisfied.")
    return make_result(
        claim_id,
        claim_type,
        natural_language,
        STATUS_SUPPORTED,
        "correctness",
        evidence,
    )


def infer_presence_target_type(claim: JSONDict) -> str:
    for field_name in ("subject", "object"):
        entity = claim.get(field_name)
        if isinstance(entity, dict):
            entity_type = entity.get("type")
            if isinstance(entity_type, str) and entity_type.strip():
                return entity_type
    raise ScoringError("Could not determine the target type for the object_presence claim.")


def evaluate_event_claim(
    claim: JSONDict,
    context: EvalContext,
    claim_id: str,
    claim_type: str,
    natural_language: str,
) -> ClaimResult:
    evidence: list[str] = []
    polarity = get_claim_polarity(claim)

    if context.events is None:
        evidence.append("Ground truth has no 'events' list, so event claims cannot be evaluated.")
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_NOT_EVALUABLE,
            "unsupported_answer",
            evidence,
        )

    event_kind = normalize_event_kind(claim)
    if event_kind is None:
        raise ScoringError("Event claim must include predicate/event_type for entry or exit.")
    evidence.append(f"Normalized event kind to '{event_kind}'.")

    candidates = [
        event
        for event in context.events
        if event_matches_claim(event, claim, context, event_kind)
    ]
    evidence.append(f"Found {len(candidates)} matching ground truth event candidate(s).")

    if not candidates:
        if polarity == "negative":
            evidence.append("No matching ground truth event exists, so the negative claim is satisfied.")
            return make_result(
                claim_id,
                claim_type,
                natural_language,
                STATUS_SUPPORTED,
                "correctness",
                evidence,
            )

        evidence.append("No matching ground truth event exists for the asserted claim.")
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_CONTRADICTED,
            "hallucination",
            evidence,
        )

    claim_time = claim.get("time")
    if claim_time is None:
        if polarity == "negative":
            evidence.append("Negative event claim contradicted by at least one matching ground truth event.")
            return make_result(
                claim_id,
                claim_type,
                natural_language,
                STATUS_CONTRADICTED,
                "unsupported_answer",
                evidence,
            )

        evidence.append("Matched event by type/participants without requiring a temporal comparison.")
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_SUPPORTED,
            "correctness",
            evidence,
        )

    comparable_times = []
    for event in candidates:
        event_time = event.get("time")
        if isinstance(event_time, dict):
            comparable_times.append((event, event_time))

    if not comparable_times:
        evidence.append("Matching events exist, but none expose a comparable 'time' object.")
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_NOT_EVALUABLE,
            "unsupported_answer",
            evidence,
        )

    for event, event_time in comparable_times:
        if times_match(claim_time, event_time, DEFAULT_EVENT_TOLERANCE_SECONDS):
            event_label = describe_event(event)
            evidence.append(
                f"Matched event time against {event_label} within {DEFAULT_EVENT_TOLERANCE_SECONDS:.1f}s tolerance."
            )
            if polarity == "negative":
                evidence.append("Negative event claim contradicted by a matching timed event.")
                return make_result(
                    claim_id,
                    claim_type,
                    natural_language,
                    STATUS_CONTRADICTED,
                    "wrong_temporal_event",
                    evidence,
                )
            return make_result(
                claim_id,
                claim_type,
                natural_language,
                STATUS_SUPPORTED,
                "correctness",
                evidence,
            )

    evidence.append("Matching event exists, but the provided claim time does not align with ground truth.")
    if polarity == "negative":
        evidence.append("No event matched the claimed time, so the negative claim is satisfied.")
        return make_result(
            claim_id,
            claim_type,
            natural_language,
            STATUS_SUPPORTED,
            "correctness",
            evidence,
        )

    return make_result(
        claim_id,
        claim_type,
        natural_language,
        STATUS_CONTRADICTED,
        "wrong_temporal_event",
        evidence,
    )


def event_matches_claim(
    event: JSONDict,
    claim: JSONDict,
    context: EvalContext,
    required_kind: str,
) -> bool:
    event_kind = normalize_event_kind(event)
    if event_kind != required_kind:
        return False

    subject = claim.get("subject")
    if isinstance(subject, dict) and not event_role_matches(event.get("subject"), subject, context):
        return False

    obj = claim.get("object")
    if isinstance(obj, dict) and not event_role_matches(event.get("object"), obj, context):
        return False

    return True


def event_role_matches(role_value: Any, claim_entity: JSONDict, context: EvalContext) -> bool:
    claim_type = claim_entity.get("type")
    if not isinstance(claim_type, str) or not claim_type.strip():
        raise ScoringError("Event claim entity is missing a usable 'type'.")

    if isinstance(role_value, str):
        ground_truth_object = context.object_index.get(role_value)
        if ground_truth_object is None:
            return False
        return ground_truth_object.get("type") == claim_type

    if isinstance(role_value, dict):
        role_type = role_value.get("type")
        if isinstance(role_type, str) and role_type == claim_type:
            return True
        role_id = role_value.get("id")
        if isinstance(role_id, str):
            ground_truth_object = context.object_index.get(role_id)
            return ground_truth_object is not None and ground_truth_object.get("type") == claim_type

    return False


def normalize_event_kind_from_value(raw_value: Any) -> str | None:
    if isinstance(raw_value, str):
        return EVENT_ALIASES.get(raw_value.strip().lower())
    return None


def normalize_event_kind(event_or_claim: JSONDict) -> str | None:
    return (
        normalize_event_kind_from_value(event_or_claim.get("predicate"))
        or normalize_event_kind_from_value(event_or_claim.get("event_type"))
    )


def times_match(claim_time: Any, event_time: Any, tolerance_seconds: float) -> bool:
    claim_range = normalize_time_range(claim_time, tolerance_seconds)
    event_range = normalize_time_range(event_time, tolerance_seconds)
    if claim_range is None or event_range is None:
        raise ScoringError("Unsupported event time format.")

    claim_start, claim_end, claim_is_point = claim_range
    event_start, event_end, event_is_point = event_range

    if claim_is_point and event_is_point:
        return abs(claim_start - event_start) <= tolerance_seconds

    if claim_is_point and not event_is_point:
        return event_start - tolerance_seconds <= claim_start <= event_end + tolerance_seconds

    if not claim_is_point and event_is_point:
        return claim_start - tolerance_seconds <= event_start <= claim_end + tolerance_seconds

    return (
        abs(claim_start - event_start) <= tolerance_seconds
        and abs(claim_end - event_end) <= tolerance_seconds
    )


def normalize_time_range(time_obj: Any, tolerance_seconds: float) -> tuple[float, float, bool] | None:
    if not isinstance(time_obj, dict):
        return None

    time_type = time_obj.get("type")
    if time_type == "point":
        point = time_obj.get("t")
        if not isinstance(point, (int, float)) or isinstance(point, bool):
            return None
        value = float(point)
        return (value, value, True)

    if time_type == "interval":
        start = time_obj.get("start")
        end = time_obj.get("end")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
        ):
            return None
        if float(end) < float(start):
            raise ScoringError("Event interval end must be greater than or equal to start.")
        return (float(start), float(end), False)

    return None


def describe_event(event: JSONDict) -> str:
    subject = event.get("subject", "?")
    predicate = event.get("predicate") or event.get("event_type") or "event"
    obj = event.get("object")
    if obj is None:
        return f"{subject} {predicate}"
    return f"{subject} {predicate} {obj}"


def write_output(report: JSONDict, out_path: str | None) -> None:
    payload = json.dumps(report, indent=2)
    if out_path is None:
        print(payload)
        return

    destination = Path(out_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        raise ScoringError(f"Could not write output file {destination}: {exc}") from exc


def main() -> int:
    args = parse_args()

    try:
        ground_truth = load_json_file(args.ground_truth)
        claims_payload = load_json_file(args.claims)
        context = build_context(ground_truth, claims_payload)
        report = evaluate_all_claims(context, claims_payload)
        write_output(report, args.out)
    except ScoringError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
