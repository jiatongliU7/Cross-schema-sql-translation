#!/usr/bin/env python3
"""Add one harder table-and-column renaming pair per Spider-dev database.

This augments, rather than replaces, the Dr.Spider synonym/abbreviation pairs.
The existing database-level split is reused so no base database crosses splits.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewrite import rewrite
from scripts.build_pair_dataset import (
    gold_mapping,
    link_or_copy,
    materialize_renamed_database,
    read_json,
    rename_exposure_level,
    schema_document,
    sql_has_join,
    stable_shuffle_seed,
    write_json,
)


# Full table aliases are curated so every ordinary table receives a meaningful
# name with little or no exact token overlap. SQLite's internal sequence table
# is the one deliberate exception because SQLite does not allow renaming it.
TABLE_ALIASES = {
    "stadium": "arena",
    "singer": "vocalist",
    "concert": "recital",
    "singer_in_concert": "recital_lineup",
    "student": "learner",
    "has_pet": "pet_ownership",
    "pets": "companion_animals",
    "continents": "landmasses",
    "countries": "nations",
    "car_makers": "automobile_manufacturers",
    "model_list": "vehicle_catalog",
    "car_names": "vehicle_labels",
    "cars_data": "vehicle_specifications",
    "airlines": "air_carriers",
    "airports": "airfields",
    "flights": "air_trips",
    "employee": "staff_member",
    "shop": "retail_outlet",
    "hiring": "recruitment",
    "evaluation": "assessment",
    "ref_template_types": "blueprint_category_lookup",
    "templates": "document_blueprints",
    "documents": "files",
    "paragraphs": "text_blocks",
    "course": "class_offering",
    "teacher": "instructor",
    "course_arrange": "teaching_assignment",
    "museum": "gallery",
    "visitor": "guest",
    "visit": "attendance_record",
    "players": "competitors",
    "matches": "contests",
    "rankings": "standings",
    "battle": "combat_event",
    "ship": "vessel",
    "death": "casualty",
    "addresses": "locations",
    "courses": "subjects",
    "departments": "academic_units",
    "degree_programs": "qualifications",
    "sections": "class_sections",
    "semesters": "academic_terms",
    "students": "learners",
    "student_enrolment": "registrations",
    "student_enrolment_courses": "registration_subjects",
    "transcripts": "academic_records",
    "transcript_contents": "record_entries",
    "tv_channel": "broadcast_network",
    "tv_series": "television_program",
    "cartoon": "animated_program",
    "poker_player": "card_competitor",
    "people": "persons",
    "area_code_state": "dialing_region",
    "contestants": "candidates",
    "votes": "ballots",
    "city": "municipality",
    "sqlite_sequence": "sqlite_sequence",
    "country": "nation",
    "countrylanguage": "national_tongue",
    "conductor": "music_director",
    "orchestra": "ensemble",
    "performance": "rendition",
    "show": "production",
    "highschooler": "secondary_student",
    "friend": "social_connection",
    "likes": "preferences",
    "breeds": "animal_varieties",
    "charges": "fees",
    "sizes": "dimensions",
    "treatment_types": "procedure_categories",
    "owners": "custodians",
    "dogs": "canines",
    "professionals": "specialists",
    "treatments": "procedures",
    "song": "track",
    "ref_feature_types": "amenity_category_lookup",
    "ref_property_types": "asset_category_lookup",
    "other_available_features": "optional_amenities",
    "properties": "realty_assets",
    "other_property_features": "asset_amenities",
}


FULL_COLUMN_ALIASES = {
    "id": "record_key",
    "uid": "record_key",
    "name": "label",
    "title": "heading",
    "type": "category",
    "code": "tag",
    "code2": "secondary_tag",
    "date": "event_day",
    "year": "calendar_annum",
    "age": "years_old",
    "country": "nation",
    "city": "municipality",
    "state": "province",
    "location": "site",
    "first_name": "given_name",
    "middle_name": "additional_name",
    "last_name": "family_name",
    "fname": "given_name",
    "lname": "family_name",
    "fullname": "complete_name",
    "phone_number": "contact_number",
    "cell_number": "mobile_contact",
    "cell_mobile_number": "mobile_contact",
    "home_phone": "residential_contact",
    "email_address": "electronic_mail",
    "airportcode": "airfield_tag",
    "airportname": "airfield_label",
    "flightno": "trip_number",
    "countrycode": "nation_tag",
    "country_code": "nation_tag",
    "countryid": "nation_key",
    "countryname": "nation_label",
    "countryabbrev": "nation_short_tag",
    "contid": "landmass_key",
    "stuid": "learner_key",
    "petid": "animal_key",
    "makeid": "manufacturer_key",
    "modelid": "design_key",
    "gnp": "gross_national_product",
    "gnpold": "previous_gross_national_product",
    "mpg": "miles_per_gallon",
    "ssn": "national_identifier",
    "capacity": "occupancy_limit",
    "highest": "maximum_value",
    "lowest": "minimum_value",
    "average": "mean_value",
    "theme": "topic",
    "result": "outcome",
    "score": "points",
    "ranking": "standing_position",
    "share": "portion",
    "rating": "assessment_score",
    "weight": "mass_value",
    "height": "stature_value",
    "gender": "sex_category",
    "sex": "gender_category",
    "is_male": "masculine_flag",
    "is_full_time": "full_schedule_flag",
    "population": "resident_count",
    "surfacearea": "land_area",
    "lifeexpectancy": "expected_lifespan",
    "indepyear": "independence_annum",
    "headofstate": "national_leader",
    "governmentform": "governance_structure",
    "localname": "native_label",
    "language": "tongue",
    "isofficial": "official_flag",
    "percentage": "usage_ratio",
    "pixel_aspect_ratio_par": "display_pixel_ratio",
    "hight_definition_tv": "high_definition_flag",
    "viewers_m": "audience_millions",
    "official_ratings_in_millions": "certified_audience_millions",
    "18_49_rating_share": "adult_demographic_portion",
    "pay_per_view_ppv": "paid_viewing_flag",
    "net_worth_millions": "wealth_millions",
    "best_finish": "top_placement",
    "money_rank": "earnings_position",
    "best_of": "maximum_rounds",
    "draw_size": "bracket_size",
    "tourney_level": "competition_tier",
    "tourney_date": "competition_day",
    "tourney_name": "competition_label",
    "winner_ioc": "victor_nation_tag",
    "loser_ioc": "defeated_nation_tag",
    "winner_ht": "victor_stature",
    "loser_ht": "defeated_stature",
    "winner_entry": "victor_entry_type",
    "loser_entry": "defeated_entry_type",
    "winner_hand": "victor_hand",
    "loser_hand": "defeated_hand",
    "sourceairport": "origin_airfield",
    "destairport": "destination_airfield",
    "airline": "carrier",
    "abbreviation": "short_tag",
    "accelerate": "acceleration_time",
    "horsepower": "engine_power",
    "cylinders": "engine_cylinder_count",
    "edispl": "engine_displacement",
    "maker": "manufacturer",
    "make": "brand",
    "model": "design",
    "continent": "landmass",
    "region": "territory",
    "street": "road_name",
    "zip_code": "postal_tag",
    "zip_postcode": "postal_tag",
    "line_1": "location_line_one",
    "line_2": "location_line_two",
    "line_3": "location_line_three",
    "other_details": "supplementary_notes",
    "note": "annotation",
    "content": "body_text",
}


TOKEN_ALIASES = {
    "id": "key",
    "name": "label",
    "code": "tag",
    "type": "category",
    "description": "summary",
    "date": "day",
    "year": "annum",
    "number": "count",
    "num": "count",
    "first": "given",
    "last": "family",
    "middle": "additional",
    "country": "nation",
    "city": "municipality",
    "state": "province",
    "address": "location",
    "phone": "contact",
    "email": "mailbox",
    "student": "learner",
    "course": "subject",
    "teacher": "instructor",
    "employee": "staff",
    "shop": "outlet",
    "hiring": "recruitment",
    "evaluation": "assessment",
    "document": "file",
    "template": "blueprint",
    "paragraph": "block",
    "museum": "gallery",
    "visitor": "guest",
    "visit": "attendance",
    "player": "competitor",
    "match": "contest",
    "ranking": "standing",
    "rank": "position",
    "battle": "combat",
    "death": "casualty",
    "degree": "qualification",
    "program": "curriculum",
    "department": "unit",
    "section": "class",
    "semester": "term",
    "enrolment": "registration",
    "transcript": "record",
    "content": "entry",
    "channel": "network",
    "series": "program",
    "cartoon": "animation",
    "poker": "card",
    "people": "person",
    "area": "region",
    "contestant": "candidate",
    "vote": "ballot",
    "language": "tongue",
    "conductor": "director",
    "orchestra": "ensemble",
    "performance": "rendition",
    "show": "production",
    "friend": "connection",
    "liked": "preferred",
    "breed": "variety",
    "charge": "fee",
    "size": "dimension",
    "treatment": "procedure",
    "owner": "custodian",
    "dog": "canine",
    "professional": "specialist",
    "property": "asset",
    "feature": "amenity",
    "available": "offered",
    "other": "supplementary",
    "ref": "lookup",
    "singer": "vocalist",
    "song": "track",
    "concert": "recital",
    "stadium": "arena",
    "airline": "carrier",
    "airport": "airfield",
    "flight": "trip",
    "car": "vehicle",
    "model": "design",
    "maker": "manufacturer",
    "continent": "landmass",
    "pet": "animal",
    "amount": "value",
    "price": "cost",
    "selling": "sale",
    "offered": "proposed",
    "requested": "asked",
    "current": "present",
    "permanent": "lasting",
    "created": "established",
    "effective": "valid",
    "birth": "nativity",
    "tourney": "competition",
    "winner": "victor",
    "loser": "defeated",
    "record": "entry",
    "company": "firm",
    "release": "publication",
    "weekly": "seven_day",
    "total": "aggregate",
    "cost": "expense",
    "room": "space",
    "count": "quantity",
    "sales": "revenue",
}


def snake_tokens(identifier: str) -> list[str]:
    identifier = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    return [token for token in re.split(r"[^A-Za-z0-9]+", identifier.lower()) if token]


def semantic_column_alias(identifier: str) -> str:
    full = FULL_COLUMN_ALIASES.get(identifier.lower())
    if full:
        return full
    tokens = snake_tokens(identifier)
    renamed = [TOKEN_ALIASES.get(token, token) for token in tokens]
    return "_".join(renamed) if renamed != tokens else identifier


def build_target_schema(source: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(source)
    target["db_id"] = f"{source['db_id']}_challenge_0"

    target_tables: list[str] = []
    for table_name in source["table_names_original"]:
        key = table_name.lower()
        if key not in TABLE_ALIASES:
            raise ValueError(f"Missing curated table alias: {table_name}")
        target_tables.append(TABLE_ALIASES[key])
    if len({name.casefold() for name in target_tables}) != len(target_tables):
        raise ValueError(f"Target table collision in {source['db_id']}")

    target_columns: list[list[Any]] = []
    used_by_table: defaultdict[int, set[str]] = defaultdict(set)
    for column_index, (table_index, column_name) in enumerate(
        source["column_names_original"]
    ):
        if table_index < 0:
            target_columns.append([table_index, column_name])
            continue
        if target_tables[table_index] == "sqlite_sequence":
            candidate = column_name
        else:
            candidate = semantic_column_alias(column_name)
        base = candidate
        suffix = 2
        while candidate.casefold() in used_by_table[table_index]:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_by_table[table_index].add(candidate.casefold())
        target_columns.append([table_index, candidate])

    target["table_names_original"] = target_tables
    target["table_names"] = [name.replace("_", " ") for name in target_tables]
    target["column_names_original"] = target_columns
    target["column_names"] = [
        [table_index, name.replace("_", " ")]
        for table_index, name in target_columns
    ]
    return target


def token_overlap(left: str, right: str) -> bool:
    return bool(set(snake_tokens(left)) & set(snake_tokens(right)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("work/raw_drspider"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("outputs/challenge_data_audit.json"),
    )
    parser.add_argument("--seed", type=int, default=4705)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    data_root = args.data_root.resolve()
    split_path = data_root / "splits.json"
    split_document = read_json(split_path)

    split_for_db = {
        db_id: split
        for split, db_ids in split_document["metadata"]["base_databases"].items()
        for db_id in db_ids
    }
    source_schemas = read_json(raw_root / "Spider-dev" / "tables.json")
    questions = read_json(raw_root / "Spider-dev" / "questions.json")
    questions_by_db: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        questions_by_db[row["db_id"]].append(row)

    total_tables = renamed_tables = overlap_tables = 0
    total_columns = renamed_columns = overlap_columns = 0
    pair_audits: dict[str, Any] = {}
    new_pair_ids: list[str] = []
    split_additions: Counter[str] = Counter()
    query_additions: Counter[str] = Counter()

    for source_schema in source_schemas:
        base_db_id = source_schema["db_id"]
        target_schema = build_target_schema(source_schema)
        mapping = gold_mapping(source_schema, target_schema)
        pair_id = f"{base_db_id}__challenge_v0"
        pair_dir = data_root / "pairs" / pair_id
        if pair_dir.exists():
            raise FileExistsError(f"Challenge pair already exists: {pair_dir}")
        pair_dir.mkdir()

        source_db = (
            raw_root
            / "Spider-dev"
            / "databases"
            / base_db_id
            / f"{base_db_id}.sqlite"
        )
        if not source_db.is_file():
            raise FileNotFoundError(source_db)
        link_or_copy(source_db, pair_dir / "db_A.sqlite")
        materialize_renamed_database(source_db, pair_dir / "db_B.sqlite", mapping)

        shuffle_seed = stable_shuffle_seed(args.seed, pair_id)
        write_json(
            pair_dir / "schema_A.json",
            schema_document(source_schema, shuffle=False, seed=shuffle_seed),
        )
        write_json(
            pair_dir / "schema_B.json",
            schema_document(target_schema, shuffle=True, seed=shuffle_seed),
        )
        write_json(pair_dir / "gold_mapping.json", mapping)

        level_counts: Counter[str] = Counter()
        join_count = 0
        pair_questions = questions_by_db[base_db_id]
        with (pair_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
            for row in pair_questions:
                source_qid = int(row["q_id"])
                qid = f"{pair_id}__q{source_qid:04d}"
                is_join = sql_has_join(row["sql"])
                level = rename_exposure_level(
                    row["sql"], source_schema, target_schema
                )
                join_count += int(is_join)
                level_counts[level] += 1
                output = {
                    "qid": qid,
                    "q_A": row["query"],
                    "gold_q_B": rewrite(row["query"], mapping),
                    "is_join": is_join,
                    "level": level,
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

        ordinary_source_tables = [
            name
            for name in source_schema["table_names_original"]
            if name != "sqlite_sequence"
        ]
        ordinary_table_pairs = [
            (source, target)
            for source, target in mapping["tables"].items()
            if source != "sqlite_sequence"
        ]
        pair_total_tables = len(ordinary_source_tables)
        pair_renamed_tables = sum(a != b for a, b in ordinary_table_pairs)
        pair_overlap_tables = sum(token_overlap(a, b) for a, b in ordinary_table_pairs)

        source_column_records = source_schema["column_names_original"]
        target_column_records = target_schema["column_names_original"]
        ordinary_column_pairs = [
            (source_name, target_name)
            for (table_index, source_name), (_, target_name) in zip(
                source_column_records, target_column_records
            )
            if table_index >= 0
            and source_schema["table_names_original"][table_index] != "sqlite_sequence"
        ]
        pair_total_columns = len(ordinary_column_pairs)
        pair_renamed_columns = sum(a != b for a, b in ordinary_column_pairs)
        pair_overlap_columns = sum(
            token_overlap(a, b)
            for a, b in ordinary_column_pairs
            if a != b
        )

        total_tables += pair_total_tables
        renamed_tables += pair_renamed_tables
        overlap_tables += pair_overlap_tables
        total_columns += pair_total_columns
        renamed_columns += pair_renamed_columns
        overlap_columns += pair_overlap_columns

        split = split_for_db[base_db_id]
        split_document[split].append(pair_id)
        split_document[split].sort()
        split_additions[split] += 1
        query_additions[split] += len(pair_questions)
        pair_metadata = {
            "base_db_id": base_db_id,
            "target_db_id": target_schema["db_id"],
            "perturbation": "challenge_semantic_table_and_column_rename",
            "split": split,
            "query_count": len(pair_questions),
            "join_query_count": join_count,
            "level_counts": dict(sorted(level_counts.items())),
            "changed_table_count": pair_renamed_tables,
            "changed_column_count": pair_renamed_columns,
        }
        split_document["metadata"]["pairs"][pair_id] = pair_metadata
        pair_audits[pair_id] = {
            **pair_metadata,
            "ordinary_table_count": pair_total_tables,
            "table_token_overlap_count": pair_overlap_tables,
            "ordinary_column_count": pair_total_columns,
            "renamed_column_token_overlap_count": pair_overlap_columns,
        }
        new_pair_ids.append(pair_id)

    table_rename_rate = renamed_tables / total_tables
    column_rename_rate = renamed_columns / total_columns
    table_overlap_rate = overlap_tables / renamed_tables
    column_overlap_rate = overlap_columns / renamed_columns
    if table_rename_rate < 0.99:
        raise ValueError(f"Table rename coverage too low: {table_rename_rate:.3f}")
    if column_rename_rate < 0.65:
        raise ValueError(f"Column rename coverage too low: {column_rename_rate:.3f}")

    split_document["metadata"]["dataset_groups"] = {
        "standard": {
            "description": "Dr.Spider synonym and abbreviation variants",
            "pair_count": sum(
                meta["perturbation"] in {"synonym", "abbreviation"}
                for meta in split_document["metadata"]["pairs"].values()
            ),
        },
        "challenge": {
            "description": "Curated semantic table and column renaming variants",
            "pair_count": len(new_pair_ids),
            "pair_ids": sorted(new_pair_ids),
        },
    }
    write_json(split_path, split_document)

    audit = {
        "status": "generated; run validate_pair_dataset.py --execute before use",
        "pair_count": len(new_pair_ids),
        "base_database_count": len(source_schemas),
        "query_count": sum(query_additions.values()),
        "split_pair_counts": dict(sorted(split_additions.items())),
        "split_query_counts": dict(sorted(query_additions.items())),
        "ordinary_table_count": total_tables,
        "renamed_table_count": renamed_tables,
        "table_rename_rate": table_rename_rate,
        "renamed_table_token_overlap_rate": table_overlap_rate,
        "ordinary_column_count": total_columns,
        "renamed_column_count": renamed_columns,
        "column_rename_rate": column_rename_rate,
        "renamed_column_token_overlap_rate": column_overlap_rate,
        "sqlite_sequence_note": "SQLite internal table and its columns remain unchanged.",
        "independence_limit": (
            "Challenge pairs reuse the same 20 Spider-dev base databases and "
            "therefore increase renaming difficulty, not independent schema diversity."
        ),
        "pairs": dict(sorted(pair_audits.items())),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.audit, audit)

    print(json.dumps({key: value for key, value in audit.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
