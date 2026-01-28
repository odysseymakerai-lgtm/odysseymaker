import os
import json
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from openai import OpenAI

# -----------------------------
# Domain Schemas (contract)
# -----------------------------

Tone = Literal["grimdark", "heroic", "whimsical", "horror", "mystery", "epic"]
Theme = Literal["fey", "undead", "dragons", "political", "heist", "cosmic", "wilderness", "dungeon", "urban"]

EncounterType = Literal["combat", "social", "exploration", "puzzle", "skill_challenge"]
Difficulty = Literal["easy", "medium", "hard", "deadly"]

LevelingMode = Literal["milestone", "xp"]
Ruleset = Literal["5e", "system_agnostic"]

class OutlineRequest(BaseModel):
    concept: str = Field(min_length=10, description="User's general concept / pitch")
    party_level_start: int = Field(ge=1, le=20, default=3)
    party_level_end: int = Field(ge=1, le=20, default=5)
    ruleset: Ruleset = "5e"
    leveling_mode: LevelingMode = "milestone"
    tone: Tone = "mystery"
    theme: Theme = "dungeon"
    session_count_target: int = Field(ge=1, le=20, default=3)
    constraints: List[str] = Field(default_factory=list, description="Hard constraints (e.g., 'no demons', 'include a heist')")
    include_travel: bool = True

class KeyNPC(BaseModel):
    name: str
    role: str
    public_face: str
    secret: str
    leverage: str

class Faction(BaseModel):
    name: str
    goal: str
    method: str
    complication: str

class StoryBeat(BaseModel):
    beat_id: str  # e.g., B1, B2...
    title: str
    purpose: str
    stakes: str
    twist_or_reveal: str

class AdventureOutline(BaseModel):
    title: str
    logline: str
    central_conflict: str
    villain_or_antagonist: str
    themes: List[str]
    hooks: List[str]
    key_npcs: List[KeyNPC]
    factions: List[Faction]
    beats: List[StoryBeat]
    continuity_promises: List[str] = Field(default_factory=list, description="Facts the later detailed outline must honor")

class Encounter(BaseModel):
    encounter_id: str  # E1, E2...
    type: EncounterType
    difficulty: Difficulty
    summary: str
    win_condition: str
    fail_forward: str
    setup: List[str] = Field(default_factory=list)
    scaling_notes: List[str] = Field(default_factory=list, description="How to scale up/down for party strength")

class Scene(BaseModel):
    scene_id: str  # S1, S2...
    title: str
    location: str
    goal: str
    boxed_text: str
    obstacles: List[str]
    encounters: List[Encounter]
    clues_and_info: List[str]
    rewards: List[str]
    consequences: List[str]
    links_to_beats: List[str] = Field(default_factory=list, description="Beat IDs this scene fulfills")
    estimated_minutes: int = Field(ge=5, le=240, default=45)

class LevelProgressionStep(BaseModel):
    step_id: str  # L1, L2...
    after_scene_id: str
    level: int
    rationale: str
    optional_side_objectives: List[str] = Field(default_factory=list)

class DetailedAdventureOutline(BaseModel):
    outline_title: str
    structure_notes: List[str]
    scenes: List[Scene]
    level_progression: List[LevelProgressionStep]
    optional_side_quests: List[str] = Field(default_factory=list)
    recap_questions: List[str] = Field(default_factory=list, description="Questions to ask at session end to guide next gen")

class OutlineResponse(BaseModel):
    outline: AdventureOutline
    detailed: DetailedAdventureOutline

# -----------------------------
# AI helper (Structured JSON)
# -----------------------------

SYSTEM = """You are a veteran tabletop RPG adventure designer.
You output ONLY valid JSON that conforms exactly to the given schema.
Be concrete and playable: specific names, locations, motivations, clear objectives.
Include BOTH combat and non-combat encounters.
Avoid copyrighted setting text; create original material.
If ruleset is 'system_agnostic', avoid 5e jargon (CR, XP tables), but keep difficulty labels.
"""

class AIEngine:
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _extract_text(self, resp) -> str:
        out_text = None
        for item in getattr(resp, "output", []):
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", None) == "output_text":
                        out_text = c.text
                        break
        if not out_text:
            raise RuntimeError("No output_text found in model response.")
        return out_text

    def generate_json_schema(self, *, schema_name: str, schema: dict, user_payload: dict, extra_instructions: str) -> dict:
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": extra_instructions},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        text = self._extract_text(resp)
        return json.loads(text)

# -----------------------------
# Adventure Outline generator
# -----------------------------

def build_outline_prompt(req: OutlineRequest) -> dict:
    return {
        "concept": req.concept,
        "party_level_start": req.party_level_start,
        "party_level_end": req.party_level_end,
        "ruleset": req.ruleset,
        "leveling_mode": req.leveling_mode,
        "tone": req.tone,
        "theme": req.theme,
        "session_count_target": req.session_count_target,
        "constraints": req.constraints,
        "include_travel": req.include_travel,
        "requirements": [
            "Beats should be 5–9 items depending on session_count_target.",
            "Include at least 2 hooks.",
            "Include at least 3 key NPCs and at least 1 faction (2 preferred).",
            "continuity_promises: 5–12 bullet-style facts that must remain true later.",
        ],
    }

def build_detailed_prompt(req: OutlineRequest, outline: AdventureOutline) -> dict:
    return {
        "request": req.model_dump(),
        "outline": outline.model_dump(),
        "requirements": [
            "Create 5–9 scenes (S1..S#). Each scene must link_to_beats.",
            "Each scene must contain encounters including at least one non-combat encounter overall, and at least one combat encounter overall.",
            "Encounters must include win_condition and fail_forward.",
            "Level progression must be structured and sensible from party_level_start to party_level_end.",
            "If leveling_mode is milestone, set level-up steps after key scenes; if xp, describe XP targets in rationale text (don’t include giant tables).",
            "Estimated_minutes should help match session_count_target; keep scenes around 30–75 minutes typically.",
            "Ensure continuity_promises are honored.",
        ],
    }

def generate_outline_pair(ai: AIEngine, req: OutlineRequest) -> OutlineResponse:
    # 1) High-level outline
    outline_schema = AdventureOutline.model_json_schema()
    outline_data = ai.generate_json_schema(
        schema_name="adventure_outline",
        schema=outline_schema,
        user_payload=build_outline_prompt(req),
        extra_instructions="Create the HIGH-LEVEL AdventureOutline JSON now.",
    )
    outline = AdventureOutline.model_validate(outline_data)

    # 2) Detailed outline (scenes + encounters + leveling)
    detailed_schema = DetailedAdventureOutline.model_json_schema()
    detailed_data = ai.generate_json_schema(
        schema_name="detailed_adventure_outline",
        schema=detailed_schema,
        user_payload=build_detailed_prompt(req, outline),
        extra_instructions="Now expand into a DETAILED Adventure outline with scenes, encounters, and level progression.",
    )
    detailed = DetailedAdventureOutline.model_validate(detailed_data)

    # Safety check: title consistency (optional guardrail)
    if detailed.outline_title.strip() == "":
        detailed.outline_title = outline.title

    return OutlineResponse(outline=outline, detailed=detailed)

# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI(title="D&D Adventure Builder", version="0.1.0")

_ai = AIEngine(api_key=os.environ.get("OPENAI_API_KEY", ""), model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))

@app.post("/v1/adventure/outline", response_model=OutlineResponse)
def create_adventure_outline(req: OutlineRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in environment.")

    try:
        return generate_outline_pair(_ai, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outline generation failed: {e}")
