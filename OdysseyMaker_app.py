import json
import os
from typing import List, Literal

import streamlit as st
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI

# =========================
# Domain Schemas (contract)
# =========================

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
    constraints: List[str] = Field(default_factory=list, description="Hard constraints (one per line in UI)")
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
    continuity_promises: List[str] = Field(default_factory=list, description="Facts that must remain true in detailed outline")


class Encounter(BaseModel):
    encounter_id: str  # E1, E2...
    type: EncounterType
    difficulty: Difficulty
    summary: str
    win_condition: str
    fail_forward: str
    setup: List[str] = Field(default_factory=list)
    scaling_notes: List[str] = Field(default_factory=list)


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
    links_to_beats: List[str] = Field(default_factory=list)
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
    recap_questions: List[str] = Field(default_factory=list)


class OutlineResponse(BaseModel):
    outline: AdventureOutline
    detailed: DetailedAdventureOutline


# =========================
# OpenAI helpers
# =========================

SYSTEM = """You are a veteran tabletop RPG adventure designer.
You output ONLY valid JSON that conforms exactly to the given schema.
Be concrete and playable: specific names, locations, motivations, clear objectives.
Include BOTH combat and non-combat encounters.
Avoid copyrighted setting text; create original material.
If ruleset is 'system_agnostic', avoid 5e jargon (CR, XP tables), but keep difficulty labels.
Never output markdown, only JSON.
"""


def _extract_output_text(resp) -> str:
    """
    OpenAI Responses API returns a list in resp.output, containing message items with content blocks.
    We extract the first output_text block.
    """
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


def generate_json_schema(client: OpenAI, model: str, schema_name: str, schema: dict, user_payload: dict, extra_instructions: str) -> dict:
    resp = client.responses.create(
        model=model,
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
    return json.loads(_extract_output_text(resp))


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
            "Across the whole adventure: include at least 1 combat encounter and at least 1 non-combat encounter.",
            "Encounters must include win_condition and fail_forward (never dead-end the adventure).",
            "Level progression must be structured and sensible from party_level_start to party_level_end.",
            "If leveling_mode is milestone, place level-ups after key scenes. If xp, describe XP targets in rationale text (no big tables).",
            "Estimated_minutes should roughly match session_count_target (scenes usually 30–75 minutes).",
            "Honor continuity_promises exactly unless a constraint forces a change (then explain in structure_notes).",
        ],
    }


def generate_outline_pair(client: OpenAI, model: str, req: OutlineRequest) -> OutlineResponse:
    outline_data = generate_json_schema(
        client=client,
        model=model,
        schema_name="adventure_outline",
        schema=AdventureOutline.model_json_schema(),
        user_payload=build_outline_prompt(req),
        extra_instructions="Create the HIGH-LEVEL AdventureOutline JSON now.",
    )
    outline = AdventureOutline.model_validate(outline_data)

    detailed_data = generate_json_schema(
        client=client,
        model=model,
        schema_name="detailed_adventure_outline",
        schema=DetailedAdventureOutline.model_json_schema(),
        user_payload=build_detailed_prompt(req, outline),
        extra_instructions="Now expand into a DETAILED Adventure outline with scenes, encounters, and level progression.",
    )
    detailed = DetailedAdventureOutline.model_validate(detailed_data)

    # Ensure a title is present
    if not detailed.outline_title.strip():
        detailed.outline_title = outline.title

    return OutlineResponse(outline=outline, detailed=detailed)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="OdysseyMaker — Adventure Outline", layout="wide")
st.title("OdysseyMaker — D&D Adventure Outline Generator")

# Prefer Streamlit secrets; fall back to env var
api_key = None
if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("Missing OPENAI_API_KEY. Set it in Streamlit Secrets (recommended) or as an environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)

with st.sidebar:
    st.header("Generation Settings")

    model = st.text_input("Model", value=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    ruleset = st.selectbox("Ruleset", ["5e", "system_agnostic"], index=0)
    leveling_mode = st.selectbox("Leveling mode", ["milestone", "xp"], index=0)
    tone = st.selectbox("Tone", ["mystery", "heroic", "grimdark", "whimsical", "horror", "epic"], index=0)
    theme = st.selectbox("Theme", ["dungeon", "fey", "undead", "dragons", "political", "heist", "cosmic", "wilderness", "urban"], index=0)

    colA, colB = st.columns(2)
    with colA:
        lvl_start = st.number_input("Start level", min_value=1, max_value=20, value=3)
    with colB:
        lvl_end = st.number_input("End level", min_value=1, max_value=20, value=5)

    sessions = st.number_input("Target sessions", min_value=1, max_value=20, value=3)
    include_travel = st.checkbox("Include travel", value=True)

    st.caption("Constraints (one per line):")
    constraints_text = st.text_area(
        " ",
        value="include at least one social scene\ninclude a puzzle that foreshadows the villain",
        height=90,
    )

concept = st.text_area(
    "General concept / pitch",
    value="A grief-stricken wizard is siphoning ley lines into a newborn material plane, causing planar incursions. The party must trace anomalies through a fey-touched forest toward ruins of an ancient dragonborn city.",
    height=140,
)

colX, colY = st.columns([1, 1])
with colX:
    generate_btn = st.button("Generate outline", type="primary")
with colY:
    clear_btn = st.button("Clear results")

if clear_btn:
    st.session_state.pop("outline_result", None)
    st.session_state.pop("last_error", None)

if generate_btn:
    constraints = [c.strip() for c in constraints_text.splitlines() if c.strip()]
    try:
        req = OutlineRequest(
            concept=concept,
            party_level_start=int(lvl_start),
            party_level_end=int(lvl_end),
            ruleset=ruleset,  # type: ignore
            leveling_mode=leveling_mode,  # type: ignore
            tone=tone,  # type: ignore
            theme=theme,  # type: ignore
            session_count_target=int(sessions),
            constraints=constraints,
            include_travel=include_travel,
        )
    except ValidationError as ve:
        st.session_state["last_error"] = str(ve)
        st.error("Invalid input. Check the fields in the sidebar.")
        st.code(str(ve))
        st.stop()

    with st.spinner("Generating high-level outline and detailed outline..."):
        try:
            result = generate_outline_pair(client, model, req)
            st.session_state["outline_result"] = result.model_dump()
            st.session_state.pop("last_error", None)
        except Exception as e:
            st.session_state["last_error"] = str(e)
            st.error(f"Generation failed: {e}")

if "last_error" in st.session_state and st.session_state["last_error"]:
    st.warning("Last error:")
    st.code(st.session_state["last_error"])

data = st.session_state.get("outline_result")
if data:
    result = OutlineResponse.model_validate(data)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("High-level Story Outline")
        st.markdown(f"### {result.outline.title}")
        st.write(result.outline.logline)
        st.markdown(f"**Central conflict:** {result.outline.central_conflict}")
        st.markdown(f"**Antagonist:** {result.outline.villain_or_antagonist}")

        st.markdown("#### Hooks")
        for h in result.outline.hooks:
            st.write(f"- {h}")

        st.markdown("#### Key NPCs")
        for npc in result.outline.key_npcs:
            with st.expander(npc.name):
                st.write(f"**Role:** {npc.role}")
                st.write(f"**Public face:** {npc.public_face}")
                st.write(f"**Secret:** {npc.secret}")
                st.write(f"**Leverage:** {npc.leverage}")

        st.markdown("#### Factions")
        for f in result.outline.factions:
            with st.expander(f.name):
                st.write(f"**Goal:** {f.goal}")
                st.write(f"**Method:** {f.method}")
                st.write(f"**Complication:** {f.complication}")

        st.markdown("#### Beats")
        for b in result.outline.beats:
            with st.expander(f"{b.beat_id}: {b.title}"):
                st.write(f"**Purpose:** {b.purpose}")
                st.write(f"**Stakes:** {b.stakes}")
                st.write(f"**Twist/Revelation:** {b.twist_or_reveal}")

        st.markdown("#### Continuity promises")
        for p in result.outline.continuity_promises:
            st.write(f"- {p}")

    with right:
        st.subheader("Detailed Outline (Scenes + Encounters + Leveling)")

        st.markdown("#### Structure notes")
        for n in result.detailed.structure_notes:
            st.write(f"- {n}")

        st.markdown("#### Scenes")
        for s in result.detailed.scenes:
            with st.expander(f"{s.scene_id}: {s.title} ({s.estimated_minutes} min) — {s.location}"):
                st.write(f"**Goal:** {s.goal}")
                st.markdown("**Boxed text:**")
                st.write(s.boxed_text)

                st.markdown("**Obstacles:**")
                for o in s.obstacles:
                    st.write(f"- {o}")

                st.markdown("**Encounters:**")
                for e in s.encounters:
                    st.write(f"- **{e.encounter_id} [{e.type} | {e.difficulty}]** — {e.summary}")
                    st.write(f"  - Win: {e.win_condition}")
                    st.write(f"  - Fail forward: {e.fail_forward}")
                    if e.setup:
                        st.write("  - Setup:")
                        for x in e.setup:
                            st.write(f"    - {x}")
                    if e.scaling_notes:
                        st.write("  - Scaling:")
                        for x in e.scaling_notes:
                            st.write(f"    - {x}")

                st.markdown("**Clues & info:**")
                for c in s.clues_and_info:
                    st.write(f"- {c}")

                st.markdown("**Rewards:**")
                for r in s.rewards:
                    st.write(f"- {r}")

                st.markdown("**Consequences:**")
                for c in s.consequences:
                    st.write(f"- {c}")

                if s.links_to_beats:
                    st.caption("Links to beats: " + ", ".join(s.links_to_beats))

        st.markdown("#### Level progression")
        for lp in result.detailed.level_progression:
            st.write(f"- **{lp.step_id}**: After **{lp.after_scene_id}** → Level **{lp.level}**")
            st.caption(lp.rationale)
            if lp.optional_side_objectives:
                st.caption("Optional: " + "; ".join(lp.optional_side_objectives))

        if result.detailed.optional_side_quests:
            st.markdown("#### Optional side quests")
            for q in result.detailed.optional_side_quests:
                st.write(f"- {q}")

        if result.detailed.recap_questions:
            st.markdown("#### Recap questions")
            for q in result.detailed.recap_questions:
                st.write(f"- {q}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download full JSON",
            data=json.dumps(result.model_dump(), indent=2),
            file_name="adventure_outline.json",
            mime="application/json",
        )
    with col2:
        st.download_button(
            "Download high-level outline JSON only",
            data=json.dumps(result.outline.model_dump(), indent=2),
            file_name="adventure_outline_high_level.json",
            mime="application/json",
        )
