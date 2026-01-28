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
    continuity_promises: List[str] = Field(
        default_factory=list,
        description="Facts that must remain true in detailed outline (always include this field)."
    )


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


def enforce_openai_strict_schema(schema: dict) -> dict:
    """
    OpenAI strict JSON schema requirements:
      - For every object schema:
          - additionalProperties must be false
          - required must include EVERY key in properties
    Pydantic's model_json_schema() may omit required for fields w/ defaults,
    so we patch it recursively.
    """
    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)

            if node.get("type") == "object":
                props = node.get("properties", {})
                if isinstance(props, dict) and props:
                    node["additionalProperties"] = False
                    node["required"] = sorted(list(props.keys()))

        elif isinstance(node, list):
            for item in node:
                walk(item)

    schema_copy = json.loads(json.dumps(schema))
    walk(schema_copy)
    return schema_copy


def _extract_output_text(resp) -> str:
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


def generate_json_schema(
    client: OpenAI,
    model: str,
    schema_name: str,
    schema: dict,
    user_payload: dict,
    extra_instructions: str,
) -> dict:
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
            "continuity_promises: include 5–12 bullet-style facts (always include the field, even if empty).",
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
    outline_schema = enforce_openai_strict_schema(AdventureOutline.model_json_schema())
    outline_data = generate_json_schema(
        client=client,
        model=model,
        schema_name="adventure_outline",
        schema=outline_schema,
        user_payload=build_outline_prompt(req),
        extra_instructions="Create the HIGH-LEVEL AdventureOutline JSON now.",
    )
    outline = AdventureOutline.model_validate(outline_data)

    detailed_schema = enforce_openai_strict_schema(DetailedAdventureOutline.model_json_schema())
    detailed_data = generate_json_schema(
        client=client,
        model=model,
        schema_name="detailed_adventure_outline",
        schema=detailed_schema,
        user_payload=build_detailed_prompt(req, outline),
        extra_instructions="Now expand into a DETAILED Adventure outline with scenes, encounters, and level progression.",
    )
    detailed = DetailedAdventureOutline.model_validate(detailed_data)

    if not detailed.outline_title.strip():
        detailed.outline_title = outline.title

    return OutlineResponse(outline=outline, detailed=detailed)


# =========================
# Demo Mode (no API calls)
# =========================

def demo_outline_response(req: OutlineRequest) -> OutlineResponse:
    outline = AdventureOutline(
        title="Demo: The Leyline Theft",
        logline=f"(Demo mode) A short adventure based on: {req.concept[:140]}",
        central_conflict="A hidden engine is siphoning ley energy, warping reality and drawing hostile incursions.",
        villain_or_antagonist="The Conduit Architect, a grief-twisted arcanist who believes the new world must be stabilized at any cost.",
        themes=[req.theme, req.tone, "mystery", "consequence"],
        hooks=[
            "A trading caravan vanishes along a route that now ‘loops’ back on itself.",
            "A village well reflects a different sky at night; those who stare too long dream of a door of bone and glass.",
        ],
        key_npcs=[
            KeyNPC(
                name="Elder Myrla Fen",
                role="Hamlet leader and guide",
                public_face="Practical, suspicious, deeply tired",
                secret="She’s been trading favors with the phenomenon to keep her people safe.",
                leverage="Knows a hidden path to the ruins and who first brought the ‘lens’ relic to town.",
            ),
            KeyNPC(
                name="Archivist Rell",
                role="Obsessive researcher",
                public_face="Eager, verbose, helpful",
                secret="Stole a focus crystal that worsened the siphon.",
                leverage="Can translate the Draconic lintel and identify the correct ‘alignment’ for the door puzzle.",
            ),
            KeyNPC(
                name="Silk-in-the-Mist",
                role="Fey emissary (ambiguous ally)",
                public_face="Playful and polite; never answers directly",
                secret="Wants the conduit left partially open to expand Fey influence.",
                leverage="Can grant safe passage if the party accepts a bargain—small now, costly later.",
            ),
        ],
        factions=[
            Faction(
                name="The Lantern Wardens",
                goal="Keep the road open and people alive",
                method="Curfews, patrols, controlled information",
                complication="They’re one bad night from turning into a paranoid mob.",
            ),
            Faction(
                name="The Mirror Court (Fey)",
                goal="Exploit the thinning veil for territory",
                method="Bargains, gifts, subtle replacements",
                complication="Their ‘help’ always changes someone.",
            ),
        ],
        beats=[
            StoryBeat(
                beat_id="B1",
                title="The Road that Repeats",
                purpose="Hook the party and establish reality glitches.",
                stakes="If they can’t break the loop, supplies and people won’t reach the region.",
                twist_or_reveal="The party finds their own footprints coming from the opposite direction.",
            ),
            StoryBeat(
                beat_id="B2",
                title="A Hamlet on Stilts",
                purpose="Learn lore and meet key NPCs.",
                stakes="If the hamlet collapses, the region loses its only refuge.",
                twist_or_reveal="The well shows a different constellation than the real sky.",
            ),
            StoryBeat(
                beat_id="B3",
                title="The Draconic Gate",
                purpose="Enter the primary site via a puzzle/skill challenge.",
                stakes="Failing draws a guardian and alerts the antagonist.",
                twist_or_reveal="The lintel names the Architect as a ‘mourner’ rather than a conqueror.",
            ),
            StoryBeat(
                beat_id="B4",
                title="The Conduit Chamber",
                purpose="Main set-piece confrontation.",
                stakes="If the siphon spikes, a planar tear opens permanently.",
                twist_or_reveal="The ‘enemy’ is stabilizing the newborn plane with stolen power.",
            ),
            StoryBeat(
                beat_id="B5",
                title="Choice at Dawn",
                purpose="Resolve with consequences and future hooks.",
                stakes="Who controls the relic shapes the region’s fate.",
                twist_or_reveal="A fey bargain can ‘fix’ things—by changing someone’s past.",
            ),
        ],
        continuity_promises=[
            "Reality glitches manifest as loops, mirrored skies, and repeated sounds.",
            "A focus crystal/lens is a key component to see or tune the ley flow.",
            "The hamlet depends on the party to keep the road open.",
            "The ruins are draconic/dragonborn in origin with readable sigils.",
            "A fey faction benefits from the veil staying thin.",
            "The antagonist is motivated by grief and believes they are ‘saving’ something.",
        ],
    )

    # Basic milestone progression respecting requested start/end levels
    mid_level = req.party_level_start
    if req.party_level_end > req.party_level_start:
        mid_level = min(req.party_level_start + 1, req.party_level_end)

    detailed = DetailedAdventureOutline(
        outline_title=outline.title,
        structure_notes=[
            "Demo mode: no API calls were made. This content is prebuilt.",
            "Includes combat and non-combat encounters plus milestone leveling.",
        ],
        scenes=[
            Scene(
                scene_id="S1",
                title="The Road that Repeats",
                location="A fog-choked causeway where landmarks reappear",
                goal="Break the travel loop and find evidence of the missing caravan",
                boxed_text="Fog presses close. Your lanternlight seems to lag behind your movements, like it’s remembering you rather than following you.",
                obstacles=["Disorienting loop", "False trail markers", "Echoing footsteps that aren’t yours"],
                encounters=[
                    Encounter(
                        encounter_id="E1",
                        type="skill_challenge",
                        difficulty="medium",
                        summary="Navigate the loop by triangulating repeating signs and anchoring reality with a clever method.",
                        win_condition="Accumulate enough successes to identify the ‘fixed’ landmark and exit the loop.",
                        fail_forward="You exit, but lose time and arrive as something begins stalking the hamlet’s outskirts.",
                        setup=["Let players propose skills; reward clever anchors (chalk marks, rope, rhythmic counting)."],
                        scaling_notes=["If the party struggles, provide a clue from a half-faded draconic rune stone."],
                    )
                ],
                clues_and_info=["A caravan token stamped with a lantern sigil", "Iridescent pollen (fey sign)"],
                rewards=["Supplies cache (rations, rope, lamp oil)", "Map scrap pointing to an old draconic road gate"],
                consequences=["Failing increases tension in the hamlet; Wardens impose stricter curfews."],
                links_to_beats=["B1"],
                estimated_minutes=45,
            ),
            Scene(
                scene_id="S2",
                title="A Hamlet on Stilts",
                location="A stilted hamlet above black water",
                goal="Gain trust, learn what changed, and identify the relic’s trail",
                boxed_text="Homes balance on stilts over dark water. Wind chimes click with unsettling regularity—like a metronome for the fog.",
                obstacles=["Suspicious locals", "Conflicting rumors", "A ‘helpful’ stranger asking too many questions"],
                encounters=[
                    Encounter(
                        encounter_id="E2",
                        type="social",
                        difficulty="medium",
                        summary="Win the Wardens’ trust and convince Elder Myrla to reveal the hidden route.",
                        win_condition="Earn an invitation to a private meeting and access to the old gate key.",
                        fail_forward="You get the key, but the Wardens shadow you and may intervene at a bad moment.",
                        setup=["Present 3 rumors (true/half/false). Let players test them in conversation."],
                        scaling_notes=["If stuck, Archivist Rell blurts out a crucial fact to ‘help’."],
                    )
                ],
                clues_and_info=["The well reflects a different sky at night", "Rell mentions a ‘focus lens’ that ‘tunes the air’"],
                rewards=["Old gate key", "Local ally (one Warden)"],
                consequences=["If players antagonize locals, resource prices rise and help disappears."],
                links_to_beats=["B2"],
                estimated_minutes=50,
            ),
            Scene(
                scene_id="S3",
                title="The Draconic Gate",
                location="Collapsed draconic stonework swallowed by roots",
                goal="Open the sealed entry without triggering a full alarm",
                boxed_text="Ancient scales carved in stone stare down. Their eyes—dark glass—drink your lanternlight until it feels colder.",
                obstacles=["Sealed relief door", "Misaligned ‘scale’ panels", "Ambient magic that distorts sound"],
                encounters=[
                    Encounter(
                        encounter_id="E3",
                        type="puzzle",
                        difficulty="medium",
                        summary="Align draconic relief scales to match the ‘true’ constellation seen in the hamlet well.",
                        win_condition="Door opens quietly; you keep the initiative later.",
                        fail_forward="Door opens loudly; a guardian is active deeper within and time pressure increases.",
                        setup=["Give 3 hints: well-constellation sketch, rune-stone pattern, and a ‘missing scale’ clue."],
                        scaling_notes=["Allow Arcana/History checks to reveal the intended sequence if needed."],
                    )
                ],
                clues_and_info=["Lintel names the Architect as ‘Mourner’", "Ley hum intensifies past the threshold"],
                rewards=["Safe entry", "Foreshadow note about grief and ‘stabilizing’"],
                consequences=["Noisy entry makes later combat harder (reinforcements or worse terrain)."],
                links_to_beats=["B3"],
                estimated_minutes=45,
            ),
            Scene(
                scene_id="S4",
                title="The Conduit Chamber",
                location="A circular chamber with a central shaft and broken platforms",
                goal="Stop or redirect the siphon before a tear stabilizes",
                boxed_text="A shaft drops into darkness. The air tastes like lightning. Below, a slow pulse paints the stone with borrowed starlight.",
                obstacles=["Vertical terrain", "Unstable platforms", "Siphon pulse countdown"],
                encounters=[
                    Encounter(
                        encounter_id="E4",
                        type="combat",
                        difficulty="hard",
                        summary="Fight a guardian while the conduit pulses each round, shifting platforms and opening brief rifts.",
                        win_condition="Defeat or disable the guardian and disrupt the focus crystal’s alignment.",
                        fail_forward="You survive but the tear partially stabilizes; a future incursion is guaranteed.",
                        setup=["Use a 3-stage countdown. Each stage changes terrain (tilt platforms, rift hazards)."],
                        scaling_notes=["Scale down by reducing hazards; scale up by adding rift-spawned minions."],
                    )
                ],
                clues_and_info=["The conduit can stabilize the newborn plane—but at a cost", "The lens is a tuning key, not just treasure"],
                rewards=["Focus lens relic", "Blueprint-like etchings of ley routes"],
                consequences=["If tear stabilizes, the region changes: more fey/planar bleed, new rules at night."],
                links_to_beats=["B4"],
                estimated_minutes=60,
            ),
            Scene(
                scene_id="S5",
                title="Choice at Dawn",
                location="Ruin exit overlooking the hamlet and the fog line",
                goal="Choose what to do with the lens and how to handle the factions",
                boxed_text="The fog thins. For the first time you hear the water—just water—like the world is holding its breath.",
                obstacles=["Conflicting claims", "The ‘easy fix’ bargain", "Long-term consequences"],
                encounters=[
                    Encounter(
                        encounter_id="E5",
                        type="social",
                        difficulty="medium",
                        summary="Negotiate between the Wardens, Archivist Rell, and Silk-in-the-Mist about the lens.",
                        win_condition="Choose a path and secure allies for the fallout.",
                        fail_forward="You choose anyway, but you make an enemy who acts immediately.",
                        setup=["Give 2–3 offers with clear costs/benefits. Make the bargain tempting but specific."],
                        scaling_notes=["If table wants action, turn this into a chase or tense standoff instead."],
                    )
                ],
                clues_and_info=["The antagonist’s goal is not simple evil", "This is the first node of a larger ley network"],
                rewards=["Reputation and a faction ally", "A lead to the next anomaly site"],
                consequences=["Your choice determines how thin the veil remains and who hunts you next."],
                links_to_beats=["B5"],
                estimated_minutes=45,
            ),
        ],
        level_progression=[
            LevelProgressionStep(
                step_id="L1",
                after_scene_id="S2",
                level=mid_level,
                rationale="Milestone: uncover the core mystery and secure access to the draconic gate.",
                optional_side_objectives=["Help the Wardens with a nighttime watch to earn an extra resource."],
            ),
            LevelProgressionStep(
                step_id="L2",
                after_scene_id="S4",
                level=req.party_level_end,
                rationale="Milestone: confront the conduit chamber and claim the lens relic—major arc completion.",
                optional_side_objectives=["Stabilize the tear completely with a risky alignment ritual (adds future complications)."],
            ),
        ],
        optional_side_quests=[
            "Night Watch: identify what’s stalking the hamlet’s outskirts (no combat required; can be a scare + clue trail).",
            "The Well-Sky: record the false constellation and learn who first noticed it.",
        ],
        recap_questions=[
            "Which faction did you trust the most, and why?",
            "What did the lens feel like when you held it (cold, warm, alive, whispering)?",
            "What consequence are you willing to accept to keep the world stable?",
        ],
    )
    return OutlineResponse(outline=outline, detailed=detailed)


def is_quota_error(e: Exception) -> bool:
    msg = str(e)
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="OdysseyMaker — Adventure Outline", layout="wide")
st.title("OdysseyMaker — D&D Adventure Outline Generator")

# Prefer Streamlit secrets; fall back to env var
api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
api_key = api_key or os.environ.get("OPENAI_API_KEY")

# Demo mode is allowed even without an API key
client = OpenAI(api_key=api_key) if api_key else None

with st.sidebar:
    st.header("Generation Settings")

    demo_mode = st.checkbox("Demo mode (no API calls)", value=False)
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

if not api_key and not demo_mode:
    st.warning("No OPENAI_API_KEY found. Turn on Demo mode or set the key in Streamlit Secrets.")
    st.stop()

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

# =========================
# Generation trigger
# =========================

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

    with st.spinner("Generating outline..."):
        try:
            if demo_mode:
                result = demo_outline_response(req)
            else:
                # Real generation
                result = generate_outline_pair(client, model, req)  # type: ignore

            st.session_state["outline_result"] = result.model_dump()
            st.session_state.pop("last_error", None)

        except Exception as e:
            if is_quota_error(e):
                st.warning("API quota exceeded for this key. Switching to Demo mode output.")
                result = demo_outline_response(req)
                st.session_state["outline_result"] = result.model_dump()
                st.session_state.pop("last_error", None)
            else:
                st.session_state["last_error"] = str(e)
                st.error(f"Generation failed: {e}")

if st.session_state.get("last_error"):
    st.warning("Last error:")
    st.code(st.session_state["last_error"])

# =========================
# Render results
# =========================

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
