# How the VGC-LLM Training Pipeline Works

A step-by-step walkthrough of the data → SFT pipeline, grounded in real data
from a Bo3 series in our cached corpus:

> **Running example:** [bo3-gen9vgc2026regibo3-2568132152](data_scraper/data/replays/gen9vgc2026regibo3/) —
> *The Yegorushhka* vs *jonen*, Game 1 of 2.

We'll trace what happens to this match as it flows through the four
components, and call out *why* each design choice is shaped the way it is.

---

## The shape of the system

```
Pokémon Showdown
      │  scrape replay JSONs
      ▼
data_scraper/   ──►  16,537 raw replay logs on disk
      │
      │  POST /parse_log per game
      ▼
calc_microservice/  ──►  per-turn snapshots + actionLog (Node side)
      │
      │  read JSONL, run inference + LLM
      ▼
pipeline/master_pipeline.py
      │  per turn:  threat matrix  →  teacher LLM  →  knowledge update
      ▼
sft_training_data.jsonl   ──►  one fine-tuning row per turn
```

Two languages, four components, one direction of data flow. Everything
crosses the language boundary via HTTP.

---

## Part 1: Getting the data

### Step 1 — Scrape the high-Elo replays

The `data_scraper` walks two ladder endpoints
(`pokemonshowdown.com/ladder/{format}.json`), takes the top-500 users on
each, and downloads every public replay they've saved.

**What we got:** 16,537 unique replays across both formats, ~140 MB on disk.

**The first surprising number:** only ~50% of the top-500 in regi (and ~30% in
regibo3) actually save replays publicly. Our corpus is biased toward
"high-Elo replay-savers", not strictly toward the highest Elo. Worth
remembering — when we look at training-data distribution we're sampling a
specific behavior, not pure rating.

### Step 2 — Why a Node microservice?

Showdown's protocol is text — a stream of `|move|p1a: foo|...|p2b: bar` lines.
**We tried parsing it in Python with regex** and gave up almost immediately.
The reasons are real game mechanics, not laziness:

- **Zoroark Illusion**: a Zoroark on the field appears under the *previous*
  Pokémon's identity. The protocol switches to the real identity mid-battle
  when the illusion breaks. Tracking who-is-actually-who across turns
  requires a full state machine.
- **End-of-turn ordering**: weather damage, status damage, Leftovers,
  ability proc — all happen in a specific order that depends on Speed
  stats. Damage attribution is non-trivial.
- **Forme changes**: Mega Evolution, Tera, Aegislash blade/shield, etc.
- **Multi-hit, spread, redirection, sub-targets**, …

Smogon publishes the official TS state machine (`@pkmn/client`). It costs
us one Node service to use it instead of reinventing it.

The service exposes three endpoints:

| Endpoint | Backed by | Purpose |
|---|---|---|
| `POST /calc` | `@smogon/calc` | Damage range for one (attacker, move, defender, field). Accepts `isCrit`. |
| `POST /parse_log` | `@pkmn/protocol` + `@pkmn/client` | Raw log → per-turn snapshots + per-turn `actionLog` (damage events). |
| `GET /dex/move/:name` | `@pkmn/dex` | Move metadata (category, type, base power). Used to skip Status moves. |

---

## Part 2: From log to structured battle states

### Step 3 — Parse and stitch

`pipeline/replay_parser.py` POSTs each replay's `log` field to `/parse_log`,
gets back a snapshot per turn, and stitches Bo3 series. **Bo3 stitching
rule:** group games by sorted player-pair, sort by upload time, split into
series whenever consecutive games are >30 min apart **or** the current
series already has 3 games (the Bo3 ceiling — back-to-back matches between
the same players otherwise glue together).

**Final corpus shape:** 13,919 matches: 10,997 single-game (Bo1) + 2,922
multi-game (Bo3 series of 1–3 games).

Each snapshot looks like:

```json
{
  "turn": 1,
  "field": { "weather": null, "terrain": null, "tailwindP1": false, "tailwindP2": false },
  "p1": {
    "active": [
      { "slot": "a", "species": "Miraidon", "hpPercent": 100,
        "ability": "hadronengine", "item": null, "revealedMoves": [], ... },
      { "slot": "b", "species": "Iron Valiant", ... }
    ],
    "bench": [ {"species": "Lunala", "fainted": false}, ... ]
  },
  "p2": { ... },
  "actionLog": [ ... events that happen DURING this turn ... ]
}
```

**Key design choice:** `actionLog` is *forward-looking*. Events for turn N
attach to `snapshot[N].actionLog`, so we always pair `(snap[N], snap[N+1],
snap[N].actionLog)` for inference and label extraction.

Real turn 1 actionLog from the running example:

```json
[
  {"attacker_slot": "p1a", "defender_slot": "p2a", "move_name": "Volt Switch",
   "hp_before_pct": 100, "hp_after_pct": 79, "is_crit": false, "is_ko": false},
  {"attacker_slot": "p2a", "defender_slot": "p1b", "move_name": "Moongeist Beam",
   "hp_before_pct": 100, "hp_after_pct": 1, "is_crit": false, "is_ko": false}
]
```

Two events. Miraidon hit Lunala for 21%. Lunala hit Iron Valiant for 99%.
Notice that turn 1 had Iron Valiant doing *something else* too — but no
damage event for `p1b`. That something is detectable from the next snapshot
(it'll show `quickguard` in Iron Valiant's `revealedMoves`).

---

## Part 3: The intelligence layer

### Step 4 — Reconstruct P1's team (the "OTS we don't have")

In live VGC, Open Team Sheet means we know the opponent's species, items,
abilities, and Tera types from turn 1. But our offline replays **don't
expose OTS data** — we only see what gets revealed during play (an item
when it activates, a move when it's used, a Tera when it triggers).

We *do* however know the full Bo3 series in advance — we've seen all the
turns. So we **forward-scan** every snapshot in the match and aggregate
everything that was ever revealed about each P1 species:

- Item (from the first snapshot it surfaces)
- Ability (from the first activation)
- Tera type (from the team-preview line or the actual Terastallization)
- All moves used across all games

For our running example, after scanning game 1, we'd have:
```
Miraidon: ability=hadronengine, item=?, tera=?, moves=[Volt Switch, Electro Drift, Discharge, ...]
Iron Valiant: ability=quarkdrive, item=?, tera=?, moves=[Quick Guard, Encore, ...]
Lunala: ability=shadowshield, ..., moves=[Moongeist Beam, ...]
Incineroar: ability=intimidate, ..., moves=[...]
Ursaluna: ..., moves=[...]
```

**Why we pad to exactly 4 moves with `"[UNREVEALED_MOVE]"`:** a Pokémon's
revealed moves are exactly the moves the human chose to USE. If a Pokémon
has only 2 revealed moves across an entire 3-game Bo3, that means moves 3
and 4 were judged worse than even Encore/Quick Guard for every situation
this match presented. We tell the LLM (in the system prompt's *Masking
Rule*):

> *"If a Pokémon on Your Side has `[UNREVEALED_MOVE]` in its moveset, it
> means that move was never utilized by the human expert in this entire
> Bo3 series. You must assume that the unrevealed move was completely
> suboptimal, irrelevant, or unusable for this specific matchup. Do not
> attempt to guess what it is, and do not factor it into your strategic
> reasoning."*

The Masking Rule is what lets us train without leaking team-builder
guesses into the model's reasoning.

### Step 5 — KnowledgeState: solving the EV puzzle frame by frame

This is the most technically interesting piece. The problem:

> **We don't know either side's EVs/IVs/Natures.** All we have is observed
> damage. How do we figure out what the spreads are?

Naïve answer: "Rillaboom did 60% with Wood Hammer, so Calyrex-Ice has X
HP+Def." But **the attacker's EVs are also unknown**. Was that 60% from a
0-Atk Rillaboom on max-bulk Calyrex-Ice? Or a 252-Atk Rillaboom on
zero-bulk Calyrex-Ice? Both are mathematically consistent with "60%
damage" — they're just different points on the joint (Atk × Def × HP)
surface. One observation, three unknowns.

**Our solution: dual-state interval arithmetic + binary search.**

We maintain two `KnowledgeState`s (one per side), each species mapping to
`{min_evs, max_evs}` per stat in `[0, 252]`. We start fully open. Each
damage event runs **six binary searches** against the calc service:

- Defender: `min_def`, `max_def`, `min_hp`, `max_hp`
- Attacker: `min_off`, `max_off` (Atk if physical, SpA if special)

The trick is **how we hold the OTHER side's bounds during each search**:

| Search | Held at … | Why |
|---|---|---|
| Defender's `min_def` | Attacker at `min_off`, defender HP at `max_hp` | Hardest case for defender to need any defense at all |
| Defender's `max_def` | Attacker at `max_off`, defender HP at `min_hp` | Hardest case for defender to be allowed lots of defense |
| Attacker's `min_off` | Defender at `min_hp + min_def` | Easiest case for low offense to still produce observed damage |
| Attacker's `max_off` | Defender at `max_hp + max_def` | Easiest case for high offense to not exceed observed damage |

(Each "hardest case for X" pair holds the OTHER side at its current
*least restrictive* edge — interval arithmetic — so we never over-tighten
because of cross-side coupling.)

All six searches use **pre-update bounds** and apply atomically at the
end. Order-independent.

**Worked example** (real numbers from our integration test):

> Rillaboom (p2a) Wood Hammers Calyrex-Ice (p1a). Observed 60% damage.
> Both sides start at fully-open `[0, 252]` bounds.

After this single event:
- **Defender** Calyrex-Ice: `max_hp` 252→**75**, `max_def` 252→**67**.
  (Calyrex-Ice can't be much bulkier — even max-Atk Rillaboom on max-bulk
  Calyrex-Ice wouldn't get to 60%.)
- **Attacker** Rillaboom: `min_atk` 0→**188**.
  (Rillaboom must have invested at least 188 EVs to deal 60% to even
  zero-bulk Calyrex-Ice.)

Three numbers learned from one observation, on both sides. That's the
two-way ambiguity payoff.

**Then the 508-EV constraint pass kicks in.** A Pokémon has only 508 usable
EVs total. So once we've proven `min_atk ≥ 188`, the room for HP/Def/SpA/
SpD/Spe collapses: each of those stats can be at most `508 − 188 = 320`,
already pre-clamped to 252. But the example that pays off most:

> Suppose later observations prove `min_atk ≥ 252` AND `min_spe ≥ 252`
> (504 EVs locked in). The constraint immediately crushes max for HP, Def,
> SpA, SpD all to 4. The HP/Def coupling that the binary search alone
> converges on slowly is solved in a single arithmetic pass.

KnowledgeStates **persist across turns and across games** in a Bo3 series.
By turn 5 of game 2, the bounds on commonly-used Pokémon are usually
tight enough that the threat matrix becomes meaningfully decisive.

### Step 6 — Threat matrix: dual-track damage envelope

For each turn, we render the damage landscape between the active
Pokémon. Two damage tracks per matchup-move:

- **Absolute** — strict math from KnowledgeState bounds. Wide, but
  provable. (At turn 1, it's basically `[0, 252]` everywhere → ranges
  like 18%–89% — useless for decisions.)
- **Probable (meta)** — single calc result assuming both Pokémon run
  their canonical Smogon meta spread (from `canonical_priors`). Narrow,
  fast — but only as good as the prior.

When the canonical prior falls *outside* the proven KnowledgeState bounds
on any relevant stat, we append `[PRIOR CONTRADICTED]`. That's the
LLM's signal: *"this opponent is off-meta, lean on the Absolute envelope."*

Real example output (turn 4 of a different match):

```
[opp Rillaboom] vs [us Calyrex-Ice]
  woodhammer  Absolute: 45.7%–62.3%  |  Probable (meta): 48.3%–57.0%  (42.6% chance to 2HKO)  [PRIOR CONTRADICTED]
[us Farigiraf] vs [opp Rillaboom]  (atk_boosts={'atk': -1})
  foulplay   Absolute: 17.9%–32.6%  |  Probable (meta): 33.1%–39.4%  (22.3% chance to 3HKO after Grassy Terrain recovery)
```

Volatile state — Atk-1 from Intimidate, Grassy Terrain recovery, Tera —
threaded through every payload. The LLM sees the actual board, not a
sterile lab calc.

### Step 7 — Canonical priors: what does "meta" mean?

The Probable track is only useful if "the canonical spread" is real. We
fetch monthly Smogon usage stats:

```
https://www.smogon.com/stats/{YYYY-MM}/chaos/{format_id}-0.json
```

…walk back month by month until we find a 200, save to disk, and look up
the per-species `Spreads` dict at runtime. Take the single most-used
spread (`"Nature:hp/atk/def/spa/spd/spe"`), parse it, return.

**A non-obvious example from our 2026-04 cache:**

> `Iron Hands` → **Brave 76 / 180 / 12 / 0 / 236 / 0**

That's a Trick Room set (Brave nature = -Spe, 0 Spe EVs). NOT the obvious
"max Atk / max HP" you'd get from a base-stat heuristic. This is signal
you can't synthesize — it requires actual usage data.

For the running example's Iron Valiant: Smogon's most-used is **Jolly
0/252/0/0/4/248** (a physical sweeper). But our human player's Iron
Valiant used Quick Guard — pointing at a *support* build, not the standard
sweeper. That's exactly the kind of off-meta call the
`[PRIOR CONTRADICTED]` flag will eventually fire on once enough evidence
builds up.

---

## Part 4: Synthesizing the SFT dataset

### Step 8 — Action extraction (the ground-truth label)

For each turn, we need to know what each P1 active slot DID. That's the
training label. We reverse-engineer it from `snap_pre`, `snap_post`, and
`snap_pre.actionLog`:

| Signal | Action type |
|---|---|
| `attacker_slot == "p1a"` in actionLog | move (target = single defender_slot, or `"spread"` if multiple) |
| `isTerastallized` flips false→true at this slot | tera flag set on the move |
| Species at slot changes between pre and post | switch (`switch_to = post.species`) |
| Same species, exactly 1 new revealed move, no damage event | status move (target = `"self"`) |
| None of the above | **ambiguous → skip turn** |

Real example, our running game 1 turn 1:

- **slot a (Miraidon)**: `attacker_slot="p1a"` appears in actionLog → move.
  Single defender_slot=`"p2a"` → `target="p2a"`. No Tera. →
  `{"action_type": "move", "move": "Volt Switch", "target": "p2a", "tera": false}`.
- **slot b (Iron Valiant)**: no actionLog entry. Species same as turn 2
  start. `revealedMoves` diff: post has `["quickguard"]`, pre had `[]`.
  Exactly 1 new move. → `{"action_type": "move", "move": "quickguard",
  "target": "self", "tera": false}`.

**We're conservative:** if any slot is ambiguous, the WHOLE turn is
skipped. Bad labels poison SFT — better to lose 30% of turns than label
30% of them wrong.

(In our 1-match dry run: 5 turns yielded SFT examples, 3 skipped as
ambiguous.)

### Step 9 — Teacher LLM: reverse-engineer the pro's play

Now the keystone insight that makes the whole pipeline work:

> **The teacher LLM doesn't have to FIND the optimal play. It has to
> JUSTIFY a known good play.**

Finding the optimal VGC play requires Day-2-Worlds intuition. GPT-4o
doesn't have that. But *rationalizing* a play it's been told is correct
— given the board state and a precomputed threat matrix — is a much
easier task. We get high-quality CoT essentially for free from a model
that doesn't itself know VGC at pro level.

**Synthesis call shape:**

```
system: "You are a world-class VGC competitor… [Masking Rule]… [Output schema]…"

user:   [board state: turn N, field, P1/P2 active+bench]
        [threat matrix: dual-track Absolute + Probable per matchup-move]
        ────── EXPERT'S DECISION (oracle truth) ──────
        { "slot_1": {"action_type": "move", "move": "Volt Switch", ...},
          "slot_2": {"action_type": "move", "move": "quickguard", ...} }

assistant: [CoT explaining why this play is correct]
           [optional calculate_damage tool call to verify a critical assumption]
tool:      [calc result]
assistant: { "pre_tool_thought": "Volt Switch lets us pivot Miraidon out of
              the incoming Moongeist Beam from Lunala while still chipping
              p2a, and Quick Guard from Iron Valiant blocks any priority
              follow-up on slot a as it switches in.",
              "action": { "slot_1": {...}, "slot_2": {...} } }
```

**Two things to notice:**

1. **The `=== EXPERT'S DECISION ===` suffix is stripped before saving.**
   The trained model never sees the ground truth in its prompt. It learns
   to produce the same chain of reasoning *without* the cheat — that's the
   whole SFT magic.

2. **The `calculate_damage` tool is the critical-call escape hatch.** The
   teacher LLM can verify any specific claim ("does Tera-Fairy Calyrex
   actually OHKO Rillaboom with this build?") before committing. The
   calc service handles `isCrit`, `isTera`, `boosts`, `status`, weather,
   terrain — all the volatile state the LLM might want to vary.

Final output is constrained by OpenAI's `response_format: json_schema`
(strict mode) to the `{ pre_tool_thought, action: {slot_1, slot_2} }` shape.

### Step 10 — Write the SFT JSONL row, update knowledge, repeat

Each successful turn writes one row:

```json
{
  "match_id": "bo3-gen9vgc2026regibo3-2568132152",
  "game_index": 0,
  "turn": 1,
  "format_id": "gen9vgc2026regibo3",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},          ← state + threat matrix only (no ground truth)
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "{...JSON...}"}
  ]
}
```

After the row is written, we feed the same `actionLog` to
`damage_inferencer.update_knowledge` — tightening both KnowledgeStates so
the *next* turn's threat matrix is sharper. Bo3 series compound this:
by turn 5 of game 2, the bounds are usually meaningfully tight.

The whole loop is **resumable**. `(match_id, game_index, turn)` keys
already in the output JSONL are skipped on rerun — useful when the
OpenAI run inevitably 429s and we need to pick up where we left off.

---

## Where we are, what's next

**Current state:** end-to-end pipeline works in `--dry-run` mode (no
OpenAI call). One match through the orchestrator produces 5 SFT examples
covering damage moves, switches, status moves, and spread moves with
correctly-extracted ground-truth labels.

**Rough scale:** 13,919 matches × ~5 turns/match × ~70% extractable ≈
**~50k SFT examples** at full corpus scale. At GPT-4o pricing, the
synthesis run is roughly $1.5K–$3K — manageable, but worth running
the first ~100 matches and inspecting before committing.

**What we still want:**

1. **Real OpenAI run on the first 1 match → 10 → 100 → full**, with
   spot-checks on CoT quality at each step.
2. **A holdout eval set** — withhold a few hundred matches, measure
   whether the trained model's plays match the human's labels.
3. **RLHF on top** — once SFT gives us a model that can think like a
   pro, RLHF gives it the chance to outclass one. The SFT corpus we're
   building here is the floor, not the ceiling.

But the floor is the hard part. That's what's done.
