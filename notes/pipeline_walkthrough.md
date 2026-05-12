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
calc_microservice/  ──►  per-turn snapshots + events stream (Node side)
      │
      │  read JSONL, run inference + LLM
      ▼
pipeline/master_pipeline.py   (--mode sync | batch | hybrid)
      │
      │  per match:
      │    • prep:   match-final P1 bounds  →  threat matrix  →  prompts
      │    • synth:  teacher LLM (sync per-turn  OR  batch state machine)
      │    • judge:  one judge call per match  →  retry flagged turns
      │    • commit: per-match atomic write of surviving rows
      ▼
sft_training_data.jsonl   ──►  one fine-tuning row per turn
                              (per-match atomic commit)
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
| `POST /parse_log` | `@pkmn/protocol` + `@pkmn/client` + `@pkmn/sets` | Raw log → per-turn snapshots + per-turn `events` (TurnEvent discriminated union: move / switch / cant_move / tera / faint / item_event) + Bo3 `teamSheets` (Open Team Sheet decoded from `\|showteam\|`). |
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
        "ability": "Hadron Engine", "item": "Choice Specs",
        "teraType": "Fairy",
        "revealedMoves": [],
        "knownMoves": ["Volt Switch", "Draco Meteor", "Dazzling Gleam", "Electro Drift"],
        ... },
      { "slot": "b", "species": "Iron Valiant", ... }
    ],
    "bench": [ {"species": "Lunala", "fainted": false}, ... ]
  },
  "p2": { ... },
  "events": [ ... TurnEvent[] that happen DURING this turn ... ]
}
```

For OTS Bo3 replays, the response also carries a top-level `teamSheets`
with both players' full 6-Pokémon team sheets (decoded from `|showteam|`).
That's how `item` and `knownMoves` are filled in at turn 1 above — none of
those moves had been used yet. In Bo1 CTS replays, `teamSheets` is `null`
and `knownMoves` is `null` for every active mon (we only know what's been
revealed chronologically).

**Key design choice:** `events` is *forward-looking*. Events for turn N
attach to `snapshot[N].events`, so we always pair `(snap[N], snap[N+1],
snap[N].events)` for inference and label extraction.

**Symmetric brought-set bench, perspective applied in Python.** The
parser pre-scans the log once to find every Pokémon each side ever
sends to the field, and emits `bench` for **both sides** as the full
brought-set minus the current actives — symmetrically, regardless of
format. Each side also carries a `seenSpecies: string[]` field with
the chronological set of species ever active up to the current turn.

The perspective gating happens in `pipeline/master_pipeline.py`'s
`format_user_prompt`:

- **YOUR (P1) BENCH** renders the full brought-set. The player knows
  their own selection from team preview, so we always show all 4
  brought (minus any currently active). This applies in both Bo3 OTS
  and Bo1 CTS — in both formats the player's perspective at preview
  is "I know which 4 I'm bringing."
- **OPP (P2) BENCH** is filtered by `seenSpecies`: only species that
  have actually appeared on field at any turn ≤ current are shown.
  At turn 1 of any game, that's empty even when the parser knows P2
  has 4 brought somewhere — the player only learns the opponent's
  selection as they switch in.

This symmetric parser output is what makes
`flip_match_to_winner` clean: when we relabel the protocol-P2 winner
as the new P1, both sides already carry both views' worth of data, and
the post-flip P1 inherits the full-brought view of their own team
(which, before the flip, lived in `snap.p2`).

Real turn 1 events from the running example (the new discriminated-union
schema replaces the previous flat `DamageEvent[]`):

```json
[
  { "type": "move", "attacker_slot": "p1a", "move_name": "Volt Switch",
    "called_via": null,
    "hits": [{ "defender_slot": "p2a", "outcome": "damage",
               "hp_before_pct": 100, "hp_after_pct": 79,
               "is_crit": false, "is_ko": false }] },
  { "type": "move", "attacker_slot": "p2a", "move_name": "Moongeist Beam",
    "called_via": null,
    "hits": [{ "defender_slot": "p1b", "outcome": "damage",
               "hp_before_pct": 100, "hp_after_pct": 1,
               "is_crit": false, "is_ko": false }] }
]
```

The full TurnEvent union has six variants (`move`, `cant_move`, `tera`,
`switch`, `faint`, `item_event`) — see `calc_microservice/README.md`
for the schema.

**Move-caller filter.** Each `move` event carries `called_via`: null
when the attacker used the move directly, or the calling-move name
when something else triggered it. The parser's
`derivedRevealedMoves` (the `revealedMoves` field on each active mon)
and the Python `damage_inferencer.events_to_damage_events()` filter
both keep only `called_via in {null, "Sleep Talk"}`. The excluded
callers — Metronome, Copycat, Sketch, Snatch, Me First, Dancer,
Instruct, Mirror Move, Assist, Nature Power — can call moves the
attacker doesn't actually own, so attributing the called move to
their kit (or feeding the damage observation to EV inference) would
corrupt downstream state. Sleep Talk is included because it can only
call own moves. Mimic stays out of the filter set because in-battle
it permanently overwrites a slot, so subsequent uses really are in
the kit.

This filter directly fixes a CTS-Bo1 bug we used to have where
Hatterene/Smeargle Metronome calls were polluting their reconstructed
movesets.

Two events. Miraidon hit Lunala for 21%. Lunala hit Iron Valiant for 99%.
Notice that turn 1 had Iron Valiant doing *something else* too — but no
damage event for `p1b`. That something is detectable from the next snapshot
(it'll show `quickguard` in Iron Valiant's `revealedMoves`).

---

## Part 3: The intelligence layer

### Step 4 — Building the team picture: OTS (Bo3) vs CTS (Bo1)

In real VGC, **Bo3 uses Open Team Sheet** — both players see each other's
full 6-Pokémon roster, items, abilities, all 4 moves, and Tera types
before turn 1; only EVs / IVs / Nature stay hidden. **Bo1 uses Closed
Team Sheet** — only species are visible at team preview; everything else
is hidden until it activates or is used during the match.

Our pipeline branches accordingly. Same dataset, two completely different
team-knowledge regimes.

#### Bo3 (OTS): we see what the human saw

The replay log includes two `|showteam|` lines (one per side) carrying a
packed-team payload. We decode them with `Teams.unpackTeam` from
`@pkmn/sets` and surface the result as `teamSheets` on the parse_log
response.

For our running example (game 1 against jonen), P1's team sheet:

```
★ Miraidon       @ Choice Specs, ability=Hadron Engine, tera=Fairy
                  moves: Volt Switch / Draco Meteor / Dazzling Gleam / Electro Drift
★ Iron Valiant   @ Booster Energy, ability=Quark Drive, tera=Fairy
                  moves: Quick Guard / Encore / Moonblast / Close Combat
★ Lunala         @ Power Herb, ability=Shadow Shield, tera=Water
                  moves: Moongeist Beam / Meteor Beam / Wide Guard / Trick Room
★ Incineroar     @ Sitrus Berry, ability=Intimidate, tera=Ghost
                  moves: Knock Off / Fake Out / Parting Shot / Will-O-Wisp
  Ursaluna       @ Flame Orb, ability=Guts, tera=Ghost
                  moves: Headlong Rush / Facade / Earthquake / Protect
  Brute Bonnet   @ Mental Herb, ability=Protosynthesis, tera=Water
                  moves: Seed Bomb / Sucker Punch / Spore / Rage Powder
```

The ★ markers tag the 4 the human actually brought to *this* game — we
can compute them per-game by intersecting the OTS sheet with the
"ever-on-field" set from the log. (In a Bo3 series the same player can
bring different 4s in game 1 vs game 2; the system prompt is rendered
per-game so the brought-flag stays correct.)

The opponent's team sheet is shown in full too, but **without** brought
flags — at turn 1 we genuinely don't know which 4 of jonen's 6 will
come out. We learn that one switch at a time as the game unfolds.

#### Bo1 (CTS): we reconstruct from what got used

Bo1 replays have no `|showteam|`. We don't know items, abilities, moves,
or Tera types until they're revealed in play. So we **forward-scan**
every snapshot in the match and aggregate everything that ever surfaced
for each P1 species:

- Item — from the first snapshot it appears in (e.g. Sitrus Berry when
  it activates).
- Ability — from the first activation (Intimidate on switch-in,
  Protosynthesis from sun, etc.).
- Tera type — from the actual Terastallize event.
- Moves — every move the human used across the whole match.

If a Pokémon ends the match with fewer than 4 known moves, we pad with
the literal string `"[UNREVEALED_MOVE]"`. Then the system prompt's
**Masking Rule** tells the LLM what to do with the placeholders:

> *"If a Pokémon on Your Side has `[UNREVEALED_MOVE]` in its moveset, it
> means that move was never utilized by the human expert in this entire
> Bo3 series. You must assume that the unrevealed move was completely
> suboptimal, irrelevant, or unusable for this specific matchup. Do not
> attempt to guess what it is, and do not factor it into your strategic
> reasoning."*

The Masking Rule trains the model to reason from what's known and stay
agnostic about what isn't, instead of hallucinating moves it might want.
**Bo1 only** — the Bo3 prompt has no Masking Rule, because the OTS
sheet really does show all 4 moves and we want the model to use them.

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
By turn 3 of a real game, the inferencer can have already proven things
like "Miraidon's HP EVs are at most 35" and "SpD EVs are at most 11" —
which is why the threat matrix's Absolute envelope tightens visibly as
the game progresses.

**OTS doesn't replace this.** A common confusion: doesn't the OTS sheet
make all this binary-search machinery redundant? No — VGC OTS exposes
species / item / ability / moves / Tera type, but **not** EVs / IVs /
Nature. Those are still hidden, and they're exactly what shapes damage
ranges. OTS only collapses uncertainty on the *static* fields (which is
why our calc payloads on Bo3 turn 1 are immediately tighter — the
defender's item is locked in even before Sitrus activates). The spread
math stays unchanged.

**Three KnowledgeStates per match (Plan v3 asymmetry).** A subtle but
important refinement: we maintain THREE states, not two:

- `p2_running` — chronological, what the player has learned about
  the opponent through play so far.
- `p1_final` — computed **once per match** via
  `damage_inferencer.infer_match_final_bounds(games, ...)`, which runs
  the inferencer across every turn of every game in the match offline
  and returns the tightest bounds derivable from the *entire* match.
- `p1_running` — also chronological for P1, but unused in prompts; kept
  only for diagnostics and the inspector.

The match-final P1 state is the key insight: **the player knew their
own spread from day one.** At deploy time we'll present exact values
from the team-builder JSON. At training time, the closest available
approximation is "what the inferencer could prove by looking at the
whole match" — tighter than turn-by-turn inference would give us, and
more honest about what the player knew.

This produces an **asymmetric threat matrix**: the matrix calls in
plan v3 are now `generate_threat_matrix(snap_pre, "p1", p1_final,
p2_running, ...)`. Our damage ranges are tight on the P1 side (we
know our team's spreads); the opponent's damage ranges stay
chronologically loose (we genuinely don't know their EVs until we
observe them). Realism, not artificial uncertainty.

**P1's match-final bounds get surfaced to the LLM.** The same
`p1_final` that drives the matrix's P1 side also feeds a `=== YOUR
SPREADS ===` block in the user prompt — per-stat ranges for each
active P1 Pokémon. (The tag dropped its `(inferred)` qualifier in
Plan v3 because at deploy time this is exact knowledge, not
inferred.)

The render uses **one-sided constraints** rather than masking everything
that hasn't narrowed below an arbitrary width:

- Tightened upper bound only → `Stat ≤N`
- Tightened lower bound only → `Stat ≥N`
- Both sides tightened → `Stat lo–hi`
- Pinned to a single value → `Stat N`
- Fully open `[0, 252]` → not shown; trailing `, others ?` summarizes them
- Mon with no observations yet → `(no observations yet)`

Concrete example from a real bake-off Bo3:
`Calyrex-Ice: Hp ≥244, Atk ≤8, Def 28–36, Spa ≤8, Spd 228–236, Spe ≤8`
— a classic specially-defensive Calyrex build pinned tight by the
match-final pass. Because this is match-final not per-turn, the same
mon shows **identical numbers** at every turn of the same match (we
verified this property explicitly in the bake-off).

The system prompt's *Spread Rule* tells the model how to use the
ranges: worst-case for survival checks, best-case for offensive
checks. At deploy time the operator can present exact spreads from a
team-builder JSON instead of inferred ranges; the trained model is
robust to either form.

**Acknowledged implicit leak.** Mons that never took damage in the
match render as `(no observations yet)`. A student model could in
principle learn "if Kingambit shows '(no observations yet)' in YOUR
SPREADS, then Kingambit never took damage in this match." Plan v3
flagged this as accepted risk; a follow-up workstream
(canonical-prior substitution for fully-open stats) would mask the
signal at the cost of slightly fictional spreads in some cases.

### Step 6 — Threat matrix: dual-track damage envelope

For each turn, we render the damage landscape between the active
Pokémon. Two damage tracks per matchup-move:

- **Absolute** — strict math from KnowledgeState bounds. Wide, but
  provable.
- **Probable (meta)** — single calc result assuming both Pokémon run
  their canonical Smogon meta spread (from `canonical_priors`). Narrow,
  fast — but only as good as the prior.

**When the prior is wrong, we drop it.** When the canonical spread is
clipped from the inferred bounds by **≥ 40 EVs** on any relevant stat
(off-stat for the attacker prior, HP / def-stat for the defender), we
skip the Probable calc entirely and tag the line `(off-meta)`. Showing
a "Probable" range we've already disproven would actively mislead the
LLM. The 40-EV threshold avoids firing on edge-case clips of 1–10 EVs.

**Chip filter and spread grouping.** Two presentation tweaks make the
matrix decision-relevant rather than exhaustive:

- Moves whose Absolute max-percent is `< 15%` across every active
  defender get rolled into a one-line footer per attacker
  (`…plus N chip move(s): Snarl, Icy Wind`). The matrix stops being
  9HKO clutter.
- Spread moves (`allAdjacentFoes` / `allAdjacent` / `foeSide`) render
  as a single `[spread]` line listing every defender. The 0.75× spread
  modifier auto-applies via `@smogon/calc` when
  `field.gameType: "Doubles"`.

**Real example** (turn 3 of a Bo3 game where Chi-Yu has been Snarl'd
to -2 SpA and the inferencer has therefore proven the prior wrong on
all of Chi-Yu's offensive plays):

```
=== THREAT MATRIX  (turn 3, us=p1) ===

--- OUTGOING (us → opp) ---
[us Miraidon]  (boosts={spa: -2})
  Electro Drift → Chi-Yu      29.5%–44.6%  | meta 35.8%–42.3%  [guaranteed 3HKO]
  Electro Drift → Lunala       7.8%–15.1%  | meta 11.3%–13.3%  [possible 8HKO]
  Volt Switch  → Chi-Yu      20.5%–32.3%  | meta 24.8%–30.7%  [99.9% chance to 4HKO]
  Draco Meteor → Chi-Yu      29.5%–44.6%  | meta 35.8%–42.3%  [guaranteed 3HKO]
  …plus 1 chip move(s): Snarl

--- INCOMING (opp → us) ---
[opp Chi-Yu]  (boosts={spe: -1, spa: -2})
  Dark Pulse  → Miraidon       17.3%–22.3%  [possible 5HKO]  (off-meta)
  Heat Wave [spread]: Miraidon 11.7%–13.7%, Iron Bundle 47.8%–62.6%  [Iron Bundle: guaranteed 2HKO]  (off-meta)
  Overheat   → Iron Bundle    85.5%–116.8%  [87.5% chance to OHKO]  (off-meta)
[opp Lunala]
  Moongeist Beam → Iron Bundle  76.1%–119.8%  | meta 109.1%–129.5%  [guaranteed OHKO]
  Meteor Beam   → Iron Bundle  179.7%–287.0%  | meta 260.6%–307.6%  [guaranteed OHKO]
```

Notice four things at once:
- **Volatile state in the calc** — Miraidon's `spa: -2` from Snarl is
  threaded through Electro Drift / Volt Switch / Draco Meteor; numbers
  are real-board numbers, not sterile-lab.
- **`(off-meta)` on Chi-Yu's outgoing** — the inferencer has proven
  Chi-Yu's SpA is clipped well below the canonical "max SpA Modest"
  prior (Snarl-stacking does this fast), so the Probable column is
  dropped entirely.
- **`Heat Wave [spread]` as one line** — the 0.75× spread modifier is
  baked into the per-defender numbers, and both Miraidon and Iron
  Bundle's ranges sit on a single row.
- **Chip footer** — Snarl is `< 15%` max against either opponent so it
  collapses into the footer instead of taking 2 lines of prompt.

**Move enumeration changes per format.** For each attacker, the threat
matrix iterates `knownMoves` if present (Bo3 OTS — all 4 moves from the
sheet) and falls back to `revealedMoves` otherwise (Bo1 CTS —
chronologically-revealed only). Concrete consequence: at turn 1 of a
Bo3 game, the threat matrix already shows all four moves on every
attacker, including ones the human hasn't used yet. At turn 1 of a Bo1
game, every attacker's `revealedMoves` is empty — the threat matrix
section for that turn is therefore basically empty too, and gets
gradually fleshed out as moves come into view.

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
sweeper. That's exactly the kind of off-meta call the threat matrix's
`(off-meta)` tag will eventually fire on once enough evidence builds
up to clip the prior by ≥ 40 EVs.

---

## Part 4: Synthesizing the SFT dataset

### Step 8 — Action extraction (the ground-truth label)

**Whose play do we extract?** Before extraction, the orchestrator runs
`flip_match_to_winner(match)` so that the **series winner** becomes P1
for the rest of the pipeline. Bo3: the player who took 2 of 3 games.
Bo1: the per-game winner. The flip rewrites the entire match record —
top-level `players[0] ↔ players[1]`, every snapshot's `p1 ↔ p2` and
`tailwindP1 ↔ tailwindP2` (plus `tailwindP*TurnsLeft`), and every
`events[i]` slot/side fields (per discriminated-union variant), and
per-game `teamSheets.p1 ↔ teamSheets.p2`.

Why winner-only: we're training the model to play *correctly*, not to
mimic losing-Elo patterns. Including in-series losses (the games the
series-winner dropped along the way) preserves variance — sometimes a
pro plays correctly and loses to RNG, and that reasoning is still
high-quality training data. We don't filter game-by-game inside a
series.

For each turn, we need to know what each P1 active slot DID. That's the
training label. We reverse-engineer it from `snap_pre`, `snap_post`, and
the new `snap_pre.events` discriminated-union stream:

| Signal | Action type |
|---|---|
| `move` event with `attacker_slot == "p1a"` and `called_via in {null, "Sleep Talk"}` | move (target = single `defender_slot`, or `"spread"` if multiple, or `"self"` if `hits[]` is empty) |
| `isTerastallized` flips false→true at this slot | tera flag set on the move |
| `switch` event with `side="p1"`, `slot="p1a"`, `forced_by` is `null` | switch (`switch_to = to_species`) |
| `cant_move` event for this slot | pass (reason: `asleep` / `paralyzed` / `flinch` / `disable` / …) |
| Slot is empty or fainted at `snap_pre` | pass |
| None of the above (likely forced out by opponent before acting) | **ambiguous → skip turn** |

Real example, our running game 1 turn 1:

- **slot a (Miraidon)**: a `move` event with `attacker_slot="p1a"`,
  `move_name="Volt Switch"`, `called_via=null`, single damage hit on
  `p2a`. No Tera. → `{"action_type": "move", "move": "Volt Switch",
  "target": "p2a", "tera": false}`.
- **slot b (Iron Valiant)**: a `move` event with `move_name="Quick
  Guard"` and empty `hits[]` (status / self-target). →
  `{"action_type": "move", "move": "Quick Guard", "target": "self",
  "tera": false}`.

Forced switches (Volt Switch redirect, Eject Button, Roar, Whirlwind)
are NOT the human's choice for the slot they affect — those events
have `forced_by` set, and `extract_p1_actions` ignores them. If a slot
has no move / intentional switch / cant_move and isn't empty at
`snap_pre`, the turn is skipped (we can't recover what the human
*would* have chosen).

**We're conservative:** if any slot is ambiguous, the WHOLE turn is
skipped. Bad labels poison SFT — better to lose 30% of turns than label
30% of them wrong.

(In our 1-match dry run: 5 turns yielded SFT examples, 3 skipped as
ambiguous.)

### Step 9 — Teacher LLM: reverse-engineer the pro's play

Now the keystone insight that makes the whole pipeline work:

> **The teacher LLM doesn't have to FIND the optimal play. It has to
> JUSTIFY a known good play.**

Finding the optimal VGC play requires Day-2-Worlds intuition. No
frontier LLM has that out of the box. But *rationalizing* a play it's
been told is correct — given the board state and a precomputed threat
matrix — is a much easier task. We get high-quality CoT essentially
for free from a model that doesn't itself know VGC at pro level. (The
teacher itself is provider-pluggable: OpenAI / Anthropic / Google
adapters all behind one `TeacherProvider` ABC.)

**Synthesis call shape:**

```
system: "You are a top-tier competitive VGC Reg I player…
         [format-specific rule 1] [Tool] [Threat-Matrix] [Spread] [Alternatives] [Output]"

user:   [board state: turn N, field, P1/P2 active+bench]
        [YOUR SPREADS (inferred per-stat ranges)]
        [threat matrix: Absolute + Probable, off-meta lines drop Probable]
        ────── EXPERT'S DECISION (oracle truth) ──────
        { "slot_1": {"action_type": "move", "move": "Volt Switch", ...},
          "slot_2": {"action_type": "move", "move": "quickguard", ...} }

assistant: [CoT articulating why this play is correct + briefly evaluating 1–2 alternatives]
           [optional calculate_damage tool call to verify a critical assumption]
tool:      [calc result]
assistant: { "pre_tool_thought": "Volt Switch pivots Miraidon out of Lunala's
              incoming Moongeist Beam while still chipping p2a; Quick Guard
              blocks any priority follow-up. Considered Discharge instead —
              calc'd ~10% on Lunala — too little pressure for the slot loss.",
              "action": { "slot_1": {...}, "slot_2": {...} } }
```

**Six rules in the system prompt** (same backbone, format-specific
only on rule 1):

1. **Masking Rule** (Bo1) / **OTS Rule** (Bo3) — what's known about
   the team sheet.
2. **Tool Rule** *(rewritten in Plan v3 — no per-turn minimum)* —
   `calculate_damage` is a precision instrument for **hypotheticals
   the threat matrix doesn't already cover**:
   - Switch-ins from the bench ("if I bring Calyrex in, what does
     Lunala do?")
   - Backline matchups
   - Future-state ("at +2 SpA after Calm Mind…")
   - Tera predictions
   The matrix already enumerates every active-vs-active damage cell
   for the current turn, so re-calc'ing those is forbidden. The model
   **may** commit via `submit_decision` immediately if the matrix
   suffices. The earlier "must call calculate_damage at least once"
   minimum was dropped — it was creating redundant calc calls.
   `MAX_CALC_CALLS_BEFORE_FORCE_SUBMIT = 5` is the upper cap; past 5
   calc calls the next iteration forces `submit_decision`.
3. **Threat-Matrix Rule** — each line shows Absolute (provable) and,
   when not contradicted, Probable (meta). Off-meta lines tagged
   `(off-meta)` show only Absolute.
4. **Spread Rule** — your spreads may be exact values or inferred
   ranges. With ranges: worst-case for survival, best-case for offense.
5. **Alternatives Rule** *(known limitation)* — briefly evaluate 1–2
   plausible alternatives (other moves, switches) before committing.
   `# TODO(rlhf-followup)`: this prompt-driven approach lets the
   teacher cherry-pick weak alternatives because it knows the answer;
   real fix is minimax / MCTS distillation in a separate workstream.
6. **Output Rule** — commit via `submit_decision` tool with arguments
   `{pre_tool_thought, action: {slot_1, slot_2}}`.

**Two templates, one branch.** Bo1 includes the Masking Rule and
shows the *reconstructed* P1 team with `[UNREVEALED_MOVE]`
placeholders for moves the player hasn't yet used this match. Bo3
includes the OTS Rule and shows full sheets for both sides with ★
markers on P1's brought 4 (different per game in a series). Picked by
`if match["format"] == "bo3"` in the orchestrator. **All wording is
present-tense** — the model is trained to think it's playing live, not
reviewing a recording.

**Tool architecture, not response_format.** The model has only one
output channel: tool calls. It can either call `calculate_damage` or
commit via `submit_decision` (called exactly once per turn). The
JSON schema lives on `submit_decision`'s parameters, not on
`response_format`. This was an explicit fix — an earlier
`response_format: json_schema` setup let the model bypass the calc
tool entirely on some turns. With `submit_decision` being the only
way to produce structured output, the model is forced through the
tool loop.

**Two things to notice:**

1. **The `=== TRAINING-MODE TARGET ===` suffix is stripped before
   saving.** The trained model never sees the ground truth in its
   prompt. It learns to produce the same chain of reasoning *without*
   the cheat — that's the whole SFT magic. (Originally tagged
   `=== EXPERT'S DECISION ===`; Plan v3 rewrote it to lean less
   into "expert" framing, which the model was repeating back in
   CoT.)

2. **The `calculate_damage` tool is the critical-call escape hatch.** The
   teacher LLM can verify any specific claim ("does Tera-Fairy Calyrex
   actually OHKO Rillaboom with this build?") before committing. The
   calc service handles `isCrit`, `isTera`, `boosts`, `status`, weather,
   terrain — all the volatile state the LLM might want to vary.

3. **Two-stage leak filter (Plan v3 regex + Plan v4 judge).** Even
   with the rewritten suffix and the present-tense system prompt,
   models occasionally produce CoTs that reference the training
   framing. The orchestrator runs every CoT through two filters:
   - **Regex** (`detect_oracle_leak`): fast first pass on the saved
     `pre_tool_thought`. Catches "oracle", "ground-truth", "the
     target {is,says,action,field}", "training {mode,section,target,
     example}". Tightened in May 2026 after a bake-off audit found
     Anthropic produced "the target action" / "training section" in
     32% of saved rows.
   - **Model judge** (`teacher/judge.py`): runs once per match after
     all turns synthesize. Catches the long tail — "clearly the right
     move", "the data points to X" — phrases too varied for a regex.
   See **Step 11** below for the judge mechanics.

#### The user prompt's structure (eight sections)

The user prompt is composed turn-by-turn from the snapshot + threat
matrix + accumulated KnowledgeStates. In order of appearance:

1. **Turn / field header** — turn number, weather, terrain, tailwind
   booleans.
2. **YOUR (P1) ACTIVE / BENCH** — current actives with HP %, status,
   item, ability, Tera state, boosts, revealed moves; bench is the
   full brought-set (the player knows their selection from team
   preview). When a slot is genuinely vacant (last Pokémon, no
   replacement), it's annotated as `[b] (empty — no Pokémon
   remaining)` so the model doesn't have to infer slot vacancy.
3. **OPP (P2) ACTIVE / BENCH** — same shape, but bench is gated
   chronologically by `seenSpecies` (the player only learns opp's
   selection as they switch in).
4. **`=== GAME-STATE LEDGER ===`** — only-when-active rows: faints
   (always), Tera used per side (when fired), field/pseudo-weather
   with turns-left, side conditions (screens / spikes / safeguard /
   tailwind), volatiles on actives (substitute / encore / taunt /
   etc.), choice locks (when item + lastMove imply one), recent item
   events, and a **Cumulative damage** row showing per-active total
   damage taken across past turns and turns-on-field. Empty rows
   omit; the section is dense but compact.
5. **`=== TURN-BY-TURN (game N) ===`** — every prior turn this game
   rendered as one-liners, indented continuation. No length cap.
   Sequence-aware reasoning is the entire point — we don't truncate.
6. **`=== SERIES STATE (Bo3, game N of M) ===`** *(Bo3 game ≥ 2
   only)* — for each prior game: header (winner, turns, brought
   rosters, Tera resolutions) followed by the **full inlined
   turn-by-turn rollup** of that game. There used to be a `Notable`
   heuristic line here that fired only on choice-lock + Trick Room;
   that wasn't robust enough so we dropped it in favor of raw
   inlining. There's a `# TODO(token-efficient-series-summary)`
   marker — distilling priors via a learned summarizer would
   conserve attention, but until that's built, raw is the right
   move.
7. **`=== YOUR SPREADS ===`** — per-active P1 mon, every tightened EV
   bound (one-sided constraints surface real signal that earlier
   render thresholds hid). Plan v3 changed this from per-turn
   chronological bounds to **match-final** bounds (the inferencer
   runs across the whole match once at the start). At deploy time
   the operator surfaces exact spreads from team-builder JSON here.
8. **`=== THREAT MATRIX (turn N, us=p1) ===`** — dual-track damage
   envelope from `threat_matrix.py`. Plan v3 drove this with the
   **asymmetric** `(p1_final, p2_running)` knowledge pair: tight on
   our side, chronologically loose on the opponent's.

The total prompt is ~2,500–4,500 tokens depending on game length.

### Step 10 — Buffer the row, update knowledge, repeat (then commit the whole match)

Plan v4 changed the write semantics. Each successful turn used to write
one row to the JSONL immediately. Now turns **buffer in memory** per
match; the orchestrator commits the entire match atomically after the
match-level judge runs (Step 11). One match either lands complete or
doesn't show up at all — no half-written matches mid-crash.

The row shape itself is unchanged: OpenAI fine-tuning JSONL.

```json
{
  "match_id":   "bo3-gen9vgc2026regibo3-2590204993",
  "game_index": 0,
  "turn":       3,
  "format_id":  "gen9vgc2026regibo3",
  "messages":   [<system>, <user>, <assistant tool_calls>?, <tool>?, …, <final assistant>]
}
```

The `messages` array can include intermediate `assistant` (with
`tool_calls`) and `tool` messages from the calc-tool loop, before the
final structured `assistant` content. `--dry-run` mode skips the loop
and produces a 3-message form (system / user / placeholder assistant)
useful for verifying orchestration without OpenAI cost.

**Concrete user-prompt body** for one mid-game Bo1 turn — turn 23 of a
real match where the series winner is P1, with Amoonguss and a
Tera'd-Water Calyrex-Ice active and 2 unrevealed bench mons:

```
=== TURN 23 ===
Field: weather=Rain, P1-tailwind=no, P2-tailwind=no

YOUR (P1) ACTIVE:
  [a] Amoonguss     | HP 30% | revealed=Spore,Rage Powder
  [b] Calyrex-Ice   | HP 50% | status=par | item=leftovers | TERA-ACTIVE (Water) | boosts=spa-1,atk+3 | revealed=Protect,Leech Seed,Trick Room,Glacial Lance
YOUR (P1) BENCH: Flutter Mane, Incineroar

OPP (P2) ACTIVE:
  [b] Kyogre | HP 15% | item=leftovers | ability=drizzle
OPP (P2) BENCH: Grimmsnarl (fainted), Annihilape (fainted), Calyrex-Ice (fainted)

=== GAME-STATE LEDGER ===
Faints:        P1 0/4   |   P2 3/4
Tera used:     P1 ✓ Calyrex-Ice → Water on T2; P2 ✓ Annihilape → Poison on T1
Field:         Rain (5 turns left)
P2 side:       reflect L1
Item events:   P1[a] Focus Sash consumed (T1); P1[a] Eject Button consumed (T2)
Cumulative:    P1[a] Amoonguss took 69% across 1 hit(s) over 13 turn(s) on field
               P1[b] Calyrex-Ice took 189% across 6 hit(s) over 22 turn(s) on field
               P2[b] Kyogre took 31% across 1 hit(s) over 8 turn(s) on field

=== TURN-BY-TURN (game 1) ===
T1: P1[b] Fake Out → P2[a] 95%
    P2[a] couldn't move (flinch)
    …(22 prior turns)…
T22: P1[a] Spore → P2[a] sleep
     P1[b] Glacial Lance (spread) → P2[a] 0% KO, P2[b] 22%
     P2 switched P2[a]: (empty slot) → Kyogre

=== YOUR SPREADS (inferred) ===
  Amoonguss:    HP ≤91, Def ≤120, others ?
  Calyrex-Ice:  HP ≤140, others ?

=== THREAT MATRIX  (turn 23, us=p1) ===

--- OUTGOING (us → opp) ---
[us Calyrex-Ice]  (boosts={atk: +3, spa: -1})
  Glacial Lance → Kyogre  20.5%–48.6%  | meta 38.3%–45.3%  [guaranteed 2HKO]
  …plus 2 chip moves: Protect, Leech Seed

--- INCOMING (opp → us) ---
[opp Kyogre]
  Water Spout → Calyrex-Ice  6.8%–11.2%  | meta 8.4%–10.0%  [possible 9HKO]
  …plus 2 chip moves: Origin Pulse, Ice Beam
```

For a Bo3 game-2-or-later turn, an additional `=== SERIES STATE (Bo3,
game 2 of 3) ===` block sits between TURN-BY-TURN and YOUR SPREADS,
inlining the full action log of every prior game in the series.

The structured ground-truth label (the human's actual play, kept *out*
of saved messages but supplied to the teacher LLM during synthesis):

```json
{
  "slot_1": {"action_type": "move", "move": "Spore",         "target": "p2b", "tera": false, "switch_to": null},
  "slot_2": {"action_type": "move", "move": "Glacial Lance", "target": "spread", "tera": false, "switch_to": null}
}
```

A real teacher LLM run replaces the final assistant message with a
`submit_decision` tool call carrying a `pre_tool_thought` chain that
justifies this play — likely *"Amoonguss Spores Kyogre to take it out
of the picture for several turns; +3 Calyrex-Ice's Glacial Lance pins
Kyogre to a guaranteed 2HKO and threatens any switch-in. Considered
Leech Seed instead — the recovery isn't worth the lost damage this
turn."* — preceded by 1–2 `calculate_damage` tool calls verifying
decisive hypotheticals.

**Action shapes.** A slot's action is one of:

- `move` — `move`, `target` (`p2a` / `p2b` / `spread` / `self`),
  `tera` (bool).
- `switch` — `switch_to` (species name).
- `pass` — slot is genuinely vacant (last Pokémon, no replacement
  available) or the mon couldn't act (sleep / paralysis / flinch /
  disable). Real Pokémon mechanics, not a "do nothing" option.

**After the row is buffered**, we filter the same `events` stream
through `damage_inferencer.events_to_damage_events()` (keeping only
`type=="move"` events with `called_via in {null, "Sleep Talk"}` and
`hits[i].outcome == "damage"`) and feed those into
`damage_inferencer.update_knowledge` — tightening `p2_running` so the
*next* turn's threat matrix is sharper on the P2 side. (`p1_final` was
computed once at the start of the match; it doesn't need turn-by-turn
updates.) Bo3 series compound this: by turn 3 the inferencer has
already proven Miraidon's HP EVs are at most 35.

The whole loop is **resumable**. `(match_id, game_index, turn)` keys
already in the output JSONL are skipped on rerun — useful when the
OpenAI run 429s and we need to pick up where we left off. Per-match
atomic write means even mid-run crashes don't leave partial matches.

### Step 11 — Match-level model judge (Plan v4)

The regex leak filter is fast but narrow. The May 2026 bake-off audit
turned up a long tail of softer phrasings the regex doesn't catch
("clearly the right move", "the data points to X", "looking at the
training section") that a model can spot easily. So before the
buffered match writes, the orchestrator runs **one** call to a cheap
OpenAI model with every CoT from that match:

```python
turn_records = [
    {"turn_idx": i, "match_id": "...", "game_idx": ..., "turn": ...,
     "pre_tool_thought": <the assistant's commit CoT>}
    for i, r in enumerate(match_buffer)
]
jr = await judge_match_cots(turn_records, client=judge_client)
# jr.flagged_turn_indices  →  [3, 7]   (e.g.)
# jr.reasons               →  {3: '"Looking at the training section..."',
#                              7: '"Clearly the right move is..."'}
```

The judge call uses OpenAI's structured-output schema (
`{flagged_turns: [{turn_idx, reason}]}`), so its response is
deterministic JSON — no partial parses, no regex on the response.

If any turns are flagged, the orchestrator **re-synthesizes** them
through the same sync teacher (even in `--mode batch` — batch
latency is too high for retry). Up to `--judge-retries` passes
(default 2). After exhaustion, only the still-flagged turns drop;
the rest of the match writes cleanly.

**Why per-match, not per-row.** Amortizes the system prompt across N
turns. An 8-turn match costs ~$0.014 with `gpt-5.5` (~$0.0015 with
`gpt-5.5-mini` when access opens up) versus ~$0.04 if we judged each
row separately. Also lets the judge spot cross-turn patterns —
multiple consecutive references to the training framing are a
stronger signal than each in isolation.

**Why match-level fits buffered-write naturally.** The judge needs
every CoT in memory anyway. Buffering per match for atomic write
puts them right there. Per-turn write + per-row judge would have
needed a separate buffer-then-delete dance.

**Fail-open.** If the judge call errors (network, rate limit,
schema mismatch), we write the match as if the judge passed. Better
to ship a few possibly-leaky rows than drop a whole match for an
infrastructure hiccup. The regex filter is still the first line of
defense; the judge is the long-tail catch.

**Sample output** (real fixture from the smoke test):

```
calling judge (gpt-5.5)...
  flagged: [1, 2]
  reasons: {1: '"Looking at the training section, the target action is..."',
            2: '"Clearly the right move here is Protect on both slots."'}
  cost: $0.00729  (994 input / 116 output)
```

Turn 0 (a clean, fully-derived "Calyrex outspeeds and OHKOs after
Tera…") was correctly spared.

### Step 12 — Batch mode + hybrid (Plan v4)

The bake-off picked OpenAI as the production teacher. At full corpus
scale (~13K matches × ~5 turns × ~70% extractable ≈ 50k SFT rows ≈
$2.3K sync), submitting via the OpenAI Batch API saves ~50% on both
input and output tokens — meaningful at this volume.

**The catch:** OpenAI Batch requires each request line to be a single
self-contained API call. Our teacher LLM tool loop is N sequential
calls (call calc, get result, decide, call calc again or commit). The
Batch API can't represent that loop inside one line.

**Solution: one batch cycle per tool-loop iteration.** All in-flight
turns at `iter=K` bundle into one batch upload. After that batch
completes, the orchestrator processes responses sequentially — running
any `calculate_damage` tool calls synchronously against the local calc
microservice — and advances each turn's `iter` counter. Turns that
called `submit_decision` are done; turns that need another calc round
go back into the next cycle's batch.

```
cycle 0:  [all 8 turns at iter=0]  →  one Batch upload, ~30min–3h
            ↓ poll until completed → fetch → apply per turn
            • run calc for any calc tool calls (sync)
            • mark done | advance to iter=1

cycle 1:  [3 turns still pending]  →  one Batch upload
            ↓
cycle 2:  [1 turn still pending]   →  one Batch upload
            ↓
[regex leak filter + match-level judge as in Step 11]
[atomic per-match write]
```

In smoke testing, an 8-turn Bo3 match completed in 2-3 cycles, each
cycle <2 minutes wall-clock. A corpus-scale run will land closer to
the documented 1–3h p50 per cycle.

**Per-match state files.** `pipeline/batch_state/{match_id}.json`
carries every `BatchWorkItem` (turn) — `api_messages` history,
`iter`, `calc_calls`, `status` (`pending | submitted | committed | failed
| leak_persistent | judge_flagged_persistent | written`), and an
**`active_batch_id` breadcrumb** that points at whichever batch the
item is currently waiting on. `--resume` reads every state file,
drains in-flight batches via `_resume_inflight_batches`, then
re-enters the cycle loop with everything back in known state.
Verified in unit tests across four scenarios: orphan items (no
breadcrumb), poll failure, missing custom_id in fetch results, clean
recovery.

**`--mode hybrid`** is the recommended production strategy.

1. First `--hybrid-sync-n` matches (default 50) run **sync** with the
   judge ON. The orchestrator computes match-rate (`written / attempted`)
   and leak-rate (`dropped / attempted`).
2. If `match_rate ≥ --hybrid-min-match-rate` (default 0.95) AND
   `leak_rate ≤ --hybrid-max-leak-rate` (default 0.02), the rest of the
   corpus goes through batch.
3. If either threshold fails: the run **halts** before submitting any
   batch upload. Surfaces a regressed prompt or a busted model version
   loudly instead of silently shipping thousands of dollars to a
   broken run.

The sync portion serves as a quality gate AND a sanity check on the
model's current behavior. The batch portion gets the cost win on the
bulk of the corpus.

---

## Where we are, what's next

**Current state:** end-to-end pipeline works in `--dry-run`, sync,
batch, and hybrid modes. Live with provider adapters for OpenAI
(production default), Anthropic, and Google (`teacher/openai.py` /
`teacher/anthropic.py` / `teacher/google.py`, all re-exported via
`teacher/__init__.py`). One match through the orchestrator in sync
mode produces 8–10 SFT rows covering damage moves, switches, status
moves, and spread moves with correctly-extracted ground-truth labels,
match-final P1 spreads, an asymmetric threat matrix, and the
historical-context prompt structure. A model-judge call confirms CoT
hygiene before per-match atomic commit.

**Rough scale:** 13,919 matches × ~5 turns/match × ~70% extractable ≈
**~50k SFT examples** at full corpus scale. At gpt-5.5 pricing the
sync run is roughly $2K–$3K. Plan v4's `--mode batch` halves that
via the OpenAI Batch API (50% off both input and output tokens;
~$1.15K saved on full corpus). The judge layer adds ~$5–40 across
the corpus depending on whether mini access is available.

**What's landed beyond the original walkthrough:**

- Provider-agnostic teacher LLM (OpenAI / Anthropic / Google).
- `submit_decision` tool architecture (replaced `response_format`).
- Historical-context user prompt (GAME-STATE LEDGER with Cumulative
  damage row, TURN-BY-TURN, SERIES STATE with full prior-game
  rollups, empty-slot annotation, present-tense system prompt).
- Symmetric brought-set bench rendering with perspective gating in
  Python (`seenSpecies`); Tera-form aliasing for Terapagos / Ogerpon.
- Move-caller filter expanded (Mirror Move / Assist / Nature Power
  added to NON_OWN_CALLERS).
- Choice-lock detection working (uses OTS-resolved item, normalizes
  move ID → display name via the dex).
- `bakeoff.py` for head-to-head provider comparison — **resolved**:
  OpenAI gpt-5.5 won (100% match rate, 0% leak, $0.07/row). Google
  also clean and cheaper. Anthropic produced 32% near-miss meta-leak
  rate — informed the regex tightening and motivated the judge layer.
- **Plan v3 (post-bake-off):** rewrote Tool Rule, dropped per-turn
  calc minimum, switched YOUR SPREADS to match-final bounds, made
  the threat matrix asymmetric `(p1_final, p2_running)`, added the
  `=== TRAINING-MODE TARGET ===` framing for the ground-truth suffix,
  added regex leak retry with `--leak-retries`.
- **Plan v4:** match-level model-judge validator
  (`teacher/judge.py`) + `--mode {sync,batch,hybrid}` dispatcher with
  OpenAI Batch API orchestration (`batch_runner.py` +
  `teacher/batch_openai.py`) + per-match atomic write + resume
  support via `active_batch_id` breadcrumbs in
  `batch_state/{match_id}.json`.

**What we still want:**

1. **Real corpus run** in hybrid mode — 50 matches sync as gate, then
   batch the rest. Spot-checks on CoT quality at each step.
2. **A holdout eval set** — withhold a few hundred matches, measure
   whether the trained model's plays match the human's labels.
3. **Anthropic / Google batch adapters** — `BatchTeacherProvider`
   abstraction is in place; siblings `batch_anthropic.py` /
   `batch_google.py` slot in mechanically when those providers
   move into production rotation.
4. **Canonical-prior substitution in YOUR SPREADS** — mask the
   implicit "this mon never took damage" leak from
   `(no observations yet)` rendering.
5. **A separate selection-model SFT corpus** — the team-preview
   4-of-6 pick is a different decision (matchup theory, no tactical
   state) and deserves its own model.
6. **Minimax / MCTS distillation for the alternatives loop** —
   replaces the current prompt-driven Alternatives Rule (which lets
   the teacher cherry-pick weak alternatives because it knows the
   answer) with a real search step. Tracked as
   `# TODO(rlhf-followup)` in `teacher/base.py`.
7. **RLHF on top** — once SFT gives us a model that can think like a
   pro, RLHF gives it the chance to outclass one. The SFT corpus
   we're building here is the floor, not the ceiling.

But the floor is the hard part. That's what's done.
