# calc_microservice

Express + TypeScript service that wraps the official Smogon TypeScript libraries
and exposes them over HTTP. Two endpoints:

| Endpoint | Backed by | Purpose |
|---|---|---|
| `POST /calc` | [`@smogon/calc`](https://www.npmjs.com/package/@smogon/calc) | Deterministic damage range for an attacker × move × defender × field. |
| `POST /parse_log` | [`@pkmn/protocol`](https://www.npmjs.com/package/@pkmn/protocol) + [`@pkmn/client`](https://www.npmjs.com/package/@pkmn/client) | Replay log → array of turn-by-turn battle-state snapshots. |
| `GET /dex/move/:name` | [`@pkmn/dex`](https://www.npmjs.com/package/@pkmn/dex) | Move metadata lookup (category, type, base power, target, priority). Used by the threat matrix to skip Status moves. |

Both endpoints are the *only* place in the project that knows about Showdown's
internal data formats. Everything downstream (Python pipeline, teacher LLM tool
calls) goes through this HTTP boundary, so the underlying libraries can be
upgraded independently.

## Setup

```bash
cd calc_microservice
npm install
```

Requires Node ≥20.

## Run

```bash
npm run dev          # tsx watch on src/server.ts (auto-reload)
npm run build        # tsc → dist/
npm start            # node dist/server.js (after build)
```

Listens on `http://localhost:3000` by default. Override with `PORT=8080 npm run dev`.

```bash
curl http://localhost:3000/health
# → {"status":"ok"}
```

## `POST /calc`

### Request body

```ts
{
  attacker: PokemonInput,
  defender: PokemonInput,
  move: string | { name: string, isCrit?: boolean, hits?: number },
  field?: FieldInput
}
```

`move` accepts either a bare string (legacy form, treats as a normal hit)
or an object form. The object form forwards `isCrit` and `hits` directly to
the `@smogon/calc` `Move` constructor — `isCrit: true` applies the 1.5×
crit multiplier and bypasses the defender's positive boosts (used by
`damage_inferencer.py` when an observed event was flagged as a crit).

#### `PokemonInput`

| Field | Required | Type | Notes |
|---|---|---|---|
| `species` | required | string | e.g. `"Miraidon"`, `"Calyrex-Shadow"`. |
| `item` | required | string | Held item. Use `""` for none. |
| `ability` | required | string | Active ability. |
| `level` | optional | number | Defaults to `50` (VGC). |
| `currentHP` | optional | number \| string | Either a flat HP value (`120`) or a percentage string (`"75%"`). Defaults to full HP. |
| `status` | optional | `'' \| 'brn' \| 'par' \| 'psn' \| 'tox' \| 'slp' \| 'frz'` | Defaults to no status. |
| `teraType` | required | string | The Pokémon's intrinsic Tera type (always known from the open team sheet). |
| `isTera` | optional | boolean | Whether the Pokémon has Terastallized this game. Defaults to `false`. The calc only treats the mon as Tera'd when this is `true`. |
| `boosts` | required | `{ atk?, def?, spa?, spd?, spe?, accuracy?, evasion? }` | Stat-stage boosts (`-6` … `+6`). Pass `{}` for none. |
| `evs` | optional | partial stats table | Omit to inject "worst-case" assumed spread on the Python side. When omitted the calc falls back to 0 EVs. |
| `ivs` | optional | partial stats table | Omit → 31 IVs. |
| `nature` | optional | string | Omit → Hardy (neutral). |

The optional `evs` / `ivs` / `nature` exist because under VGC Open Team Sheet
rules we know an opponent's Pokémon's species / item / ability / Tera type but
**not** their EV / IV / nature spread. The calling code is responsible for
deciding what assumed spread to inject (e.g. "max-rolls offensive" for opponent
threats, "bulkiest plausible" for our own survival checks).

#### `FieldInput`

| Field | Default | Notes |
|---|---|---|
| `gameType` | `"Doubles"` | Always doubles for VGC. |
| `weather` | none | `"Sun"`, `"Rain"`, `"Sand"`, `"Snow"`, `"Harsh Sunshine"`, etc. |
| `terrain` | none | `"Electric"`, `"Grassy"`, `"Misty"`, `"Psychic"`. |
| `isGravity`, `isMagicRoom`, `isWonderRoom` | false | |
| `attackerSide` | `{}` | See below. |
| `defenderSide` | `{}` | See below. |

Each side accepts: `spikes`, `isReflect`, `isLightScreen`, `isAuroraVeil`,
`isProtected`, `isHelpingHand`, `isFriendGuard`, `isTailwind`, `isFlowerGift`,
`isBattery`, `isPowerSpot`, `isSteelySpirit`, `isSR`.

### Response

```ts
{
  damageRolls: number[],         // raw HP-damage rolls from @smogon/calc
  minDamage: number,             // min(damageRolls)
  maxDamage: number,             // max(damageRolls)
  defenderMaxHP: number,
  defenderCurrentHP: number,
  minPercent: number,            // minDamage / defenderMaxHP * 100, 1 decimal
  maxPercent: number,
  koChance: string,              // e.g. "guaranteed OHKO", "18.8% chance to OHKO", "guaranteed 2HKO"
  description: string,           // full Smogon-style description
  moveDescription: string        // short percent-range string
}
```

Errors return `400` with `{ "error": "..." }`.

## Example

```bash
curl -s -X POST http://localhost:3000/calc \
  -H "Content-Type: application/json" \
  -d '{
    "attacker": {
      "species": "Miraidon",
      "item": "Choice Specs",
      "ability": "Hadron Engine",
      "teraType": "Electric",
      "boosts": {},
      "evs": {"spa": 252}, "nature": "Modest"
    },
    "defender": {
      "species": "Calyrex-Shadow",
      "item": "Life Orb",
      "ability": "As One (Spectrier)",
      "teraType": "Normal",
      "boosts": {},
      "evs": {"hp": 252, "spd": 252}, "nature": "Calm"
    },
    "move": "Electro Drift",
    "field": {"gameType": "Doubles", "terrain": "Electric"}
  }' | jq
```

```json
{
  "damageRolls": [180, 184, 186, 187, 189, 192, 193, 196, 198, 201, 202, 205, 207, 210, 211, 213],
  "minDamage": 180,
  "maxDamage": 213,
  "defenderMaxHP": 207,
  "defenderCurrentHP": 207,
  "minPercent": 86.9,
  "maxPercent": 102.8,
  "koChance": "18.8% chance to OHKO",
  "description": "252+ SpA Choice Specs Hadron Engine Miraidon Electro Drift vs. 252 HP / 252+ SpD Calyrex-Shadow in Electric Terrain: 180-213 (86.9 - 102.8%) -- 18.8% chance to OHKO",
  "moveDescription": "86.9 - 102.8%"
}
```

## `POST /parse_log`

Turns a raw Pokémon Showdown replay log (the pipe-delimited transcript stored
in each replay JSON's `log` field) into a sequence of structured snapshots, one
per turn. Replaces what would otherwise be a brittle Python regex parser; uses
the official `@pkmn/client` `Battle` state machine, which correctly handles
edge cases like Zoroark illusion, end-of-turn ordering, multi-hit moves,
forme changes, and revealed-info tracking.

### Request body

```ts
{ "log": "raw pipe-delimited Showdown log string" }
```

### Response

```ts
{
  "snapshots":  TurnSnapshot[],
  "teamSheets": { p1: OtsPokemonSet[]; p2: OtsPokemonSet[] } | null
}
```

`teamSheets` is populated only when the log contains `|showteam|` lines —
i.e. on **OTS Bo3 replays**. CTS Bo1 returns `teamSheets: null`. Each side's
`OtsPokemonSet[]` is the full 6-Pokémon roster decoded from the
`|showteam|` packed payload via `@pkmn/sets`'s `Teams.unpackTeam`. VGC OTS
exposes species / item / ability / 4 moves / Tera type — but **not** EVs /
IVs / Nature, so those fields come back as `null`.

```ts
interface OtsPokemonSet {
  species: string;
  item: string;
  ability: string;
  moves: string[];          // exactly 4 (padded with "" if shorter)
  teraType: string | null;
  level: number;
  gender: string | null;
  // VGC OTS hides these — always null on real Bo3 replays:
  nature: string | null;
  evs: Record<string, number> | null;
  ivs: Record<string, number> | null;
}
```

One snapshot is emitted at the **start of every turn** (i.e. immediately after
the `|turn|N` protocol line is processed). The state reflects everything that
happened up to and including the end of turn `N-1`.

```ts
interface TurnSnapshot {
  turn: number;
  field: {
    weather: string | null;          // "Sun", "Rain", "Sand", "Snow"
    weatherTurnsLeft?: number;
    terrain: string | null;          // "Grassy", "Electric", "Misty", "Psychic"
    terrainTurnsLeft?: number;
    pseudoWeather: { [id: string]: { turnsLeft: number } };  // trickroom, gravity, magicroom, ...
    tailwindP1: boolean;
    tailwindP1TurnsLeft?: number;
    tailwindP2: boolean;
    tailwindP2TurnsLeft?: number;
  };
  p1: SideSnapshot;
  p2: SideSnapshot;
  events: TurnEvent[];               // discriminated union — see below
}

// A discriminated union of every per-turn event the parser can surface.
// Replaces the old `actionLog: DamageEvent[]` shape. Forward-looking:
// `snapshot[N].events` describes things that happened DURING turn N.
type TurnEvent =
  | { type: "move",
      attacker_slot: string,         // "p1a" | "p1b" | "p2a" | "p2b"
      move_name: string,             // PS display name, e.g. "Glacial Lance"
      called_via: string | null,     // null for direct use; else "Sleep Talk"/"Metronome"/etc.
      hits: Array<{
        defender_slot: string,
        outcome: "damage" | "miss" | "blocked" | "immune" | "no_effect" | "fail",
        hp_before_pct?: number,      // present iff outcome="damage"
        hp_after_pct?: number,
        is_crit?: boolean,
        is_ko?: boolean,
        cause?: string               // e.g. "Protect", "Wide Guard" for blocked
      }>
    }
  | { type: "cant_move",
      slot: string,
      reason: string,                // "asleep" | "paralyzed" | "frozen" | "flinch" | ...
      attempted_move?: string
    }
  | { type: "tera",
      side: "p1" | "p2",
      slot: string,
      species: string,
      to_type: string                // "Water", "Fairy", "Stellar", ...
    }
  | { type: "switch",
      side: "p1" | "p2",
      slot: string,
      from_species: string | null,   // null when slot was empty (post-faint)
      to_species: string,
      forced_by: string | null       // "Volt Switch" / "U-turn" / "Eject Button" / "Roar" / null
    }
  | { type: "faint",
      side: "p1" | "p2",
      slot: string,
      species: string
    }
  | { type: "item_event",
      slot: string,
      kind: "consumed" | "knocked_off" | "tricked" | "flung" | "stolen" | "harvested" | "incinerated" | "popped",
      item: string,
      cause?: string                 // e.g. "Knock Off", "Trick", "Magician"
    };

interface SideSnapshot {
  player: string;                    // username
  active: ActivePokemonSnapshot[];   // doubles slots a, b (length 2 in VGC)
  bench: BenchPokemonSnapshot[];     // not-currently-active known team members
  faints: number;                    // cumulative faints this game
  teraUsed?: {                       // sticky once a side Tera's; absent until then
    species: string;
    teraType: string;
    onTurn: number;
  };
  sideConditions: { [id: string]: { level?: number; turnsLeft?: number } };
}

interface ActivePokemonSnapshot {
  slot: 'a' | 'b' | 'c';
  species: string;                   // e.g. "Calyrex-Shadow"
  hpPercent: number;                 // 0 – 100, one decimal
  fainted: boolean;
  status: 'brn' | 'par' | 'psn' | 'tox' | 'slp' | 'frz' | null;
  ability: string | null;            // OTS-known if isOTS, else revealed via play
  item: string | null;               // OTS-known if isOTS, else revealed via play
  revealedMoves: string[];           // PS display names — derived from events stream (excludes
                                     // moves called via Metronome/Copycat/etc.; Sleep Talk OK).
  knownMoves: string[] | null;       // OTS full moveset (4 entries) when isOTS; null in CTS
  teraType: string | null;           // OTS-known if isOTS, else revealed via team preview / Tera event
  isTerastallized: boolean;
  terastallizedAs: string | null;
  boosts: Record<string, number>;
  volatiles: {                       // only-when-active subset; only fields that apply are present
    substitute?: { hp: number };
    encoredInto?: string;            // move display name
    disabled?: string;
    taunt?: { turnsLeft: number };
    healBlock?: { turnsLeft: number };
    perishCount?: number;
    confusion?: { turnsLeft: number };
    leechSeed?: boolean;
  };
  choiceLockedInto: string | null;   // move display name when item+last-move imply Choice lock
  toxicCounter?: number;             // for Toxic damage doubling
}

interface BenchPokemonSnapshot {
  species: string;
  fainted: boolean;
}
```

### Example

```bash
curl -s -X POST http://localhost:3000/parse_log \
  -H "Content-Type: application/json" \
  --data "$(jq -c '{log: .log}' < data_scraper/data/replays/gen9vgc2026regi/<some_id>.json)" \
  | jq '.snapshots[0]'
```

Truncated example output (turn 2 of a real Reg I replay):

```json
{
  "turn": 2,
  "field": {
    "weather": null,
    "terrain": "Grassy", "terrainTurnsLeft": 4,
    "pseudoWeather": {},
    "tailwindP1": false, "tailwindP2": false
  },
  "p2": {
    "player": "VJ2511",
    "active": [
      {
        "slot": "a", "species": "Calyrex-Shadow", "hpPercent": 91, "fainted": false,
        "status": null, "ability": "unnerve", "item": "lifeorb",
        "revealedMoves": ["Psychic"],
        "teraType": null, "isTerastallized": false, "terastallizedAs": null,
        "boosts": {}, "volatiles": {}, "choiceLockedInto": null
      },
      {
        "slot": "b", "species": "Zamazenta-Crowned", "hpPercent": 54, "fainted": false,
        "status": null, "ability": "dauntlessshield", "item": null,
        "revealedMoves": ["Body Press"],
        "teraType": "Water", "isTerastallized": true, "terastallizedAs": "Water",
        "boosts": { "def": 1 }, "volatiles": {}, "choiceLockedInto": null
      }
    ],
    "bench": [...],
    "faints": 0,
    "teraUsed": { "species": "Zamazenta-Crowned", "teraType": "Water", "onTurn": 1 },
    "sideConditions": {}
  },
  "p1": { ... },
  "events": [
    { "type": "move", "attacker_slot": "p2a", "move_name": "Volt Switch",
      "called_via": null,
      "hits": [{ "defender_slot": "p1b", "outcome": "blocked", "cause": "Protect" }] },
    { "type": "switch", "side": "p2", "slot": "p2a", "from_species": "Miraidon",
      "to_species": "Terapagos", "forced_by": "Volt Switch" }
  ]
}
```

### Notes

- IDs (`ability`, `item`, `revealedMoves`) are normalised lowercase no-spaces
  (`"lifeorb"`, `"hadronengine"`). Translate to display names downstream if you
  need them.
- "Revealed" means "the @pkmn/client Battle state machine inferred this from
  the protocol stream." For VGC Open Team Sheet, you should layer the OTS data
  on top — `parse_log` only knows what the spectator sees.
- The Battle is constructed in omniscient (no specific player) mode; both sides
  are tracked symmetrically.
- Malformed individual lines are skipped silently rather than failing the whole
  parse.
- **`events` semantics** — events are forward-looking: `snapshot[N].events`
  contains everything that happened during turn N (between this snapshot and
  the next). For inference: pair `snapshot_pre = snapshots[N]`,
  `snapshot_post = snapshots[N+1]`, `events = snapshots[N].events`.
- The Python `damage_inferencer.events_to_damage_events()` helper filters
  for damage observations: `type == "move" AND called_via in {None, "Sleep
  Talk"} AND hit.outcome == "damage"`. Metronome / Copycat / Sketch /
  Snatch / Me First / Dancer / Instruct hits are excluded — those moves
  may not be in the user's actual kit and would corrupt EV bounds.
- Only direct move damage is captured under `outcome: "damage"`. `|-damage|`
  events with a `[from] …` kwarg (burn / sand / Life Orb / Rocky Helmet /
  Future Sight / etc.) are surfaced separately as `item_event` /
  `cant_move` / `faint` — never as part of a move's `hits[]`.
- Multi-hit moves (Triple Axel, Bullet Seed) emit one move event with
  multiple entries in `hits[]` (typically all the same defender_slot);
  the inferencer flattens these via `events_to_damage_events` and the
  multi-hit filter (`(attacker_slot, move_name, defender_slot)` count > 1
  per turn) drops them since `/calc` can't model `hits` properly yet.
- **`derivedRevealedMoves`** is computed from the events stream rather
  than from `@pkmn/client`'s `pokemon.moves`. A move is added to a slot's
  revealed list only when the move event has `called_via in {null, "Sleep
  Talk"}`. This avoids polluting Hatterene's / Smeargle's reconstructed
  CTS moveset with Bleakwind Storm / Boomburst / etc. that they called
  via Metronome.
- **OTS / CTS bench gating** (Bo3 only — Bo1 behavior unchanged):
  - `snapshot.p1.bench` is intersected with the **brought-set** for the
    game (computed in a one-pass pre-scan over the log for every species
    that ever appears via `|switch|` / `|drag|` / `|replace|`). At turn 1
    P1 bench shows the 4 brought minus the 2 active, even before the
    bench Pokémon switch in.
  - `snapshot.p2.bench` is intersected with the **on-field-set** (running
    — populated as P2 actually switches in). At turn 1, P2 bench is empty;
    it grows over the game as the opponent's selection is revealed.
  - Active Pokémon get their `item` / `ability` / `teraType` filled from
    OTS at turn 1 (instead of waiting for in-game reveal). This is what
    makes downstream `damage_inferencer` and `threat_matrix` calc payloads
    immediately tighter on Bo3, with **zero changes** to those modules.

## `GET /dex/move/:name`

Cheap metadata lookup for a move. The Python pipeline uses this to filter out
Status-category moves (which the calc engine can't meaningfully damage-rate)
before bothering with a full `/calc` request.

`:name` accepts either the canonical name or its `@pkmn`-style ID
(lowercase, alphanumeric only): `Electro Drift` and `electrodrift` both work.

### Response

```ts
{
  name: string,           // "Electro Drift"
  id: string,             // "electrodrift"
  type: string,           // "Electric"
  category: "Physical" | "Special" | "Status",
  basePower: number,
  accuracy: number | true, // true = always hits
  target: string,          // PS target id, e.g. "normal", "allAdjacentFoes"
  priority: number
}
```

Unknown move → `404 { "error": "..." }`.

### Example

```bash
curl http://localhost:3000/dex/move/electrodrift
# → {"name":"Electro Drift","id":"electrodrift","type":"Electric","category":"Special","basePower":100,"accuracy":100,"target":"normal","priority":0}
```

## Design notes

- **`teraType` vs. `isTera`** — `teraType` is always required because under
  Open Team Sheet rules the Tera type is public knowledge from turn 1. `isTera`
  defaults to `false` so the same input shape can answer both *"max damage now"*
  (`isTera: false`) and *"max damage if they Tera"* (`isTera: true`).
- **Immunity short-circuit** — `@smogon/calc` throws an internal assertion
  inside `fullDesc()` / `kochance()` when max damage is `0` (e.g. Dragon move
  vs. Tera Fairy). The wrapper detects this and returns a clean 0-damage
  response instead.
- **Percentages vs. flat HP for `currentHP`** — Strings ending in `%` are
  parsed as percentages; numbers are treated as flat HP. This avoids the
  ambiguity for low-HP species where a value like `40` could plausibly be
  either.
- **VGC defaults** — `level` defaults to `50` and `gameType` defaults to
  `"Doubles"` since this service is purpose-built for VGC. Override either if
  you ever want singles or level-100.

## Files

```
calc_microservice/
├── package.json
├── tsconfig.json
└── src/
    ├── server.ts      # Express setup, /calc + /parse_log + /dex/move + /health
    ├── calc.ts        # buildPokemon / buildField / runCalc
    ├── parse_log.ts   # Battle state machine + per-turn snapshot extraction + OTS gating
    ├── ots.ts         # @pkmn/sets Teams.unpackTeam wrapper for |showteam| decoding
    ├── dex.ts         # @pkmn/dex Move lookup
    └── types.ts       # CalcRequest / CalcResponse / PokemonInput / FieldInput
```

Dependencies:

- `@smogon/calc` — damage formulas (powers `/calc`).
- `@pkmn/protocol`, `@pkmn/client`, `@pkmn/dex`, `@pkmn/data` — official
  Showdown protocol parser + battle state machine + Pokédex (powers `/parse_log`).
- `@pkmn/sets` — `Teams.unpackTeam` decoder for `|showteam|` packed payloads
  (powers OTS team-sheet extraction in `/parse_log`).
- `express` — HTTP server.
