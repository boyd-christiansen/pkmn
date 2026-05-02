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
  "snapshots": TurnSnapshot[]
}
```

One snapshot is emitted at the **start of every turn** (i.e. immediately after
the `|turn|N` protocol line is processed). The state reflects everything that
happened up to and including the end of turn `N-1`.

```ts
interface TurnSnapshot {
  turn: number;
  field: {
    weather: string | null;          // "Sun", "Rain", "Sand", "Snow", "Electric Terrain", etc.
    terrain: string | null;
    tailwindP1: boolean;
    tailwindP2: boolean;
  };
  p1: SideSnapshot;
  p2: SideSnapshot;
  actionLog: DamageEvent[];          // damage events that occurred DURING this turn
}

interface DamageEvent {
  attacker_slot: string;             // "p1a" | "p1b" | "p2a" | "p2b"
  defender_slot: string;
  move_name: string;                 // PS display name, e.g. "Wood Hammer"
  hp_before_pct: number;             // 0–100, defender HP just BEFORE the hit
  hp_after_pct: number;              // 0–100, after the hit (0 if KO'd)
  is_crit: boolean;
  is_ko: boolean;
}

interface SideSnapshot {
  player: string;                    // username
  active: ActivePokemonSnapshot[];   // doubles slots a, b (length 2 in VGC)
  bench: BenchPokemonSnapshot[];     // not-currently-active known team members
}

interface ActivePokemonSnapshot {
  slot: 'a' | 'b' | 'c';
  species: string;                   // e.g. "Calyrex-Shadow"
  hpPercent: number;                 // 0 – 100, one decimal
  fainted: boolean;
  status: 'brn' | 'par' | 'psn' | 'tox' | 'slp' | 'frz' | null;
  ability: string | null;            // revealed ability ID, null if not yet seen
  item: string | null;               // revealed item ID, null if not yet seen
  revealedMoves: string[];           // move IDs the Pokémon has used so far
  teraType: string | null;           // intrinsic Tera type (revealed via team preview / Tera event)
  isTerastallized: boolean;
  terastallizedAs: string | null;    // type they Tera'd into, if applicable
  boosts: Record<string, number>;    // active stat-stage boosts: { atk: 2, def: -1, ... }
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
  "field": { "weather": null, "terrain": null, "tailwindP1": false, "tailwindP2": false },
  "p2": {
    "player": "VJ2511",
    "active": [
      {
        "slot": "a", "species": "Calyrex-Shadow", "hpPercent": 91, "fainted": false,
        "status": null, "ability": "unnerve", "item": "lifeorb",
        "revealedMoves": ["psychic"],
        "teraType": null, "isTerastallized": false, "terastallizedAs": null, "boosts": {}
      },
      {
        "slot": "b", "species": "Zamazenta-Crowned", "hpPercent": 54, "fainted": false,
        "status": null, "ability": "dauntlessshield", "item": null,
        "revealedMoves": ["bodypress"],
        "teraType": "Water", "isTerastallized": true, "terastallizedAs": "Water",
        "boosts": { "def": 1 }
      }
    ],
    "bench": [...]
  },
  "p1": { ... }
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
- **`actionLog` semantics** — events are forward-looking: `snapshot[N].actionLog`
  contains damage events that happened during turn N (between this snapshot and
  the next). For inference: pair `snapshot_pre = snapshots[N]`,
  `snapshot_post = snapshots[N+1]`, `events = snapshots[N].actionLog`.
- Only direct move damage is captured. `|-damage|` events with a `[from] …`
  kwarg (burn / sand / Life Orb / Rocky Helmet / Future Sight / etc.) are
  filtered out — incidental/end-of-turn damage isn't useful for EV inference.
- Multi-hit moves (Triple Axel, Bullet Seed) emit one `DamageEvent` per hit;
  downstream consumers (the Python inferencer) detect them by counting same
  `(attacker_slot, move_name, defender_slot)` tuples per turn.

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
    ├── parse_log.ts   # Battle state machine + per-turn snapshot extraction
    ├── dex.ts         # @pkmn/dex Move lookup
    └── types.ts       # CalcRequest / CalcResponse / PokemonInput / FieldInput
```

Dependencies:

- `@smogon/calc` — damage formulas (powers `/calc`).
- `@pkmn/protocol`, `@pkmn/client`, `@pkmn/dex`, `@pkmn/data` — official
  Showdown protocol parser + battle state machine + Pokédex (powers `/parse_log`).
- `express` — HTTP server.
