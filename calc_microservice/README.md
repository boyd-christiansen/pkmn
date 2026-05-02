# calc_microservice

Express + TypeScript HTTP wrapper around [`@smogon/calc`](https://www.npmjs.com/package/@smogon/calc).
Exposes a single `POST /calc` endpoint that turns an attacker / defender / move /
field payload into a deterministic damage range.

This is the only piece of the pipeline that knows how to compute damage. Every
other component (Python pipeline, teacher LLM tool calls) goes through this
HTTP boundary, so the calc engine can be swapped or upgraded independently.

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
  move: string,
  field?: FieldInput
}
```

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
    ├── server.ts   # Express setup, /calc + /health, error wrapping
    ├── calc.ts     # buildPokemon / buildField / runCalc
    └── types.ts    # CalcRequest / CalcResponse / PokemonInput / FieldInput
```
