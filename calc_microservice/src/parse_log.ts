import { Battle } from '@pkmn/client';
import type { Pokemon, Side } from '@pkmn/client';
import { Generations } from '@pkmn/data';
import { Dex } from '@pkmn/dex';
import { Protocol } from '@pkmn/protocol';
import { decodeShowteam, type OtsPokemonSet } from './ots.js';

const GENS = new Generations(Dex as any);

// =============================================================================
// Types — events log
// =============================================================================

export type MoveOutcome =
  | 'damage'
  | 'miss'
  | 'blocked'
  | 'immune'
  | 'no_effect'
  | 'fail';

export interface MoveHit {
  defender_slot: string;
  outcome: MoveOutcome;
  hp_before_pct?: number;
  hp_after_pct?: number;
  is_crit?: boolean;
  is_ko?: boolean;
  cause?: string;          // 'Protect', 'Wide Guard', 'Substitute', etc. (for blocked/immune)
}

export interface MoveEvent {
  type: 'move';
  attacker_slot: string;
  move_name: string;
  called_via: string | null;   // null for direct use; else 'Sleep Talk' / 'Metronome' / 'Copycat' / etc.
  target_slots: string[];      // intended target slot(s), e.g. ['p1a'] or ['p1a','p1b']; [] for self/field/no-target
  is_spread: boolean;          // true when the move hit a spread of targets ([spread] kwarg)
  hits: MoveHit[];
}

export interface CantMoveEvent {
  type: 'cant_move';
  slot: string;
  reason: string;              // 'slp' | 'par' | 'frz' | 'flinch' | 'truant' | 'disable' | 'imprison' | 'taunt' | 'healblock' | 'recharge' | 'newlySwitchedIn' | 'nopp' | other
  attempted_move?: string;
}

export interface TeraEvent {
  type: 'tera';
  side: 'p1' | 'p2';
  slot: string;
  species: string;
  to_type: string;
}

export interface SwitchEvent {
  type: 'switch';
  side: 'p1' | 'p2';
  slot: string;
  from_species: string | null;     // null for first switch-in this game on that slot
  to_species: string;
  forced_by: string | null;        // 'Volt Switch' | 'U-turn' | 'Roar' | 'Whirlwind' | 'Dragon Tail' | 'Eject Button' | 'Red Card' | 'Eject Pack' | null
}

export interface FaintEvent {
  type: 'faint';
  side: 'p1' | 'p2';
  slot: string;
  species: string;
}

export type ItemEventKind =
  | 'consumed'
  | 'knocked_off'
  | 'tricked'
  | 'flung'
  | 'stolen'
  | 'harvested'
  | 'incinerated'
  | 'popped';

export interface ItemEvent {
  type: 'item_event';
  slot: string;
  kind: ItemEventKind;
  item: string;
  cause?: string;
}

export type TurnEvent = MoveEvent | CantMoveEvent | TeraEvent | SwitchEvent | FaintEvent | ItemEvent;

// Callers that DON'T deposit the called move into the user's actual moveset.
// Sleep Talk is excluded because it can only call moves the user already knows.
// Mimic is excluded because it permanently overwrites a slot in-battle, so
// subsequent uses of the copied move legitimately are "in the kit".
const NON_OWN_CALLERS: ReadonlySet<string> = new Set([
  'Metronome', 'Copycat', 'Sketch', 'Snatch', 'Me First', 'Dancer', 'Instruct',
  'Mirror Move', 'Assist', 'Nature Power',
]);

// =============================================================================
// Types — snapshot extensions
// =============================================================================

export interface Volatiles {
  substitute?: { hp: number };
  encored?: boolean;
  disabled?: boolean;
  taunt?: { turnsLeft?: number };
  healBlock?: { turnsLeft?: number };
  perishCount?: number;            // 1, 2, or 3
  confusion?: { turnsLeft?: number };
  leechSeed?: boolean;
  // (Add more as needed; only-when-active emits.)
}

export interface ActivePokemonSnapshot {
  slot: 'a' | 'b' | 'c';
  species: string;
  hpPercent: number;
  fainted: boolean;
  status: string | null;
  ability: string | null;
  item: string | null;
  revealedMoves: string[];           // DERIVED — own moves only (Sleep Talk allowed; non-own callers filtered)
  knownMoves: string[] | null;       // OTS-known full moveset (4 entries) when isOTS, else null
  teraType: string | null;
  isTerastallized: boolean;
  terastallizedAs: string | null;
  boosts: Record<string, number>;
  volatiles: Volatiles;              // only-when-set keys
  choiceLockedInto: string | null;   // move id, when item is Choice Scarf/Specs/Band AND lastMove set
  toxicCounter?: number;
}

export interface BenchPokemonSnapshot {
  species: string;
  fainted: boolean;
  hpPercent: number;           // last-known HP% (100 for a brought mon never sent in)
  status: string | null;       // 'brn' | 'par' | 'slp' | 'frz' | 'psn' | 'tox' | null
}

export interface SideConditionState {
  level?: number;        // for spikes (1-3) and toxic spikes (1-2)
  turnsLeft?: number;    // for screens / tailwind / safeguard / mist
}

export interface SideSnapshot {
  player: string;
  active: ActivePokemonSnapshot[];
  // bench renders the full brought-set minus current actives, regardless of
  // whether each species has been on field yet. Each player at team preview
  // already knows their own selection; the chronological gating for the
  // opponent's perspective is applied downstream (in Python's prompt
  // formatter) using `seenSpecies` below.
  bench: BenchPokemonSnapshot[];
  // Chronological set of species that have been active on this side at any
  // turn ≤ current. Empty at turn 1 except for the starters. Grows with each
  // |switch| / |drag| / |replace|. Used by the prompt formatter to gate the
  // OPPONENT's bench display (the spectator only learns the opp's brought
  // selection as they actually appear).
  seenSpecies: string[];
  faints: number;                                        // cumulative this game
  teraUsed?: { species: string; teraType: string; onTurn: number };
  sideConditions: { [id: string]: SideConditionState };
}

export interface FieldSnapshot {
  weather: string | null;
  weatherTurnsLeft?: number;
  terrain: string | null;
  terrainTurnsLeft?: number;
  pseudoWeather: { [id: string]: { turnsLeft: number } };
  // backward-compat booleans (still emitted for older consumers):
  tailwindP1: boolean;
  tailwindP2: boolean;
  tailwindP1TurnsLeft?: number;
  tailwindP2TurnsLeft?: number;
}

export interface TurnSnapshot {
  turn: number;
  field: FieldSnapshot;
  p1: SideSnapshot;
  p2: SideSnapshot;
  events: TurnEvent[];               // RENAMED from actionLog. See TurnEvent union above.
}

export interface TeamSheets {
  p1: OtsPokemonSet[];
  p2: OtsPokemonSet[];
}

export interface ParseLogResult {
  snapshots: TurnSnapshot[];
  teamSheets: TeamSheets | null;     // null in CTS games
  winner: 'p1' | 'p2' | null;
}

// =============================================================================
// Helpers — slot / side / species extraction
// =============================================================================

const SLOT_LETTERS: Array<'a' | 'b' | 'c'> = ['a', 'b', 'c'];

function speciesKey(species: string): string {
  return species.toLowerCase().replace(/[^a-z0-9]/g, '');
}

function extractSlot(ident: string | undefined): string {
  if (!ident) return '';
  const m = ident.match(/^(p[12][a-c])/);
  return m?.[1] ?? '';
}

function extractSide(ident: string | undefined): 'p1' | 'p2' | null {
  if (!ident) return null;
  if (ident.startsWith('p1')) return 'p1';
  if (ident.startsWith('p2')) return 'p2';
  return null;
}

function extractSpeciesFromDetails(details: string | undefined): string | null {
  if (!details) return null;
  const sp = details.split(',')[0]?.trim();
  return sp || null;
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

function hpPct(p: Pokemon | null | undefined): number {
  if (!p) return 0;
  const maxhp = p.maxhp || 1;
  return (p.hp / maxhp) * 100;
}

/** Parse the `[from]` kwarg into `{ type, name }`. Examples:
 *    "move: Sleep Talk"   → { type: "move", name: "Sleep Talk" }
 *    "ability: Intimidate" → { type: "ability", name: "Intimidate" }
 *    "Recoil"             → { type: null, name: "Recoil" }
 */
function parseFromKwarg(raw: unknown): { type: string | null; name: string } | null {
  if (typeof raw !== 'string' || !raw) return null;
  const m = raw.match(/^(move|ability|item):\s*(.+)$/);
  if (m) return { type: m[1] || null, name: m[2] || '' };
  return { type: null, name: raw };
}

/** Parse "move: Protect" / "ability: Sturdy" / "Substitute" into the cause name. */
function parseEffectName(raw: string | undefined): string | undefined {
  if (!raw) return undefined;
  const m = raw.match(/^(?:move|ability|item):\s*(.+)$/);
  return (m?.[1] ?? raw).trim() || undefined;
}

// =============================================================================
// Snapshot builders
// =============================================================================

function buildSheetIndex(sheets: OtsPokemonSet[] | null): Map<string, OtsPokemonSet> {
  const m = new Map<string, OtsPokemonSet>();
  if (!sheets) return m;
  for (const s of sheets) m.set(speciesKey(s.species), s);
  return m;
}

interface OtsContext {
  isOTS: boolean;
  p1Sheets: OtsPokemonSet[] | null;
  p2Sheets: OtsPokemonSet[] | null;
  p1SheetIndex: Map<string, OtsPokemonSet>;
  p2SheetIndex: Map<string, OtsPokemonSet>;
  p1Brought: Set<string>;
  p2Brought: Set<string>;
}

function snapshotVolatiles(p: Pokemon): Volatiles {
  const v: any = p.volatiles ?? {};
  const out: Volatiles = {};
  if (v['substitute']) out.substitute = { hp: v['substitute']?.level ?? 0 };
  if (v['encore']) out.encored = true;
  if (v['disable']) out.disabled = true;
  if (v['taunt']) out.taunt = { turnsLeft: v['taunt']?.duration };
  if (v['healblock']) out.healBlock = { turnsLeft: v['healblock']?.duration };
  if (v['perish3']) out.perishCount = 3;
  else if (v['perish2']) out.perishCount = 2;
  else if (v['perish1']) out.perishCount = 1;
  if (v['confusion']) out.confusion = { turnsLeft: v['confusion']?.duration };
  if (v['leechseed']) out.leechSeed = true;
  return out;
}

function snapshotChoiceLock(p: Pokemon, resolvedItem: string | null): string | null {
  // The item lookup must use the OTS-resolved item, not @pkmn/client's raw
  // p.item — in Bo3 OTS, p.item only populates after a `|-item|` reveal,
  // but Choice Scarf/Specs/Band are always known from the team sheet.
  if (!resolvedItem) return null;
  const item = resolvedItem.toLowerCase();
  if (!item.includes('choice')) return null;
  if (!p.lastMove) return null;
  const lastMoveId = String(p.lastMove);
  // Normalize move ID → display name via the dex (e.g. "boltstrike" → "Bolt Strike").
  const moveData = GENS.get(9).moves.get(lastMoveId);
  return moveData?.name ?? lastMoveId;
}

function snapshotActive(
  p: Pokemon,
  slotIndex: number,
  ots: OtsContext,
  sideIsP1: boolean,
  derivedMovesForSide: Map<string, Set<string>>,
): ActivePokemonSnapshot {
  const maxhp = p.maxhp || 1;
  const hpPercent = Math.round((p.hp / maxhp) * 1000) / 10;
  const species = p.speciesForme || p.baseSpeciesForme || p.name;

  const sheetIndex = sideIsP1 ? ots.p1SheetIndex : ots.p2SheetIndex;
  const sheet = ots.isOTS ? sheetIndex.get(speciesKey(species)) : undefined;

  const protocolItem = p.item ? String(p.item) : null;
  const protocolAbility = p.ability ? String(p.ability) : null;
  const protocolTera = p.teraType ?? null;

  const item = protocolItem ?? (sheet?.item || null);
  const ability = protocolAbility ?? (sheet?.ability || null);
  const teraType = protocolTera ?? sheet?.teraType ?? null;

  const knownMoves =
    ots.isOTS && sheet
      ? sheet.moves.slice(0, 4).map((m) => m || '')
      : null;

  // Derived revealedMoves: own moves only. The slot key here is the SLOT
  // identifier (e.g. "p1a") that the events log records moves under.
  const slotKey = `${sideIsP1 ? 'p1' : 'p2'}${SLOT_LETTERS[slotIndex] ?? 'a'}`;
  const slotMoves = derivedMovesForSide.get(slotKey);
  const revealedMoves = slotMoves ? Array.from(slotMoves) : [];

  const toxicCounter = (p as any).statusState?.toxicTurns;

  return {
    slot: SLOT_LETTERS[slotIndex] ?? 'a',
    species,
    hpPercent,
    fainted: !!p.fainted,
    status: p.status ?? null,
    ability,
    item,
    revealedMoves,
    knownMoves,
    teraType,
    isTerastallized: !!p.isTerastallized,
    terastallizedAs: p.terastallized ?? null,
    boosts: { ...p.boosts } as Record<string, number>,
    volatiles: snapshotVolatiles(p),
    choiceLockedInto: snapshotChoiceLock(p, item),
    ...(toxicCounter && toxicCounter > 0 ? { toxicCounter } : {}),
  };
}

function snapshotBench(p: Pokemon): BenchPokemonSnapshot {
  const maxhp = p.maxhp || 1;
  const fainted = !!p.fainted;
  // @pkmn/client leaves hp=0 for a brought mon never sent in (its HP isn't
  // known until it's been on field). A genuinely 0-HP mon is fainted, so
  // `hp===0 && !fainted` means "never on field" → untouched, full HP.
  const hpPercent = (p.hp === 0 && !fainted) ? 100 : Math.round((p.hp / maxhp) * 1000) / 10;
  return {
    species: p.speciesForme || p.baseSpeciesForme || p.name,
    fainted,
    hpPercent,
    status: fainted ? null : (p.status ?? null),
  };
}

function snapshotSideConditions(side: Side): { [id: string]: SideConditionState } {
  const out: { [id: string]: SideConditionState } = {};
  const conds = side.sideConditions ?? {};
  for (const [id, c] of Object.entries(conds)) {
    const cAny = c as any;
    const entry: SideConditionState = {};
    if (cAny.level && cAny.level > 0) entry.level = cAny.level;
    if (cAny.minDuration && cAny.minDuration > 0) entry.turnsLeft = cAny.minDuration;
    if (Object.keys(entry).length === 0) continue;
    out[id] = entry;
  }
  return out;
}

function snapshotPseudoWeather(battle: Battle): { [id: string]: { turnsLeft: number } } {
  const out: { [id: string]: { turnsLeft: number } } = {};
  const pw = (battle.field as any).pseudoWeather ?? {};
  for (const [id, c] of Object.entries(pw)) {
    const cAny = c as any;
    out[id] = { turnsLeft: cAny.minDuration ?? 0 };
  }
  return out;
}

function snapshotSide(
  side: Side,
  ots: OtsContext,
  sideIsP1: boolean,
  onFieldSet: Set<string>,
  derivedMovesForSide: Map<string, Set<string>>,
  teraUsed: { species: string; teraType: string; onTurn: number } | null,
): SideSnapshot {
  const activeSet = new Set(side.active.filter((p): p is Pokemon => p !== null));
  // Active species match keys must cover Tera transformations so that the
  // pre-Tera brought-set entry (e.g. "Terapagos") is recognized as the same
  // mon as the active's transformed form (e.g. "Terapagos-Stellar"). For
  // each active, we add the speciesForme + baseSpeciesForme + name + the
  // Tera-suffix-stripped variant.
  const activeSpeciesKeys = new Set<string>();
  for (const p of activeSet) {
    const candidates: string[] = [];
    if (p.speciesForme) candidates.push(p.speciesForme);
    if (p.baseSpeciesForme) candidates.push(p.baseSpeciesForme);
    if (p.name) candidates.push(p.name);
    for (const c of candidates) {
      activeSpeciesKeys.add(speciesKey(c));
      // Tera form variants — strip the in-battle suffixes that Showdown
      // appends after Tera Shift / Terastallize.
      const stripped = c
        .replace(/-Stellar$/, '')
        .replace(/-Terastal$/, '')
        .replace(/-Tera$/, '');
      if (stripped !== c) activeSpeciesKeys.add(speciesKey(stripped));
    }
  }

  const active: ActivePokemonSnapshot[] = [];
  side.active.forEach((p, idx) => {
    if (p) active.push(snapshotActive(p, idx, ots, sideIsP1, derivedMovesForSide));
  });

  // Bench: always the full brought-set (pre-scanned) minus current actives.
  // Each player knows their own brought selection at team preview, so this
  // is the right "what the player would see" view. The opponent's
  // perspective (only knowing what's been switched in so far) is applied
  // by the prompt formatter using `seenSpecies` below.
  const broughtSet = sideIsP1 ? ots.p1Brought : ots.p2Brought;

  // Look up each brought species' fainted state from @pkmn/client's tracked
  // team if available. Species in broughtSet but not yet in @pkmn/client's
  // team can't be fainted (haven't been on field), so default to false.
  const teamBySpecies = new Map<string, Pokemon>();
  for (const p of side.team) {
    const sp = speciesKey(p.speciesForme || p.baseSpeciesForme || p.name);
    teamBySpecies.set(sp, p);
  }

  const bench: BenchPokemonSnapshot[] = [];
  for (const sp of broughtSet) {
    if (activeSpeciesKeys.has(sp)) continue;
    const pkm = teamBySpecies.get(sp);
    if (pkm) {
      bench.push(snapshotBench(pkm));
    } else {
      // Brought but not yet switched in — full HP (never been on field).
      bench.push({ species: prettySpeciesFromKey(sp, ots, sideIsP1), fainted: false, hpPercent: 100, status: null });
    }
  }

  // Chronological "ever on field" set through the current snapshot.
  const seenSpecies = [...onFieldSet];

  const faints = side.team.filter((p) => p.fainted).length;

  return {
    player: side.name || `p${side.n + 1}`,
    active,
    bench,
    seenSpecies,
    faints,
    ...(teraUsed ? { teraUsed } : {}),
    sideConditions: snapshotSideConditions(side),
  };
}

/** Look up a display-name species from a normalized species key.
 *  Tries the OTS team sheet first (best display), falls back to the key. */
function prettySpeciesFromKey(spKey: string, ots: OtsContext, sideIsP1: boolean): string {
  const sheets = sideIsP1 ? ots.p1Sheets : ots.p2Sheets;
  if (sheets) {
    for (const s of sheets) {
      if (speciesKey(s.species) === spKey) return s.species;
    }
  }
  // No sheet (Bo1 CTS): resolve the @pkmn key → display name via the dex, so a
  // never-sent forme restricted renders as "Zamazenta-Crowned" not the raw id
  // "zamazentacrowned" — the latter dupes against the active display form when
  // reconstruct_p1_team aggregates by species string. Fall back to the raw key
  // only if the dex has no match.
  const sp = Dex.species.get(spKey);
  if (sp && sp.exists && sp.name) return sp.name;
  return spKey;
}

function snapshotBattle(
  battle: Battle,
  ots: OtsContext,
  onFieldP1: Set<string>,
  onFieldP2: Set<string>,
  derivedMovesP1: Map<string, Set<string>>,
  derivedMovesP2: Map<string, Set<string>>,
  teraUsedP1: { species: string; teraType: string; onTurn: number } | null,
  teraUsedP2: { species: string; teraType: string; onTurn: number } | null,
): Omit<TurnSnapshot, 'events'> {
  const sideTailwind = (side: Side): { active: boolean; turnsLeft?: number } => {
    const conds = side.sideConditions ?? {};
    for (const [id, c] of Object.entries(conds)) {
      if (id.toLowerCase() === 'tailwind') {
        const cAny = c as any;
        return { active: true, turnsLeft: cAny.minDuration };
      }
    }
    return { active: false };
  };

  const tw1 = sideTailwind(battle.p1);
  const tw2 = sideTailwind(battle.p2);

  const weatherState = (battle.field as any).weatherState as any;
  const terrainState = (battle.field as any).terrainState as any;

  return {
    turn: battle.turn,
    field: {
      weather: battle.field.weather ?? null,
      ...(weatherState?.minDuration ? { weatherTurnsLeft: weatherState.minDuration } : {}),
      terrain: battle.field.terrain ?? null,
      ...(terrainState?.minDuration ? { terrainTurnsLeft: terrainState.minDuration } : {}),
      pseudoWeather: snapshotPseudoWeather(battle),
      tailwindP1: tw1.active,
      tailwindP2: tw2.active,
      ...(tw1.turnsLeft !== undefined ? { tailwindP1TurnsLeft: tw1.turnsLeft } : {}),
      ...(tw2.turnsLeft !== undefined ? { tailwindP2TurnsLeft: tw2.turnsLeft } : {}),
    },
    p1: snapshotSide(battle.p1, ots, true, onFieldP1, derivedMovesP1, teraUsedP1),
    p2: snapshotSide(battle.p2, ots, false, onFieldP2, derivedMovesP2, teraUsedP2),
  };
}

// =============================================================================
// Pre-pass: OTS sheets + brought sets (unchanged)
// =============================================================================

function preprocessLog(log: string): {
  p1Sheets: OtsPokemonSet[] | null;
  p2Sheets: OtsPokemonSet[] | null;
  p1Brought: Set<string>;
  p2Brought: Set<string>;
} {
  let p1Sheets: OtsPokemonSet[] | null = null;
  let p2Sheets: OtsPokemonSet[] | null = null;
  const p1Brought = new Set<string>();
  const p2Brought = new Set<string>();

  for (const rawLine of log.split('\n')) {
    const line = rawLine.replace(/\r$/, '');
    if (!line || line[0] !== '|') continue;
    const parts = line.split('|');
    const tag = parts[1];
    if (!tag) continue;

    if (tag === 'showteam') {
      const side = parts[2];
      const packed = parts.slice(3).join('|');
      if (!packed) continue;
      const sheets = decodeShowteam(packed);
      if (!sheets) continue;
      if (side === 'p1') p1Sheets = sheets;
      else if (side === 'p2') p2Sheets = sheets;
    } else if (tag === 'switch' || tag === 'drag' || tag === 'replace') {
      const ident = parts[2];
      const details = parts[3];
      const side = extractSide(ident);
      const species = extractSpeciesFromDetails(details);
      if (!side || !species) continue;
      const target = side === 'p1' ? p1Brought : p2Brought;
      target.add(speciesKey(species));
    }
  }

  return { p1Sheets, p2Sheets, p1Brought, p2Brought };
}

// =============================================================================
// Main parser
// =============================================================================

export function parseLog(log: string): ParseLogResult {
  if (typeof log !== 'string') {
    throw new Error('log must be a string');
  }

  const pre = preprocessLog(log);
  const isOTS = pre.p1Sheets !== null || pre.p2Sheets !== null;
  const ots: OtsContext = {
    isOTS,
    p1Sheets: pre.p1Sheets,
    p2Sheets: pre.p2Sheets,
    p1SheetIndex: buildSheetIndex(pre.p1Sheets),
    p2SheetIndex: buildSheetIndex(pre.p2Sheets),
    p1Brought: pre.p1Brought,
    p2Brought: pre.p2Brought,
  };

  const battle = new Battle(GENS);
  const snapshots: TurnSnapshot[] = [];
  const seenTurns = new Set<number>();

  // Running on-field set per side
  const onFieldP1 = new Set<string>();
  const onFieldP2 = new Set<string>();

  // Derived revealedMoves per slot — accumulator for own moves only.
  // Map keyed by slot id ("p1a", "p1b", "p2a", "p2b"). We replace the
  // slot's set whenever a new mon switches in (since revealedMoves are
  // per-Pokémon, not per-slot — but slot is what the events log uses
  // for attribution, and switches reset which mon occupies the slot).
  // Concretely: we track moves under slot-key; when a switch happens we
  // clear that slot's set so the new mon starts fresh.
  const derivedMovesP1 = new Map<string, Set<string>>();
  const derivedMovesP2 = new Map<string, Set<string>>();
  const getDerivedMoves = (slot: string): Set<string> => {
    const map = slot.startsWith('p1') ? derivedMovesP1 : derivedMovesP2;
    let s = map.get(slot);
    if (!s) { s = new Set<string>(); map.set(slot, s); }
    return s;
  };

  // Tera-used per side, sticky once set for the rest of this game.
  let teraUsedP1: { species: string; teraType: string; onTurn: number } | null = null;
  let teraUsedP2: { species: string; teraType: string; onTurn: number } | null = null;

  // Per-game winner
  let winnerName: string | null = null;

  // Event-stream state
  let currentMove: MoveEvent | null = null;
  const pendingCrits = new Set<string>();   // target slots flagged by |-crit| awaiting next |-damage|
  let currentTurnEvents: TurnEvent[] = [];
  // Side-effect events that occur while a move is being accumulated
  // (Focus Sash consumed, faint, item knocked off). Pushed to the events
  // list after the move finalizes so they appear AFTER the move in stream order.
  let pendingPostMoveEvents: TurnEvent[] = [];

  const emitOrQueueEvent = (ev: TurnEvent) => {
    if (currentMove) pendingPostMoveEvents.push(ev);
    else currentTurnEvents.push(ev);
  };

  const finalizeCurrentMove = () => {
    if (currentMove) {
      currentTurnEvents.push(currentMove);
      currentMove = null;
    }
    if (pendingPostMoveEvents.length > 0) {
      currentTurnEvents.push(...pendingPostMoveEvents);
      pendingPostMoveEvents = [];
    }
    pendingCrits.clear();
  };

  const flushTurnEvents = () => {
    finalizeCurrentMove();
    if (snapshots.length > 0 && currentTurnEvents.length > 0) {
      const last = snapshots[snapshots.length - 1]!;
      last.events = [...last.events, ...currentTurnEvents];
    }
    currentTurnEvents = [];
  };

  for (const rawLine of log.split('\n')) {
    const line = rawLine.replace(/\r$/, '');
    if (!line || line[0] !== '|') continue;

    let parsed;
    try {
      parsed = Protocol.parseBattleLine(line);
    } catch {
      continue;
    }
    if (!parsed) continue;

    const argName = parsed.args[0] as string;
    const kwArgs = (parsed.kwArgs ?? {}) as Record<string, unknown>;

    // --- pre-apply: capture defender HP before |-damage| takes effect ----------
    let dmgCapture: { defenderSlot: string; targetIdent: string; hpBefore: number } | null = null;
    if (argName === '-damage' && !('from' in kwArgs) && currentMove) {
      const targetIdent = parsed.args[1] as string;
      const target = battle.getPokemon(targetIdent as any);
      if (target) {
        dmgCapture = {
          defenderSlot: extractSlot(targetIdent),
          targetIdent,
          hpBefore: hpPct(target),
        };
      }
    }

    // --- pre-apply: capture from_species for switches --------------------------
    let switchCapture: { fromSpecies: string | null } | null = null;
    if (argName === 'switch' || argName === 'drag' || argName === 'replace') {
      const targetIdent = parsed.args[1] as string;
      const slotIdx = (() => {
        const letter = (targetIdent.match(/^p[12]([a-c])/)?.[1] ?? 'a') as 'a' | 'b' | 'c';
        return SLOT_LETTERS.indexOf(letter);
      })();
      const sideObj = targetIdent.startsWith('p1') ? battle.p1 : battle.p2;
      const outgoing = sideObj.active[slotIdx] ?? null;
      switchCapture = {
        fromSpecies: outgoing ? outgoing.speciesForme || outgoing.baseSpeciesForme || outgoing.name : null,
      };
    }

    // --- apply --------------------------------------------------------------
    try {
      battle.add(parsed.args, parsed.kwArgs as any);
    } catch {
      continue;
    }

    // --- post-apply ----------------------------------------------------------
    switch (argName) {
      case 'move': {
        finalizeCurrentMove();
        const attackerSlot = extractSlot(parsed.args[1] as string);
        const moveName = String(parsed.args[2] ?? '');
        const fromInfo = parseFromKwarg(kwArgs.from);
        const calledVia = (fromInfo && fromInfo.type === 'move') ? fromInfo.name : null;
        // Target(s): protocol is `|move|attacker|move|target|[spread] slots`.
        // Single-target → args[3] is the target ident; spread → kwArgs.spread
        // is a comma list of affected slots (already in slot form).
        const toSlot = (s: string): string => {
          const t = s.trim();
          return /^p[12][a-c]$/.test(t) ? t : (extractSlot(t) || '');
        };
        let targetSlots: string[] = [];
        let isSpread = false;
        if (kwArgs.spread !== undefined) {
          isSpread = true;
          targetSlots = String(kwArgs.spread || '')
            .split(',').map(toSlot).filter(Boolean);
        } else {
          const tgt = parsed.args[3] as string | undefined;
          if (tgt) {
            const s = toSlot(tgt);
            if (s) targetSlots = [s];
          }
        }
        currentMove = {
          type: 'move',
          attacker_slot: attackerSlot,
          move_name: moveName,
          called_via: calledVia,
          target_slots: targetSlots,
          is_spread: isSpread,
          hits: [],
        };
        // Add to derived revealedMoves only if it's an own move.
        if (calledVia === null || calledVia === 'Sleep Talk') {
          if (attackerSlot) {
            getDerivedMoves(attackerSlot).add(moveName);
          }
        }
        break;
      }

      case '-crit': {
        const targetSlot = extractSlot(parsed.args[1] as string);
        if (targetSlot) pendingCrits.add(targetSlot);
        break;
      }

      case '-damage': {
        if (dmgCapture && currentMove) {
          const target = battle.getPokemon(dmgCapture.targetIdent as any);
          const hpAfter = hpPct(target);
          const isKo = hpAfter <= 0 || !!target?.fainted;
          currentMove.hits.push({
            defender_slot: dmgCapture.defenderSlot,
            outcome: 'damage',
            hp_before_pct: round1(dmgCapture.hpBefore),
            hp_after_pct: round1(hpAfter),
            is_crit: pendingCrits.has(dmgCapture.defenderSlot),
            is_ko: isKo,
          });
        }
        break;
      }

      case '-miss': {
        if (currentMove) {
          const targetSlot = extractSlot(parsed.args[2] as string);
          currentMove.hits.push({
            defender_slot: targetSlot,
            outcome: 'miss',
          });
        }
        break;
      }

      case '-block': {
        if (currentMove) {
          const targetSlot = extractSlot(parsed.args[1] as string);
          const cause = parseEffectName(parsed.args[2] as string | undefined);
          currentMove.hits.push({
            defender_slot: targetSlot,
            outcome: 'blocked',
            cause,
          });
        }
        break;
      }

      case '-immune': {
        if (currentMove) {
          const targetSlot = extractSlot(parsed.args[1] as string);
          currentMove.hits.push({
            defender_slot: targetSlot,
            outcome: 'immune',
          });
        }
        break;
      }

      case '-fail': {
        if (currentMove) {
          const targetSlot = extractSlot(parsed.args[1] as string);
          currentMove.hits.push({
            defender_slot: targetSlot,
            outcome: 'fail',
          });
        }
        break;
      }

      case 'cant': {
        finalizeCurrentMove();
        const slot = extractSlot(parsed.args[1] as string);
        const reason = String(parsed.args[2] ?? 'unknown');
        const attempted = parsed.args[3] as string | undefined;
        currentTurnEvents.push({
          type: 'cant_move',
          slot,
          reason,
          ...(attempted ? { attempted_move: String(attempted) } : {}),
        });
        break;
      }

      case '-terastallize': {
        const slot = extractSlot(parsed.args[1] as string);
        const side = extractSide(parsed.args[1] as string);
        const toType = String(parsed.args[2] ?? '');
        const pkm = battle.getPokemon(parsed.args[1] as any);
        const species = pkm ? pkm.speciesForme || pkm.baseSpeciesForme || pkm.name : '';
        if (side) {
          currentTurnEvents.push({
            type: 'tera',
            side,
            slot,
            species,
            to_type: toType,
          });
          const teraInfo = { species, teraType: toType, onTurn: battle.turn };
          if (side === 'p1') teraUsedP1 = teraInfo;
          else teraUsedP2 = teraInfo;
        }
        break;
      }

      case 'switch':
      case 'drag':
      case 'replace': {
        finalizeCurrentMove();
        const ident = parsed.args[1] as string;
        const details = parsed.args[2] as string;
        const slot = extractSlot(ident);
        const side = extractSide(ident);
        const toSpecies = extractSpeciesFromDetails(details);
        const fromInfo = parseFromKwarg(kwArgs.from);
        // forced_by accepts move (Volt Switch / U-turn / Roar / Whirlwind /
        // Dragon Tail), item (Eject Button / Red Card / Eject Pack), or
        // ability (Emergency Exit / Wimp Out). Bare names with no prefix
        // (type=null) are also accepted — Showdown emits them that way for
        // some pivot moves.
        const forcedBy = fromInfo ? fromInfo.name : null;
        if (side && toSpecies) {
          currentTurnEvents.push({
            type: 'switch',
            side,
            slot,
            from_species: switchCapture?.fromSpecies ?? null,
            to_species: toSpecies,
            forced_by: forcedBy,
          });
          (side === 'p1' ? onFieldP1 : onFieldP2).add(speciesKey(toSpecies));
          // Reset derived moves for this slot — new mon means new revealedMoves.
          (side === 'p1' ? derivedMovesP1 : derivedMovesP2).set(slot, new Set<string>());
        }
        break;
      }

      case 'faint': {
        const ident = parsed.args[1] as string;
        const slot = extractSlot(ident);
        const side = extractSide(ident);
        const pkm = battle.getPokemon(ident as any);
        const species = pkm ? pkm.speciesForme || pkm.baseSpeciesForme || pkm.name : '';
        if (side) {
          // Queue if we're inside a move (faint is the consequence — emit AFTER the move).
          emitOrQueueEvent({
            type: 'faint',
            side,
            slot,
            species,
          });
        }
        break;
      }

      case '-enditem': {
        const slot = extractSlot(parsed.args[1] as string);
        const item = String(parsed.args[2] ?? '');
        const fromInfo = parseFromKwarg(kwArgs.from);
        let kind: ItemEventKind = 'consumed';
        let cause: string | undefined = undefined;
        if (fromInfo) {
          if (fromInfo.type === 'move') {
            cause = fromInfo.name;
            const m = fromInfo.name.toLowerCase();
            if (m === 'knock off') kind = 'knocked_off';
            else if (m === 'trick' || m === 'switcheroo') kind = 'tricked';
            else if (m === 'fling') kind = 'flung';
            else if (m === 'thief' || m === 'covet') kind = 'stolen';
            else if (m === 'incinerate') kind = 'incinerated';
          }
        }
        // Air Balloon pops on hit (no [from]).
        if (item.toLowerCase().includes('air balloon')) kind = 'popped';
        // Queue if a move is in-flight — Focus Sash / Eject Button trigger
        // mid-move, but should appear in stream order AFTER the move event.
        emitOrQueueEvent({
          type: 'item_event',
          slot,
          kind,
          item,
          ...(cause ? { cause } : {}),
        });
        break;
      }

      case '-item': {
        // Item being assigned (Trick / Switcheroo swap, Pickup / Harvest, Frisk reveal).
        const slot = extractSlot(parsed.args[1] as string);
        const item = String(parsed.args[2] ?? '');
        const fromInfo = parseFromKwarg(kwArgs.from);
        if (fromInfo?.type === 'move' || (fromInfo && fromInfo.type === null && /Trick|Switcheroo|Magician/i.test(fromInfo.name))) {
          emitOrQueueEvent({
            type: 'item_event',
            slot,
            kind: 'tricked',
            item,
            cause: fromInfo.name,
          });
        } else if (fromInfo?.type === 'ability' && /Magician/i.test(fromInfo.name)) {
          emitOrQueueEvent({
            type: 'item_event',
            slot,
            kind: 'stolen',
            item,
            cause: fromInfo.name,
          });
        } else if (fromInfo?.type === 'ability' && /Harvest|Pickup/i.test(fromInfo.name)) {
          emitOrQueueEvent({
            type: 'item_event',
            slot,
            kind: 'harvested',
            item,
            cause: fromInfo.name,
          });
        }
        // Frisk-style reveals (`from: ability: Frisk`) we don't surface — not actionable.
        break;
      }

      case 'turn': {
        flushTurnEvents();
        const turnNum = battle.turn;
        if (!seenTurns.has(turnNum)) {
          seenTurns.add(turnNum);
          snapshots.push({
            ...snapshotBattle(
              battle, ots,
              onFieldP1, onFieldP2,
              derivedMovesP1, derivedMovesP2,
              teraUsedP1, teraUsedP2,
            ),
            events: [],
          });
        }
        break;
      }

      case 'win': {
        winnerName = String(parsed.args[1] ?? '') || null;
        flushTurnEvents();
        break;
      }

      case 'tie': {
        flushTurnEvents();
        break;
      }
    }
  }

  flushTurnEvents();

  const teamSheets: TeamSheets | null =
    isOTS && pre.p1Sheets && pre.p2Sheets
      ? { p1: pre.p1Sheets, p2: pre.p2Sheets }
      : null;

  let winner: 'p1' | 'p2' | null = null;
  if (winnerName) {
    if (battle.p1.name && winnerName === battle.p1.name) winner = 'p1';
    else if (battle.p2.name && winnerName === battle.p2.name) winner = 'p2';
  }

  return { snapshots, teamSheets, winner };
}
