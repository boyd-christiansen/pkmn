import { Battle } from '@pkmn/client';
import type { Pokemon, Side } from '@pkmn/client';
import { Generations } from '@pkmn/data';
import { Dex } from '@pkmn/dex';
import { Protocol } from '@pkmn/protocol';
import { decodeShowteam, type OtsPokemonSet } from './ots.js';

const GENS = new Generations(Dex as any);

export interface ActivePokemonSnapshot {
  slot: 'a' | 'b' | 'c';
  species: string;
  hpPercent: number;
  fainted: boolean;
  status: string | null;
  ability: string | null;
  item: string | null;
  revealedMoves: string[];
  knownMoves: string[] | null;   // OTS-known full moveset (4 entries) when isOTS, else null
  teraType: string | null;
  isTerastallized: boolean;
  terastallizedAs: string | null;
  boosts: Record<string, number>;
}

export interface BenchPokemonSnapshot {
  species: string;
  fainted: boolean;
}

export interface SideSnapshot {
  player: string;
  active: ActivePokemonSnapshot[];
  bench: BenchPokemonSnapshot[];
}

export interface DamageEvent {
  attacker_slot: string;
  defender_slot: string;
  move_name: string;
  hp_before_pct: number;
  hp_after_pct: number;
  is_crit: boolean;
  is_ko: boolean;
}

export interface TurnSnapshot {
  turn: number;
  field: {
    weather: string | null;
    terrain: string | null;
    tailwindP1: boolean;
    tailwindP2: boolean;
  };
  p1: SideSnapshot;
  p2: SideSnapshot;
  actionLog: DamageEvent[];
}

export interface TeamSheets {
  p1: OtsPokemonSet[];
  p2: OtsPokemonSet[];
}

export interface ParseLogResult {
  snapshots: TurnSnapshot[];
  teamSheets: TeamSheets | null;   // null in CTS games
}

const SLOT_LETTERS: Array<'a' | 'b' | 'c'> = ['a', 'b', 'c'];

function speciesKey(species: string): string {
  return species.toLowerCase().replace(/[^a-z0-9]/g, '');
}

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

function snapshotActive(
  p: Pokemon,
  slotIndex: number,
  ots: OtsContext,
  sideIsP1: boolean,
): ActivePokemonSnapshot {
  const maxhp = p.maxhp || 1;
  const hpPercent = Math.round((p.hp / maxhp) * 1000) / 10;
  const species = p.speciesForme || p.baseSpeciesForme || p.name;

  const sheetIndex = sideIsP1 ? ots.p1SheetIndex : ots.p2SheetIndex;
  const sheet = ots.isOTS ? sheetIndex.get(speciesKey(species)) : undefined;

  // Static fields: prefer in-game-revealed values, fall back to OTS.
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

  return {
    slot: SLOT_LETTERS[slotIndex] ?? 'a',
    species,
    hpPercent,
    fainted: !!p.fainted,
    status: p.status ?? null,
    ability,
    item,
    revealedMoves: (p.moves ?? []).map((m) => String(m)),
    knownMoves,
    teraType,
    isTerastallized: !!p.isTerastallized,
    terastallizedAs: p.terastallized ?? null,
    boosts: { ...p.boosts } as Record<string, number>,
  };
}

function snapshotBench(p: Pokemon): BenchPokemonSnapshot {
  return {
    species: p.speciesForme || p.baseSpeciesForme || p.name,
    fainted: !!p.fainted,
  };
}

function snapshotSide(
  side: Side,
  ots: OtsContext,
  sideIsP1: boolean,
  onFieldSet: Set<string>,
): SideSnapshot {
  const activeSet = new Set(side.active.filter((p): p is Pokemon => p !== null));

  const active: ActivePokemonSnapshot[] = [];
  side.active.forEach((p, idx) => {
    if (p) active.push(snapshotActive(p, idx, ots, sideIsP1));
  });

  // Bench filtering rules:
  //   CTS:  bench = team − active   (current behavior, unchanged)
  //   OTS P1: bench = (team − active) ∩ broughtSet  (the 4 brought minus active)
  //   OTS P2: bench = (team − active) ∩ onFieldSet  (only mons that have appeared)
  const filterSet = ots.isOTS
    ? sideIsP1
      ? ots.p1Brought
      : onFieldSet
    : null;

  const bench: BenchPokemonSnapshot[] = side.team
    .filter((p) => !activeSet.has(p))
    .filter((p) => {
      if (filterSet === null) return true;
      const sp = p.speciesForme || p.baseSpeciesForme || p.name;
      return filterSet.has(speciesKey(sp));
    })
    .map(snapshotBench);

  return {
    player: side.name || `p${side.n + 1}`,
    active,
    bench,
  };
}

function snapshotBattle(
  battle: Battle,
  ots: OtsContext,
  onFieldP1: Set<string>,
  onFieldP2: Set<string>,
): Omit<TurnSnapshot, 'actionLog'> {
  const sideTailwind = (side: Side): boolean => {
    const conds = side.sideConditions ?? {};
    return Object.keys(conds).some((id) => id.toLowerCase() === 'tailwind');
  };

  return {
    turn: battle.turn,
    field: {
      weather: battle.field.weather ?? null,
      terrain: battle.field.terrain ?? null,
      tailwindP1: sideTailwind(battle.p1),
      tailwindP2: sideTailwind(battle.p2),
    },
    p1: snapshotSide(battle.p1, ots, true, onFieldP1),
    p2: snapshotSide(battle.p2, ots, false, onFieldP2),
  };
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
  // PokemonDetails: "Species, L50, M, shiny" — take the first comma-delimited token.
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

/**
 * One-shot pre-pass over the log. Captures:
 *  - Per-side `|showteam|` payloads (decoded into OtsPokemonSet[]).
 *  - The set of species each side ever brought to the field, derived from
 *    every `|switch|` / `|drag|` / `|replace|` event in the log.
 *
 * This pass is needed because `snapshotSide` for OTS Bo3 has to gate the
 * bench at turn 1 by which 4 of the 6 OTS mons were actually brought —
 * info we wouldn't have during a strict left-to-right pass until the
 * brought-but-not-leading mon switches in.
 */
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

export function parseLog(log: string): ParseLogResult {
  if (typeof log !== 'string') {
    throw new Error('log must be a string');
  }

  // === pre-pass: OTS team sheets + brought sets ===
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

  // === main pass ===
  const battle = new Battle(GENS);
  const snapshots: TurnSnapshot[] = [];
  const seenTurns = new Set<number>();

  // Running on-field set per side (OTS-only relevance; populated on switches).
  const onFieldP1 = new Set<string>();
  const onFieldP2 = new Set<string>();

  // Action-log tracking (unchanged from previous version).
  let currentMove: { attacker_slot: string; move_name: string } | null = null;
  const pendingCrits = new Set<string>();
  let currentTurnEvents: DamageEvent[] = [];

  const flushTurnEvents = () => {
    if (snapshots.length > 0 && currentTurnEvents.length > 0) {
      const last = snapshots[snapshots.length - 1]!;
      last.actionLog = [...last.actionLog, ...currentTurnEvents];
    }
    currentTurnEvents = [];
    currentMove = null;
    pendingCrits.clear();
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

    // === pre-apply: capture defender HP before |-damage| takes effect ===
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

    // === apply ===
    try {
      battle.add(parsed.args, parsed.kwArgs as any);
    } catch {
      continue;
    }

    // === post-apply ===
    if (argName === 'switch' || argName === 'drag' || argName === 'replace') {
      const ident = parsed.args[1] as string;
      const details = parsed.args[2] as string;
      const side = extractSide(ident);
      const species = extractSpeciesFromDetails(details);
      if (side && species) {
        (side === 'p1' ? onFieldP1 : onFieldP2).add(speciesKey(species));
      }
    }

    if (argName === 'move') {
      currentMove = {
        attacker_slot: extractSlot(parsed.args[1] as string),
        move_name: String(parsed.args[2]),
      };
      pendingCrits.clear();
    } else if (argName === '-crit') {
      pendingCrits.add(extractSlot(parsed.args[1] as string));
    } else if (dmgCapture && currentMove) {
      const target = battle.getPokemon(dmgCapture.targetIdent as any);
      const hpAfter = hpPct(target);
      const isKo = hpAfter <= 0 || !!target?.fainted;
      currentTurnEvents.push({
        attacker_slot: currentMove.attacker_slot,
        defender_slot: dmgCapture.defenderSlot,
        move_name: currentMove.move_name,
        hp_before_pct: round1(dmgCapture.hpBefore),
        hp_after_pct: round1(hpAfter),
        is_crit: pendingCrits.has(dmgCapture.defenderSlot),
        is_ko: isKo,
      });
    } else if (argName === 'turn') {
      flushTurnEvents();
      const turnNum = battle.turn;
      if (!seenTurns.has(turnNum)) {
        seenTurns.add(turnNum);
        snapshots.push({
          ...snapshotBattle(battle, ots, onFieldP1, onFieldP2),
          actionLog: [],
        });
      }
    } else if (argName === 'win' || argName === 'tie') {
      flushTurnEvents();
    }
  }

  flushTurnEvents();

  const teamSheets: TeamSheets | null =
    isOTS && pre.p1Sheets && pre.p2Sheets
      ? { p1: pre.p1Sheets, p2: pre.p2Sheets }
      : null;

  return { snapshots, teamSheets };
}
