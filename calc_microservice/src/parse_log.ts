import { Battle } from '@pkmn/client';
import type { Pokemon, Side } from '@pkmn/client';
import { Generations } from '@pkmn/data';
import { Dex } from '@pkmn/dex';
import { Protocol } from '@pkmn/protocol';

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
  attacker_slot: string;     // "p1a" | "p1b" | "p2a" | "p2b"
  defender_slot: string;
  move_name: string;
  hp_before_pct: number;     // 0–100, defender HP just BEFORE the hit
  hp_after_pct: number;      // 0–100 (0 if KO'd)
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
  actionLog: DamageEvent[];  // events that occurred DURING this turn (forward-looking)
}

const SLOT_LETTERS: Array<'a' | 'b' | 'c'> = ['a', 'b', 'c'];

function snapshotActive(p: Pokemon, slotIndex: number): ActivePokemonSnapshot {
  const maxhp = p.maxhp || 1;
  const hpPercent = Math.round((p.hp / maxhp) * 1000) / 10;
  return {
    slot: SLOT_LETTERS[slotIndex] ?? 'a',
    species: p.speciesForme || p.baseSpeciesForme || p.name,
    hpPercent,
    fainted: !!p.fainted,
    status: p.status ?? null,
    ability: p.ability ? String(p.ability) : null,
    item: p.item ? String(p.item) : null,
    revealedMoves: (p.moves ?? []).map((m) => String(m)),
    teraType: p.teraType ?? null,
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

function snapshotSide(side: Side): SideSnapshot {
  const activeSet = new Set(side.active.filter((p): p is Pokemon => p !== null));

  const active: ActivePokemonSnapshot[] = [];
  side.active.forEach((p, idx) => {
    if (p) active.push(snapshotActive(p, idx));
  });

  const bench: BenchPokemonSnapshot[] = side.team
    .filter((p) => !activeSet.has(p))
    .map(snapshotBench);

  return {
    player: side.name || `p${side.n + 1}`,
    active,
    bench,
  };
}

function snapshotBattle(battle: Battle): Omit<TurnSnapshot, 'actionLog'> {
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
    p1: snapshotSide(battle.p1),
    p2: snapshotSide(battle.p2),
  };
}

function extractSlot(ident: string | undefined): string {
  if (!ident) return '';
  const m = ident.match(/^(p[12][a-c])/);
  return m?.[1] ?? '';
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

function hpPct(p: Pokemon | null | undefined): number {
  if (!p) return 0;
  const maxhp = p.maxhp || 1;
  return (p.hp / maxhp) * 100;
}

export function parseLog(log: string): TurnSnapshot[] {
  if (typeof log !== 'string') {
    throw new Error('log must be a string');
  }

  const battle = new Battle(GENS);
  const snapshots: TurnSnapshot[] = [];
  const seenTurns = new Set<number>();

  // Action-log tracking. We attribute |-damage| events (without [from]) to
  // the most recent |move|, attaching the resulting DamageEvent to the
  // currently in-flight turn. Events for turn N flush onto snapshot[N]
  // when |turn|N+1 (or end of input) is reached.
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

    // === pre-apply hook: capture defender HP before |-damage| takes effect ===
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

    // === post-apply hooks ===
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
      const isKo = hpAfter <= 0 || !!(target?.fainted);
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
      // Flush events from the previous turn onto its snapshot, then emit
      // a fresh snapshot for the new turn (with empty actionLog to be filled).
      flushTurnEvents();
      const turnNum = battle.turn;
      if (!seenTurns.has(turnNum)) {
        seenTurns.add(turnNum);
        snapshots.push({ ...snapshotBattle(battle), actionLog: [] });
      }
    } else if (argName === 'win' || argName === 'tie') {
      flushTurnEvents();
    }
  }

  // Catch any trailing events (replay ended without a |win|/|tie| or |turn|).
  flushTurnEvents();

  return snapshots;
}
