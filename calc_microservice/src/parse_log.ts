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

function snapshotBattle(battle: Battle): TurnSnapshot {
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

export function parseLog(log: string): TurnSnapshot[] {
  if (typeof log !== 'string') {
    throw new Error('log must be a string');
  }

  const battle = new Battle(GENS);
  const snapshots: TurnSnapshot[] = [];
  const seenTurns = new Set<number>();

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

    try {
      battle.add(parsed.args, parsed.kwArgs as any);
    } catch {
      // Skip lines the Battle state machine can't ingest (rare malformed events).
      continue;
    }

    if (parsed.args[0] === 'turn') {
      const turnNum = battle.turn;
      if (!seenTurns.has(turnNum)) {
        seenTurns.add(turnNum);
        snapshots.push(snapshotBattle(battle));
      }
    }
  }

  return snapshots;
}
