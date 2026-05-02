import { calculate, Field, Generations, Move, Pokemon } from '@smogon/calc';
import type { Side } from '@smogon/calc';
import type {
  CalcRequest,
  CalcResponse,
  FieldInput,
  PokemonInput,
  SideInput,
} from './types.js';

const GEN = Generations.get(9);

function parseCurrentHP(input: PokemonInput['currentHP'], maxHP: number): number {
  if (input === undefined || input === null) return maxHP;
  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (trimmed.endsWith('%')) {
      const pct = parseFloat(trimmed.slice(0, -1));
      if (!Number.isFinite(pct)) {
        throw new Error(`Invalid currentHP percentage: "${input}"`);
      }
      return Math.max(1, Math.round((pct / 100) * maxHP));
    }
    const n = Number(trimmed);
    if (!Number.isFinite(n)) {
      throw new Error(`Invalid currentHP value: "${input}"`);
    }
    return Math.max(0, Math.round(n));
  }
  if (typeof input === 'number' && Number.isFinite(input)) {
    return Math.max(0, Math.round(input));
  }
  throw new Error(`Invalid currentHP type: ${typeof input}`);
}

function buildPokemon(input: PokemonInput): Pokemon {
  if (!input || typeof input !== 'object') {
    throw new Error('Pokemon input must be an object');
  }
  for (const required of ['species', 'item', 'ability', 'teraType', 'boosts'] as const) {
    if (input[required] === undefined) {
      throw new Error(`Pokemon "${input.species ?? '?'}" is missing required field: ${required}`);
    }
  }

  const useTera = input.isTera === true;

  const opts: ConstructorParameters<typeof Pokemon>[2] = {
    level: input.level ?? 50,
    ability: input.ability,
    item: input.item,
    boosts: input.boosts,
    status: input.status ?? '',
  };

  if (useTera) opts.teraType = input.teraType as any;
  if (input.nature) opts.nature = input.nature;
  if (input.evs) opts.evs = input.evs;
  if (input.ivs) opts.ivs = input.ivs;

  const pkmn = new Pokemon(GEN, input.species, opts);

  pkmn.originalCurHP = parseCurrentHP(input.currentHP, pkmn.maxHP());

  return pkmn;
}

function buildField(input: FieldInput | undefined): Field {
  const f = input ?? {};
  const sideToState = (s?: SideInput): Partial<Side> => ({
    spikes: s?.spikes ?? 0,
    isReflect: !!s?.isReflect,
    isLightScreen: !!s?.isLightScreen,
    isAuroraVeil: !!s?.isAuroraVeil,
    isProtected: !!s?.isProtected,
    isHelpingHand: !!s?.isHelpingHand,
    isFriendGuard: !!s?.isFriendGuard,
    isTailwind: !!s?.isTailwind,
    isFlowerGift: !!s?.isFlowerGift,
    isBattery: !!s?.isBattery,
    isPowerSpot: !!s?.isPowerSpot,
    isSteelySpirit: !!s?.isSteelySpirit,
    isSR: !!s?.isSR,
  });

  return new Field({
    gameType: f.gameType ?? 'Doubles',
    weather: f.weather as any,
    terrain: f.terrain as any,
    isGravity: !!f.isGravity,
    isMagicRoom: !!f.isMagicRoom,
    isWonderRoom: !!f.isWonderRoom,
    attackerSide: sideToState(f.attackerSide) as Side,
    defenderSide: sideToState(f.defenderSide) as Side,
  });
}

function flattenDamage(damage: number | number[] | number[][]): number[] {
  if (typeof damage === 'number') return [damage];
  if (damage.length === 0) return [0];
  if (Array.isArray(damage[0])) {
    const matrix = damage as number[][];
    return matrix.map((rolls) => rolls.reduce((a, b) => a + b, 0));
  }
  return damage as number[];
}

export function runCalc(req: CalcRequest): CalcResponse {
  if (!req.move || typeof req.move !== 'string') {
    throw new Error('Request is missing required string field: move');
  }

  const attacker = buildPokemon(req.attacker);
  const defender = buildPokemon(req.defender);
  const move = new Move(GEN, req.move);
  const field = buildField(req.field);

  const result = calculate(GEN, attacker, defender, move, field);

  const rolls = flattenDamage(result.damage);
  const minDamage = Math.min(...rolls);
  const maxDamage = Math.max(...rolls);
  const defenderMaxHP = defender.maxHP();
  const defenderCurrentHP = defender.originalCurHP;
  const denom = defenderMaxHP || 1;

  // @smogon/calc throws an assertion inside fullDesc/kochance when max damage
  // is 0 (typically an immunity). Short-circuit with a clean response.
  if (maxDamage === 0) {
    return {
      damageRolls: rolls,
      minDamage: 0,
      maxDamage: 0,
      defenderMaxHP,
      defenderCurrentHP,
      minPercent: 0,
      maxPercent: 0,
      koChance: 'no damage (immune or no effect)',
      description: `${attacker.name} ${move.name} vs. ${defender.name}: 0-0 (0 - 0%) -- no damage`,
      moveDescription: '0 - 0%',
    };
  }

  return {
    damageRolls: rolls,
    minDamage,
    maxDamage,
    defenderMaxHP,
    defenderCurrentHP,
    minPercent: Math.round((minDamage / denom) * 1000) / 10,
    maxPercent: Math.round((maxDamage / denom) * 1000) / 10,
    koChance: result.kochance().text,
    description: result.fullDesc(),
    moveDescription: result.moveDesc(),
  };
}
