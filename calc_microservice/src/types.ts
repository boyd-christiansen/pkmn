import type { StatsTable } from '@smogon/calc';

export interface PokemonInput {
  species: string;
  item: string;
  ability: string;
  level?: number;
  currentHP?: number | string;
  status?: '' | 'brn' | 'par' | 'psn' | 'tox' | 'slp' | 'frz';
  teraType: string;
  isTera?: boolean;
  boosts: Partial<StatsTable>;

  evs?: Partial<StatsTable>;
  ivs?: Partial<StatsTable>;
  nature?: string;
}

export interface SideInput {
  spikes?: number;
  isReflect?: boolean;
  isLightScreen?: boolean;
  isAuroraVeil?: boolean;
  isProtected?: boolean;
  isHelpingHand?: boolean;
  isFriendGuard?: boolean;
  isTailwind?: boolean;
  isFlowerGift?: boolean;
  isBattery?: boolean;
  isPowerSpot?: boolean;
  isSteelySpirit?: boolean;
  isSR?: boolean;
}

export interface FieldInput {
  gameType?: 'Singles' | 'Doubles';
  weather?: string;
  terrain?: string;
  isGravity?: boolean;
  isMagicRoom?: boolean;
  isWonderRoom?: boolean;
  attackerSide?: SideInput;
  defenderSide?: SideInput;
}

export interface CalcRequest {
  attacker: PokemonInput;
  defender: PokemonInput;
  move: string;
  field?: FieldInput;
}

export interface CalcResponse {
  damageRolls: number[];
  minDamage: number;
  maxDamage: number;
  defenderMaxHP: number;
  defenderCurrentHP: number;
  minPercent: number;
  maxPercent: number;
  koChance: string;
  description: string;
  moveDescription: string;
}
