import { Dex } from '@pkmn/dex';

export interface MoveLookup {
  name: string;
  id: string;
  type: string;
  category: 'Physical' | 'Special' | 'Status';
  basePower: number;
  accuracy: number | true;
  target: string;
  priority: number;
}

export function lookupMove(name: string): MoveLookup | null {
  const move = Dex.moves.get(name);
  if (!move?.exists) return null;
  return {
    name: move.name,
    id: move.id,
    type: move.type,
    category: move.category,
    basePower: move.basePower,
    accuracy: move.accuracy,
    target: move.target,
    priority: move.priority,
  };
}

export interface SpeciesLookup {
  name: string;
  id: string;
  baseStats: { hp: number; atk: number; def: number; spa: number; spd: number; spe: number };
  types: string[];
  weightkg: number;
}

/** Base-stats + typing lookup. Used by the Python damage_inferencer's
 *  speed (move-order) inference, which needs base Speed to convert an
 *  observed "moved-before / moved-after" ordering into an EV bound. */
export function lookupSpecies(name: string): SpeciesLookup | null {
  const sp = Dex.species.get(name);
  if (!sp?.exists) return null;
  return {
    name: sp.name,
    id: sp.id,
    baseStats: sp.baseStats,
    types: sp.types,
    weightkg: sp.weightkg,
  };
}
