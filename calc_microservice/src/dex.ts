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
