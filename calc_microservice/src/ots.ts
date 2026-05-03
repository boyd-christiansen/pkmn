import { Teams } from '@pkmn/sets';
import type { PokemonSet } from '@pkmn/sets';

export interface OtsPokemonSet {
  species: string;
  item: string;
  ability: string;
  moves: string[];          // up to 4 — pad to 4 if shorter
  teraType: string | null;
  level: number;
  gender: string | null;
  // VGC OTS does not reveal these, but we surface them as nulls for shape stability:
  nature: string | null;
  evs: Record<string, number> | null;
  ivs: Record<string, number> | null;
}

function _empty(s?: string | null): boolean {
  return !s || s.length === 0;
}

function _normalizeSet(s: PokemonSet): OtsPokemonSet {
  // Pad moves to 4 entries so the shape is stable for downstream consumers.
  const moves = (s.moves ?? []).slice(0, 4);
  while (moves.length < 4) moves.push('');

  // VGC OTS hides EVs/IVs/Nature; if the packed payload didn't include them
  // we surface explicit nulls rather than the @pkmn defaults.
  const evsHasAny = s.evs && Object.values(s.evs).some((v) => v !== 0);
  const ivsHasAny = s.ivs && Object.values(s.ivs).some((v) => v !== 31);

  return {
    species: s.species,
    item: s.item ?? '',
    ability: s.ability ?? '',
    moves,
    teraType: s.teraType ?? null,
    level: s.level ?? 50,
    gender: _empty(s.gender) ? null : s.gender,
    nature: _empty(s.nature) ? null : s.nature,
    evs: evsHasAny ? { ...s.evs } : null,
    ivs: ivsHasAny ? { ...s.ivs } : null,
  };
}

/**
 * Decode a single side's `|showteam|` packed payload into a structured
 * list of OtsPokemonSet entries. Returns null on parse failure.
 */
export function decodeShowteam(packed: string): OtsPokemonSet[] | null {
  if (!packed || typeof packed !== 'string') return null;
  let team;
  try {
    team = Teams.unpackTeam(packed);
  } catch {
    return null;
  }
  if (!team || !team.team) return null;
  return team.team.map(_normalizeSet);
}
