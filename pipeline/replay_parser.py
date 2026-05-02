"""Stitch Bo3 series and extract turn-by-turn board states from raw Showdown logs.

Inputs:
    Raw replay JSON dicts (as produced by data_scraper) — one per game.
    For Bo3 the three game replays are grouped into a single series first.

Outputs:
    A list of `BoardState` snapshots per turn: active Pokémon for both players,
    HP / status / boosts / item / known moves / weather / terrain / side
    conditions / Tera state, plus the player decision actually taken on that
    turn (move + target / switch / Tera flag).

Isolation contract:
    No network. No LLM. No calc. Pure deterministic transformation
    `raw_log -> structured states`. Other modules consume `BoardState` only;
    they never touch raw Showdown logs.
"""
