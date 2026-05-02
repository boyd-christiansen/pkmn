"""Stitch Bo3 series and extract turn-by-turn board states from raw Showdown logs.

Inputs:
    Raw replay JSON dicts (as produced by data_scraper) — one per game. For
    Bo3 the three game replays are grouped into a single series first.
    Also: URL of the calc_microservice (defaults to http://localhost:3000).

Outputs:
    A list of `BoardState` snapshots per turn: active Pokémon for both players,
    HP / status / boosts / item / known moves / weather / terrain / side
    conditions / Tera state, plus the player decision actually taken on that
    turn (move + target / switch / Tera flag).

Isolation contract:
    The actual log-to-state translation is delegated to the calc_microservice
    `POST /parse_log` endpoint, which wraps the official @pkmn/client Battle
    state machine. This module only:
      1. POSTs raw logs to /parse_log,
      2. shapes the snapshots into `BoardState` objects,
      3. stitches Bo3 series (three replays → one series),
      4. extracts the player decision per turn (the "label").

    No regex parsing. No LLM. No calc.
"""
