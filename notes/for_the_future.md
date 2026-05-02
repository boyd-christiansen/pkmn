# For the future: getting more PS data

Right now the scraper covers top-500 ladder users × their saved replays in the
ladder's format. If we need more data later, here are the next levers in order
of leverage, plus what to watch out for.

## Other Pokémon Showdown endpoints

### Verified in the initial scraper build
- `pokemonshowdown.com/ladder/{format_id}.json` — top 500 ladder users, Elo desc.
- `replay.pokemonshowdown.com/search.json?user=X&format=Y&page=N` — paginated
  replay search for one user (50/page).
- `replay.pokemonshowdown.com/{replay_id}.json` / `.log` — full battle replay
  (metadata + pipe-delimited log).

### Known but not verified — `curl` first before building on them
- **`search.json?format=Y&page=N`** — format-only search (no `user=` filter).
  Returns *all* public replays for the format, newest first. This is the path
  past the top-500 ceiling.
- **`search.json?...&before={uploadtime}`** — timestamp cursor for deep
  pagination. PS may cap page-number paging around ~25 pages and force you to
  use `before` to go further back.
- **`pokemonshowdown.com/users/{userid}.json`** — public user profile: per-
  format ratings, registration date. Lets you filter by Elo without pulling
  the whole ladder.
- **`play.pokemonshowdown.com/data/pokedex.json`, `moves.json`, `items.json`,
  `abilities.json`, `learnsets.json`, `formats-data.json`** — canonical game
  data. Exactly the knowledge base to expose to the LLM as a tool-callable
  reference for damage math, legality checks, etc.
- **`smogon.com/stats/{YYYY-MM}/{format}-{cutoff}.json`** — monthly Smogon
  usage stats: top Pokémon, move/item/teammate distributions, win rates per
  Elo cutoff. Great for evaluation and team-building context.

## Can we scrape *all* battles?

Two senses:

### 1. All publicly saved replays — yes, mostly
Via format-only `search.json` paginated until empty (using `before={uploadtime}`
once page-number paging caps out).

**Hard limit:** only games where a player clicks "save replay" land on the
replay server — probably <5% of all ladder games. PS also ages out very old
replays for low-traffic formats, but active formats like `gen9vgc2026regi`
have effectively full history.

### 2. All games actually played, including unsaved — only via WebSocket
PS exposes `wss://sim3.psim.us/showdown/websocket`. You connect, query the
room/battle list, and join active battles as a spectator. The protocol is
documented in `pokemon-showdown/PROTOCOL.md` on GitHub.

**Caveats:**
- Forward-only — can't recover history this way.
- Continuous service, not a one-shot scrape — needs to run 24/7.
- Stateful protocol (more complex than HTTP polling).
- Be polite: one connection, reasonable join rate. PS is generally permissive
  about read-only spectating but it's not unlimited.

## Recommended order if we need more data

1. **(Already built)** Top-500 ladder × per-user replay search.
2. **Format-only `search.json` paginated to exhaustion**, filtered client-side
   by the `rating` field to keep only games above an Elo threshold (e.g.
   1500+). Likely a 5–20× data multiplier over (1). Small code change: one
   new helper in `ps_client.py`, one new stage in `scrape.py`.
3. **WebSocket spectator capture.** Only worth the operational cost once (1)
   and (2) plateau.

## Caveats to remember regardless of source

- **Saved-replay selection bias:** players save their highlight games, not
  their losses or boring games. Corpus skews toward decisive/interesting
  outcomes. Counterbalance with WebSocket capture if this becomes a problem.
- **Format ID drift:** VGC regulations rotate (`Reg I`, `Reg J`, ...). Format
  IDs change each season. The current scraper hard-codes
  `gen9vgc2026regi[bo3]`; revisit when the metagame moves on.
- **Private replays** (`private == 1` in search results) require a password.
  We skip them — they're rare in our top-500 sample.
- **Rate limits:** PS doesn't document any, doesn't send `Retry-After`/`X-
  RateLimit-*` headers, but be polite. Current scraper uses concurrency 8
  with exponential-backoff retry; that's been fine for top-500 crawls.