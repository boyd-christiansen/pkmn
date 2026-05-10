"""Parse saved SFT-row prompt strings back into their constituent sections.

Inputs:
    - The `system` content of an SFT row (a single string).
    - The `user` content of an SFT row (a single string with `=== HEADER ===`
      markers between sections).

Outputs:
    Structured dicts keyed by section name. Section ordering is preserved.

Schema awareness:
    SFT rows produced before the historical-context layer landed have a
    smaller user-prompt schema (no LEDGER / TURN-BY-TURN / SERIES STATE).
    The parser handles both — missing sections simply don't appear in
    the output.
"""
from __future__ import annotations

import re
from typing import Any


# All `=== HEADER ===` section names we know about, in their canonical
# order in the current schema. Used to: (a) bucket parsed sections under
# stable keys, (b) report which schema version a given row matches, and
# (c) tell the UI which sections are "missing" (rendered as collapsed
# "(not present in this row's schema)").
_KNOWN_USER_SECTIONS = [
    "GAME-STATE LEDGER",
    "TURN-BY-TURN",
    "SERIES STATE",
    "YOUR SPREADS",
    "THREAT MATRIX",
]


# Non-greedy `.*?` is important — section titles can contain `=` (e.g.
# "THREAT MATRIX  (turn 1, us=p1)"), so we can't use a `[^=]` character class.
# The trailing ` ===\s*$` is the terminator.
_USER_SECTION_RE = re.compile(r"^=== ([A-Z].*?) ===\s*$", re.MULTILINE)


def split_user_prompt(content: str) -> dict[str, Any]:
    """Split a saved user-prompt string into its constituent sections.

    Returns a dict like:
      {
        "header": {                     # the bit before the first === marker
          "raw": "=== TURN 8 ===\\nField: ...\\n\\n",
          "turn": 8,                    # parsed out for convenience
          "field_str": "weather=Rain, ...",
          "p1_active": "  [a] ...",     # raw active block
          "p1_bench": "Miraidon (fainted), ...",
          "p2_active": "...",
          "p2_bench": "...",
        },
        "sections": [                   # ordered as they appeared in the prompt
          {"name": "GAME-STATE LEDGER", "title": "GAME-STATE LEDGER", "body": "..."},
          {"name": "TURN-BY-TURN",      "title": "TURN-BY-TURN (game 1)", "body": "..."},
          ...
        ],
        "schema": "current" | "legacy",
        "missing_sections": ["SERIES STATE", ...],   # known sections this row doesn't include
      }
    """
    if not isinstance(content, str):
        return {"header": {}, "sections": [], "schema": "unknown", "missing_sections": []}

    matches = list(_USER_SECTION_RE.finditer(content))
    sections: list[dict[str, Any]] = []
    if matches:
        # Slice the body of each section: from after-its-header up to the
        # start of the next section header (or end of string).
        for idx, m in enumerate(matches):
            title = m.group(1).strip()
            body_start = m.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            body = content[body_start:body_end].strip("\n")
            name = _normalize_section_name(title)
            sections.append({"name": name, "title": title, "body": body})

    # Header = everything before the first section's header line, OR if the
    # first section is itself "TURN N" (a header-as-section) then we treat
    # that section as the header.
    header_raw = content
    if matches:
        first_match = matches[0]
        header_raw = content[: first_match.start()]
        if sections and sections[0]["name"] == "TURN":
            # The pre-historical-context schema had `=== TURN N ===` as a
            # leading section. Pull it out of the section list and into
            # the header dict.
            turn_section = sections.pop(0)
            header_raw = f"=== {turn_section['title']} ===\n{turn_section['body']}"

    header = _parse_header_block(header_raw)

    # Schema detection: "current" if it has any of the historical-context
    # sections, "legacy" otherwise.
    section_names = {s["name"] for s in sections}
    has_history = any(
        n in section_names for n in ("GAME-STATE LEDGER", "TURN-BY-TURN", "SERIES STATE")
    )
    schema = "current" if has_history else "legacy"
    missing = [n for n in _KNOWN_USER_SECTIONS if n not in section_names]

    return {
        "header": header,
        "sections": sections,
        "schema": schema,
        "missing_sections": missing,
    }


def _normalize_section_name(title: str) -> str:
    """Map a section title to a stable key.

    Examples:
      "GAME-STATE LEDGER"            → "GAME-STATE LEDGER"
      "TURN-BY-TURN (game 1)"        → "TURN-BY-TURN"
      "SERIES STATE (Bo3, game 2 of 3)" → "SERIES STATE"
      "YOUR SPREADS (inferred)"      → "YOUR SPREADS"
      "THREAT MATRIX  (turn 4, us=p1)" → "THREAT MATRIX"
      "TURN 4"                       → "TURN"
    """
    # Strip any "(...)" qualifier suffix.
    base = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    # Strip a trailing turn-number suffix (e.g. "TURN 4" → "TURN").
    base = re.sub(r"\s+\d+$", "", base).strip()
    # Collapse runs of whitespace.
    base = re.sub(r"\s+", " ", base)
    return base


def _parse_header_block(raw: str) -> dict[str, Any]:
    """Pull turn number, field string, and the four ACTIVE/BENCH blocks
    out of the leading header chunk."""
    out: dict[str, Any] = {"raw": raw}

    m_turn = re.search(r"=== TURN (\d+) ===", raw)
    if m_turn:
        out["turn"] = int(m_turn.group(1))

    m_field = re.search(r"^Field:\s*(.+)$", raw, flags=re.MULTILINE)
    if m_field:
        out["field_str"] = m_field.group(1).strip()

    # Each ACTIVE/BENCH block is delimited by its labelled header line.
    # Capture body up until the next labelled line or end-of-string.
    labels = (
        ("YOUR (P1) ACTIVE:", "p1_active"),
        ("YOUR (P1) BENCH:",  "p1_bench"),
        ("OPP (P2) ACTIVE:",  "p2_active"),
        ("OPP (P2) BENCH:",   "p2_bench"),
    )
    label_pattern = "|".join(re.escape(l) for l, _ in labels)
    for label, key in labels:
        # Match: <label> ... up to (next label OR === marker OR end).
        pat = re.compile(
            re.escape(label) + r"\s*(?P<body>.*?)(?=" + label_pattern + r"|^=== |\Z)",
            flags=re.DOTALL | re.MULTILINE,
        )
        m = pat.search(raw)
        if m:
            out[key] = m.group("body").strip("\n")

    return out


# =============================================================================
# System prompt: split rules apart for nicer rendering
# =============================================================================


_RULE_RE = re.compile(r"^(\d+)\.\s+(?:\*\*([^*]+)\*\*\s*[:\-—]?\s*)?(.*)$", re.MULTILINE)


def split_system_prompt(content: str) -> dict[str, Any]:
    """Split the system prompt into prelude + numbered rules.

    The rules block looks like:
      CRITICAL RULES:

      1. The Masking Rule: ...
      2. The Tool Rule: ...
      ...

    We slice out:
      - prelude: everything before "CRITICAL RULES:"
      - team_block: the YOUR TEAM / OPPONENT'S TEAM portions (already in prelude;
        kept whole)
      - rules: list of {number, title, body}
    """
    if not isinstance(content, str):
        return {"prelude": "", "rules": []}

    # Locate "CRITICAL RULES:" if present.
    split_marker = re.search(r"^CRITICAL RULES:\s*$", content, flags=re.MULTILINE)
    if not split_marker:
        return {"prelude": content.strip(), "rules": []}

    prelude = content[: split_marker.start()].strip()
    rules_block = content[split_marker.end():].strip()

    rules: list[dict[str, Any]] = []
    # Split on lines that begin with "<digit>. ". Capture all text up to the
    # next such line as that rule's body.
    rule_starts = [
        (m.start(), m.group(1), m.group(2) or "", m.group(3) or "")
        for m in re.finditer(r"^(\d+)\.\s+(?:\*\*([^*]+)\*\*[:\.\-\s]+)?(.*)", rules_block, flags=re.MULTILINE)
    ]
    for i, (pos, num, title, first_line) in enumerate(rule_starts):
        next_pos = rule_starts[i + 1][0] if i + 1 < len(rule_starts) else len(rules_block)
        body = rules_block[pos:next_pos].strip()
        # Try to extract a "Rule Name:" prefix from the first line if no bold title.
        explicit_title = title
        if not explicit_title:
            m_inline = re.match(r"^\d+\.\s+The\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)\s+Rule:", body)
            if m_inline:
                explicit_title = "The " + m_inline.group(1) + " Rule"
        rules.append({
            "number": int(num),
            "title": explicit_title.strip() if explicit_title else None,
            "body": body,
        })

    return {"prelude": prelude, "rules": rules}


# =============================================================================
# Assistant tool-call sequence flattener
# =============================================================================


def parse_tool_loop(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk an OpenAI-format `messages[]` array and pair up assistant
    tool_calls with their tool responses.

    Returns a list of iteration dicts:
      [
        {"type": "calc",
         "iteration": 1,
         "tool_call_id": "...",
         "args": {...},                 # parsed JSON from the tool_call's arguments
         "raw_args": "...",             # raw string in case parse fails
         "response": {...} or "...",    # parsed if JSON, else raw string
         "raw_response": "..."},
        ...
        {"type": "submit",
         "iteration": N,
         "tool_call_id": "...",
         "thought": "...",              # pre_tool_thought
         "action": {"slot_1": ..., "slot_2": ...},
         "raw_args": "..."}
      ]

    Also includes a final `text` entry if the assistant produced a plain
    content message (legacy rows used this before submit_decision was a tool).
    """
    out: list[dict[str, Any]] = []
    iter_num = 0
    pending_calls: dict[str, dict[str, Any]] = {}  # tool_call_id → entry

    def _try_parse_json(s: str) -> Any:
        if not isinstance(s, str):
            return s
        try:
            import json
            return json.loads(s)
        except (ValueError, TypeError):
            return s

    for m in messages:
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            content = m.get("content")
            if tcs:
                for tc in tcs:
                    iter_num += 1
                    fn = (tc.get("function") or {})
                    name = fn.get("name") or "?"
                    raw_args = fn.get("arguments") or ""
                    parsed_args = _try_parse_json(raw_args)
                    if name == "submit_decision":
                        thought = None
                        action = None
                        if isinstance(parsed_args, dict):
                            thought = parsed_args.get("pre_tool_thought")
                            action = parsed_args.get("action")
                        entry = {
                            "type": "submit",
                            "iteration": iter_num,
                            "tool_call_id": tc.get("id"),
                            "name": name,
                            "thought": thought,
                            "action": action,
                            "raw_args": raw_args,
                            "ack": None,           # filled by the matching tool message
                        }
                        out.append(entry)
                        pending_calls[tc.get("id") or f"_iter{iter_num}"] = entry
                    else:
                        entry = {
                            "type": "calc" if name == "calculate_damage" else "tool",
                            "iteration": iter_num,
                            "tool_call_id": tc.get("id"),
                            "name": name,
                            "args": parsed_args,
                            "raw_args": raw_args,
                            "response": None,
                            "raw_response": None,
                        }
                        out.append(entry)
                        pending_calls[tc.get("id") or f"_iter{iter_num}"] = entry
            elif content:
                out.append({"type": "text", "iteration": iter_num + 1, "content": content})
        elif role == "tool":
            tcid = m.get("tool_call_id")
            raw = m.get("content")
            entry = pending_calls.get(tcid)
            if entry is not None:
                if entry.get("type") == "submit":
                    # The tool response for submit_decision is just an ack
                    # (e.g. `{"status": "decision_committed"}`); store it
                    # separately so the UI can show it without cluttering
                    # the iteration list.
                    entry["ack"] = _try_parse_json(raw) if isinstance(raw, str) else raw
                else:
                    entry["raw_response"] = raw
                    entry["response"] = _try_parse_json(raw) if isinstance(raw, str) else raw
            else:
                # Orphaned tool message — surface it anyway.
                out.append({
                    "type": "tool_orphan",
                    "iteration": iter_num,
                    "tool_call_id": tcid,
                    "raw_response": raw,
                })

    return out
