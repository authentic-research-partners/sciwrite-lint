"""``sciwrite-lint dismiss-claim`` — mark a claim finding as false-positive."""

from __future__ import annotations

import argparse


def run_dismiss_claim(args: argparse.Namespace) -> int:
    """Dismiss a claim verification finding as false positive."""
    from datetime import date

    from sciwrite_lint.cli._common import _load_config

    config = _load_config(args)
    ws = config.paper_workspace(args.paper)
    if not ws.root.exists():
        print(
            f"No workspace found for paper '{args.paper}'. "
            f"Run 'sciwrite-lint check --paper {args.paper}' first."
        )
        return 1

    from sciwrite_lint.references.workspace_db import (
        clear_claim_dismissal,
        dismiss_claim,
        find_claim,
        get_db,
        list_claims_for_key,
    )

    with get_db(ws.root) as conn:
        claim = find_claim(conn, args.key, args.line)

        if not claim:
            print(f"No claim found for key='{args.key}' line={args.line}.")
            print(f"Available claims for '{args.key}':")
            for c in list_claims_for_key(conn, args.key):
                print(
                    f"  line {c.get('line')}: {c.get('verdict')} — "
                    f"{c.get('context', '')[:80]}"
                )
            return 1

        claim_id = claim["id"]

        if args.clear:
            if not claim.get("dismissed"):
                print(f"Claim not dismissed: {args.key} line {args.line}")
                return 1
            clear_claim_dismissal(conn, claim_id)
            print(f"Cleared dismissal for {args.key} (line {args.line}).")
            return 0

        dismiss_claim(conn, claim_id, reason=args.reason, date_str=str(date.today()))

    v = claim.get("verdict", "?")
    print(f"Dismissed: {args.key} (line {args.line}) — {v}")
    print(f"  Reason: {args.reason}")
    print(f"  Date: {date.today()}")
    print()
    print("This claim will be shown separately in summaries and UI.")
    return 0
