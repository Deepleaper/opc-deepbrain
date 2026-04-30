"""DeepBrain CLI — command-line interface for local knowledge base."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="deepbrain",
        description="OPC DeepBrain — Local self-learning knowledge base",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Initialize knowledge base")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents from a directory")
    p_ingest.add_argument("directory", help="Directory to scan")
    p_ingest.add_argument("--namespace", default="documents", help="Namespace for entries")

    # search
    p_search = sub.add_parser("search", help="Search knowledge base")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-n", "--top-k", type=int, default=5, help="Number of results")

    # stats
    sub.add_parser("stats", help="Show knowledge base statistics")

    # learn
    p_learn = sub.add_parser("learn", help="Manually add a knowledge entry")
    p_learn.add_argument("content", help="Knowledge content")
    p_learn.add_argument("--type", default="fact", help="Claim type: fact/inference/preference/constraint/observation")
    p_learn.add_argument("--source", default="manual", help="Source attribution")

    # evolve
    sub.add_parser("evolve", help="Run knowledge evolution (decay, expire)")

    # watch
    p_watch = sub.add_parser("watch", help="Watch directory for changes, auto-ingest")
    p_watch.add_argument("directory", help="Directory to watch")
    p_watch.add_argument("--namespace", default="documents", help="Namespace")
    p_watch.add_argument("--interval", type=float, default=5.0, help="Poll interval (sec)")

    # conflicts
    p_conflicts = sub.add_parser("conflicts", help="Show entries with detected conflicts")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    from deepbrain import DeepBrain

    if args.command == "init":
        brain = DeepBrain()
        print(f"DeepBrain initialized at: {brain.db_path}")
        stats = brain.stats()
        print(f"Entries: {stats['total']} (active: {stats['active']})")
        return

    brain = DeepBrain()

    if args.command == "ingest":
        from deepbrain.ingest import ingest_directory
        print(f"Scanning: {args.directory}")
        result = ingest_directory(brain, args.directory, namespace=args.namespace)
        print(f"Done: {result['ingested']} ingested, {result['skipped']} skipped, {result['errors']} errors")

    elif args.command == "search":
        query = " ".join(args.query)
        results = brain.search(query, top_k=args.top_k)
        if not results:
            print("No results found.")
            return
        for i, r in enumerate(results, 1):
            ct = r.get("claim_type", "?")
            conf = r.get("confidence", 0)
            src = r.get("source", "")[:40]
            content = r["content"][:100].replace("\n", " ")
            print(f"{i}. [{ct}|conf={conf:.1f}] {content}")
            if src:
                print(f"   source: {src}")
            print()

    elif args.command == "stats":
        stats = brain.stats()
        print(f"Database: {stats['db_path']}")
        print(f"Total entries: {stats['total']}")
        print(f"Active: {stats['active']}")
        print(f"Superseded/expired: {stats['superseded']}")
        print(f"\nBy type:")
        for ct, count in stats.get("by_claim_type", {}).items():
            print(f"  {ct}: {count}")
        print(f"\nBy namespace:")
        for ns, count in stats.get("by_namespace", {}).items():
            print(f"  {ns}: {count}")

    elif args.command == "learn":
        entry_id = brain.learn(
            content=args.content,
            source=args.source,
            claim_type=args.type,
        )
        print(f"Stored: {entry_id[:8]}... [{args.type}]")

    elif args.command == "evolve":
        result = brain.evolve()
        print(f"Evolution complete: {result['decayed']} decayed, {result['expired']} expired")

    elif args.command == "watch":
        from deepbrain.watch import watch_directory
        watch_directory(brain, args.directory, namespace=args.namespace, interval=args.interval)

    elif args.command == "conflicts":
        with brain._lock:
            rows = brain.conn.execute(
                "SELECT id, content, metadata FROM deepbrain WHERE status='active' AND metadata LIKE '%conflict_with%'"
            ).fetchall()
        if not rows:
            print("No conflicts detected.")
            return
        print(f"Found {len(rows)} entries with conflicts:\n")
        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            conflicts = meta.get("conflict_with", [])
            print(f"  ID: {row['id'][:8]}...")
            print(f"  Content: {row['content'][:80]}")
            print(f"  Conflicts with: {[c[:8] for c in conflicts]}")
            print()


if __name__ == "__main__":
    main()
