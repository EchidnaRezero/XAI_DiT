"""Unified CLI entry point: python -m scripts.tracker.cli <exp|issue|plan|sync>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import exp, issue, plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tracker", description="XAI_DiT project tracker & sync gate")
    sub = p.add_subparsers(dest="domain", required=True)

    # exp
    p_exp = sub.add_parser("exp", help="Experiment run indexing & HISTORY generation")
    exp_sub = p_exp.add_subparsers(dest="cmd", required=True)

    p_exp_new = exp_sub.add_parser("new", help="Create auto-numbered YYYYMMDD_NN_<slug> run folder")
    p_exp_new.add_argument("track", choices=exp.VALID_TRACKS)
    p_exp_new.add_argument("slug")
    p_exp_new.add_argument("--type", dest="run_type", default="t2i_bench")
    p_exp_new.add_argument("--models", default="")
    p_exp_new.add_argument("--gpu", default="L4")
    p_exp_new.add_argument("--status", default="verified")
    p_exp_new.add_argument("--tags", default="")
    p_exp_new.add_argument("--summary", default="")
    p_exp_new.add_argument("--date", dest="date_str", default=None)

    p_exp_chk = exp_sub.add_parser("check", help="Validate runs_index.jsonl and detect orphan folders")
    p_exp_chk.add_argument("track", choices=exp.VALID_TRACKS)

    p_exp_rnd = exp_sub.add_parser("render", help="Regenerate HISTORY.md from runs_index.jsonl")
    p_exp_rnd.add_argument("track", choices=exp.VALID_TRACKS)

    p_exp_sql = exp_sub.add_parser("to-sqlite", help="Export runs_index.jsonl to registry.sqlite")
    p_exp_sql.add_argument("track", choices=exp.VALID_TRACKS)
    p_exp_sql.add_argument("--out", default=None)

    # issue
    p_iss = sub.add_parser("issue", help="Issue auto-numbering & indexing")
    iss_sub = p_iss.add_subparsers(dest="cmd", required=True)

    p_iss_new = iss_sub.add_parser("new", help="Create next issues/NNN_<slug>.md")
    p_iss_new.add_argument("slug")
    p_iss_new.add_argument("--title", default=None)
    iss_sub.add_parser("index", help="Rebuild issues/issues_index.jsonl")
    iss_sub.add_parser("check", help="Check issue numbering integrity")

    # plan
    p_pln = sub.add_parser("plan", help="Plan / interrupt management & doc link checks")
    pln_sub = p_pln.add_subparsers(dest="cmd", required=True)

    p_pln_act = pln_sub.add_parser("new-active", help="Create plan/active/PLAN-NNN_<slug>.md")
    p_pln_act.add_argument("slug")
    p_pln_act.add_argument("--title", default=None)

    p_pln_int = pln_sub.add_parser("new-interrupt", help="Create plan/interrupts/YYYYMMDD_<slug>.md")
    p_pln_int.add_argument("slug")
    p_pln_int.add_argument("--title", default=None)
    pln_sub.add_parser("check", help="Check broken markdown links and *_KR.md sync")

    # sync
    p_syn = sub.add_parser("sync", help="Run all checks (and optionally refresh generated indices)")
    p_syn.add_argument("--check", action="store_true", help="Exit non-zero if any check fails")
    p_syn.add_argument("--render", action="store_true", help="Refresh HISTORY.md and issues_index.jsonl")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.domain == "exp":
        if args.cmd == "new":
            rec = exp.new_run(
                track=args.track,
                slug=args.slug,
                run_type=args.run_type,
                models=[m.strip() for m in args.models.split(",") if m.strip()],
                gpu=args.gpu,
                status=args.status,
                tags=[t.strip() for t in args.tags.split(",") if t.strip()],
                summary=args.summary,
                date_str=args.date_str,
            )
            exp.render_history(args.track)
            print(f"Created {rec['run_id']} -> experiments/{args.track}/{rec['path']}")
            return 0
        if args.cmd == "render":
            out = exp.render_history(args.track)
            print(f"Rendered {out}")
            return 0
        if args.cmd == "to-sqlite":
            out = exp.export_sqlite(args.track, Path(args.out) if args.out else None)
            print(f"Exported {out}")
            return 0
        if args.cmd == "check":
            errs = exp.check_track(args.track)
            for e in errs:
                print(e)
            print(f"exp check ({args.track}): {len(errs)} error(s)")
            return 1 if errs else 0

    if args.domain == "issue":
        if args.cmd == "new":
            out = issue.new_issue(args.slug, title=args.title)
            print(f"Created {out}")
            return 0
        if args.cmd == "index":
            out = issue.rebuild_index()
            print(f"Updated {out}")
            return 0
        if args.cmd == "check":
            errs = issue.check_issues()
            for e in errs:
                print(e)
            print(f"issue check: {len(errs)} error(s)")
            return 1 if errs else 0

    if args.domain == "plan":
        if args.cmd == "new-active":
            out = plan.new_active(args.slug, title=args.title)
            print(f"Created {out}")
            return 0
        if args.cmd == "new-interrupt":
            out = plan.new_interrupt(args.slug, title=args.title)
            print(f"Created {out}")
            return 0
        if args.cmd == "check":
            errs = plan.check_links() + plan.check_bilingual()
            for e in errs:
                print(e)
            print(f"plan check: {len(errs)} error(s)")
            return 1 if errs else 0

    if args.domain == "sync":
        if args.render:
            for t in exp.VALID_TRACKS:
                if (exp.track_dir(t) / "runs_index.jsonl").exists():
                    exp.render_history(t)
            issue.rebuild_index()
        errs: list[str] = []
        for t in exp.VALID_TRACKS:
            errs.extend(exp.check_track(t))
        errs.extend(issue.check_issues())
        errs.extend(plan.check_links())
        errs.extend(plan.check_bilingual())
        for e in errs:
            print(e)
        print(f"tracker sync: {len(errs)} error(s)")
        return 1 if (args.check and errs) else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
