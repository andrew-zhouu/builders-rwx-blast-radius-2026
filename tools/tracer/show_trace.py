"""Summarize trace shards. Usage: python show_trace.py <out.pkl> [...]"""
import collections
import pickle
import sys

for path in sys.argv[1:]:
    recs = pickle.load(open(path, "rb"))
    for r in recs:
        if r.get("error"):
            print(r["n"], r["id"], "ERROR", r["error"])
            continue
        run, imp, sub = r["run"], r["imp"], r["sub"]
        per_file = collections.Counter(f for f, _ in run["lines"])
        print(
            f"{r['n']:5d} ok={r['ok']} run_cpu={r['t_run_cpu']:.3f} run_wall={r['t_run_wall']:.3f} "
            f"imp_cpu={r['t_import_cpu']:.3f} run.lines={len(run['lines'])} imp.lines={len(imp['lines'])} "
            f"ilines={len(r['ilines'])} run.classes={len(run['classes'])} imp.classes={len(imp['classes'])} "
            f"words_in={len(run['words_in'])} words_out={len(run['words_out'])} toktypes={len(run['toktypes'])} "
            f"files={r['files'][:3]} n_sub={r['n_sub']} {r['id']}"
        )
        print("       run top files:", per_file.most_common(6))
        if sub:
            sf = collections.Counter(f for f, _ in sub["run"]["lines"])
            print(f"       sub run.lines={len(sub['run']['lines'])} imp.lines={len(sub['imp']['lines'])} top:", sf.most_common(5))
        new_mods = sorted(set(r["modules_at_end"]) - set(r["modules_at_start"]))
        print("       lazily loaded:", new_mods[:10], "| problems:", r["problems"])
        print("       sample words_in:", run["words_in"][:12], "toktypes:", run["toktypes"][:8])
