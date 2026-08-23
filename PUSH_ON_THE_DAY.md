# 🚀 Push on the day — checklist

Everything below assumes you're in the project folder with the `main` branch
checked out. It pushes to **github.com/Priyanshu8yadav/CA-CRS-Research**.

## 0 · One-time (already done for you)
- `.gitignore` excludes weights (`.pt`/`.pth`/`_parts_tmp`), datasets, `outputs/`,
  `.venv/`, videos, caches, the quarantine folder and the `original_run_dashboard.py` backup.
- `README.md`, `models/README.md`, `requirements.txt`, `LICENSE` are updated.

## 1 · Sanity check — confirm nothing huge is about to be committed
```bash
git add -A
git status                          # should list only source/docs, no .venv, no *.pth, no outputs/
git diff --cached --stat | tail -5
# hard guard: this must print NOTHING (no file ≥ 5 MB is staged)
git diff --cached --name-only | while read f; do [ -f "$f" ] && s=$(wc -c <"$f") && [ "$s" -gt 5000000 ] && echo "TOO BIG: $f ($((s/1000000))MB)"; done
```
If "TOO BIG" prints anything, **stop** and add that path to `.gitignore`, then
`git rm --cached <path>` and re-check.

## 2 · Commit
```bash
git commit -m "Round 2: DM-Count density counting, causal attribution, predictive forecast, flicker-free dashboard, honest calibration"
```

## 3 · Push
```bash
git push origin main
```
If git asks for a password, use a **GitHub Personal Access Token** (Settings →
Developer settings → Tokens), not your account password. Generate it the night before.

## 4 · Post-push — make it runnable for judges
- Upload `models/yolov8n_crowd_head.pt` (and optionally `model_qnrf.pth`) to the repo's
  **Releases** so anyone cloning can fetch weights. (Files under 2 GB are fine as release assets.)
- Open the repo on GitHub and confirm the README renders and there's **no `.venv/` or `outputs/`** in the file list.

## 5 · 60-second local smoke test before you present
```bash
source .venv/bin/activate
python tests/test_engine_upgrades.py     # expect 6/6 passed
python verify_kappa.py                    # expect R² = 0.615
streamlit run dashboard/app.py            # confirm it launches, MPS shows in Model Info
```

---
**Golden rule:** never `git add` the `.venv`, `outputs/`, `datasets/`, or `models/*.pth`.
The `.gitignore` handles it — but the guard in step 1 is your safety net.
