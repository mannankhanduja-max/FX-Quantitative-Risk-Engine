#!/usr/bin/env bash
#
# Pulls Matteo's QuantPortfolio into this folder and overlays the
# updated files that sit beside this script.
#
# Why a script instead of the files being here already: the session
# that produced the updated files had no network route to GitHub, so
# the unchanged modules (main.py, models/, utils/) are fetched from
# the source of truth rather than retyped by hand. Retyping someone
# else's 3,400 lines is how silent bugs get introduced.
#
# Run from the folder containing this script:
#
#     bash setup_from_upstream.sh
#
set -euo pipefail

UPSTREAM="git@github.com:carlonimatteoo03/Quant_Portoflio.git"
# If you use HTTPS rather than SSH keys, swap the line above for:
# UPSTREAM="https://github.com/carlonimatteoo03/Quant_Portoflio.git"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d)"

echo "==> Cloning upstream into a staging directory"
git clone --depth 1 "$UPSTREAM" "$STAGE/repo"

echo "==> Copying source files"
# Only the code. Not venv/, not .DS_Store, not the committed CSV output.
for path in main.py models utils; do
    if [ -e "$STAGE/repo/$path" ]; then
        cp -R "$STAGE/repo/$path" "$HERE/"
    fi
done

echo "==> Creating output directories"
mkdir -p "$HERE/data/raw" "$HERE/data/processed"
mkdir -p "$HERE/results/tables" "$HERE/results/figures"
touch "$HERE/data/raw/.gitkeep" "$HERE/data/processed/.gitkeep"
touch "$HERE/results/tables/.gitkeep" "$HERE/results/figures/.gitkeep"

# Bring the licence across if upstream has one.
[ -f "$STAGE/repo/LICENSE" ] && cp "$STAGE/repo/LICENSE" "$HERE/"

rm -rf "$STAGE"

echo
echo "Done. config.py, config_eu.py, README.md, requirements.txt and"
echo ".gitignore in this folder are the updated versions and were NOT"
echo "overwritten by the clone."
echo
echo "Next:"
echo "  python -m venv venv && source venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  python main.py"
