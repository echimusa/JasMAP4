#!/usr/bin/env bash
# release_to_github.sh -- publish a new JasMAP4 version ON TOP of the existing
# GitHub repository, preserving its history and earlier tags (e.g. v4.0).
#
#   ./release_to_github.sh --tag v4.1 --message "JasMAP4 v4.1: GLMM, SPA, parallel workflow"
#   ./release_to_github.sh --tag v4.1 --dry-run        # show what would change, push nothing
#
# Run it from the unpacked package folder (the one containing JasMAP4.py).
#
# Why not push_to_github.sh for an update: that script starts a NEW repository
# (git init). Pushing a fresh, unrelated history to a repository that already
# has commits is rejected -- and forcing it would erase the existing history
# and the v4.0 baseline. This script instead clones the repository, replaces
# its files with this package (files deleted from the package are deleted in
# the repository too), and commits on top.
#
# Credentials are never read or stored here: git prompts for them, or uses
# your credential helper / SSH key. When prompted for a password, paste a
# GitHub Personal Access Token, not your account password.
set -euo pipefail

REMOTE="https://github.com/echimusa/JasMAP4.git"
BRANCH="main"; TAG=""; MESSAGE=""; DRY=0
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)  REMOTE="$2"; shift 2 ;;
    --branch)  BRANCH="$2"; shift 2 ;;
    --tag)     TAG="$2"; shift 2 ;;
    --message) MESSAGE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$SRC/JasMAP4.py" ]] || { echo "ERROR: run this from the package folder (JasMAP4.py not found)." >&2; exit 1; }
[[ -n "$TAG" ]] || { echo "ERROR: --tag is required, e.g. --tag v4.1" >&2; exit 1; }
[[ -n "$MESSAGE" ]] || MESSAGE="JasMAP4 $TAG"
command -v git >/dev/null || { echo "ERROR: git is not installed." >&2; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
echo "==> Cloning $REMOTE"
git clone -q --branch "$BRANCH" "$REMOTE" "$WORK/repo"
cd "$WORK/repo"
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "ERROR: tag $TAG already exists in the repository -- choose a new version." >&2; exit 1
fi

echo "==> Syncing this package into the repository"
# Remove everything except the repository's own .git, then copy the package
# in (tar is universal; rsync is not installed on every cluster). Files that
# no longer exist in the package are therefore deleted in the repository too.
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
tar -C "$SRC" --exclude=.git -cf - . | tar -xf -
git add -A

# ---- safety checks before anything leaves the machine -------------------------
STAGED="$(git diff --cached --name-only)"
SRC_N="$(git ls-files 'jasmap/*.py' | wc -l | tr -d ' ')"
if [[ "$SRC_N" -lt 5 ]]; then
  echo "ERROR: only $SRC_N jasmap/ source files would be in the repository -- a .gitignore rule may be excluding source. Aborting." >&2; exit 1
fi
RISKY="$(git ls-files | grep -Ei '\.(pheno|cov|covar|fam|bim|bed|vcf|vcf\.gz|bgen|tped)$' | grep -v '^gene_annotation\.bed$' || true)"
if [[ -n "$RISKY" ]]; then
  echo "ERROR: files that look like participant or genotype data would be published:" >&2
  echo "$RISKY" | sed 's/^/    /' >&2
  echo "Remove them from the package folder and re-run. Aborting." >&2; exit 1
fi
BIG="$(git ls-files -z | xargs -0 -I{} sh -c 'test $(wc -c < "{}") -gt 5000000 && echo "{}"' || true)"
if [[ -n "$BIG" ]]; then
  echo "ERROR: files over 5 MB would be published (data or outputs?):" >&2
  echo "$BIG" | sed 's/^/    /' >&2; exit 1
fi

if [[ -z "$STAGED" ]]; then echo "Nothing changed relative to $BRANCH."; exit 0; fi
echo "==> Changes relative to the current $BRANCH:"
git diff --cached --stat | tail -1
git diff --cached --name-status | awk '{c[$1]++} END {for (k in c) printf "    %s %d file(s)\n", (k=="A"?"added   ":k=="M"?"modified":k=="D"?"deleted ":k), c[k]}'
echo "    jasmap/ source files: $SRC_N; participant/genotype data: none; files > 5 MB: none"

if [[ $DRY -eq 1 ]]; then echo "==> Dry run: nothing committed or pushed."; exit 0; fi

git -c user.useConfigOnly=true commit -q -m "$MESSAGE" 2>/dev/null || {
  echo "Git identity not set. Run once:  git config --global user.name \"Your Name\"; git config --global user.email you@example.org" >&2; exit 1; }
git tag -a "$TAG" -m "$MESSAGE"
echo "==> Pushing $BRANCH and tag $TAG (you may be asked for a Personal Access Token)"
git push -q origin "$BRANCH"
git push -q origin "$TAG"
echo "==> Done: ${REMOTE%.git}/releases/tag/$TAG"
