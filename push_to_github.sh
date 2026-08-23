#!/usr/bin/env bash
#
# push_to_github.sh — Linux/macOS port of push_to_github.ps1
# =============================================================================
# Initialise the JasMAP4 project as a Git repository and push it to GitHub.
#
# Why this exists: the original publishing script is PowerShell (.ps1) and
# therefore Windows-only. Development has since moved to Ubuntu/HPC, where
# pwsh is typically not installed. Same behaviour, same safety properties,
# native bash.
#
# Safe and idempotent:
#   - Verifies git (and optionally GitHub CLI) are installed
#   - Writes a .gitignore suited to this project
#   - Initialises the repo if needed, or reuses an existing one
#   - Creates the GitHub repository if GitHub CLI is available
#   - Commits and pushes to the branch of your choice
#
# NO credentials are stored in or read by this script. Authentication is
# delegated to GitHub CLI (`gh auth login`) or git-credential-manager /
# SSH keys, all of which prompt securely and cache outside this script.
#
# Usage:
#   ./push_to_github.sh                          # publish with defaults
#   ./push_to_github.sh --dry-run                # preview, change nothing
#   ./push_to_github.sh --repo JasMAP --private
#   ./push_to_github.sh --message "v4.0 baseline"
#   ./push_to_github.sh --tag v4.0               # also create + push a tag
#
# Options:
#   --repo NAME        GitHub repository name        (default: JasMAP4)
#   --user NAME        GitHub account name           (default: echimusa)
#   --path DIR         Local folder to publish       (default: script's dir)
#   --branch NAME      Branch to push                (default: main)
#   --message MSG      Commit message                (default: timestamped)
#   --tag TAG          Create and push an annotated tag after the push
#   --private          Create the GitHub repo private (only when gh creates it)
#   --dry-run          Show every action without changing anything
#   -h | --help        This help
# =============================================================================

set -uo pipefail

REPO_NAME="JasMAP4"
GITHUB_USER="echimusa"
PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="main"
COMMIT_MESSAGE=""
TAG_NAME=""
PRIVATE=0
DRY_RUN=0

C_CYAN='\033[0;36m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[0;33m'
C_GRAY='\033[0;37m'; C_WHITE='\033[1;37m'; C_RESET='\033[0m'

step()  { printf "\n${C_CYAN}==> %s${C_RESET}\n" "$1"; }
ok()    { printf "    ${C_GREEN}[OK] %s${C_RESET}\n" "$1"; }
warn2() { printf "    ${C_YELLOW}[!]  %s${C_RESET}\n" "$1"; }
info()  { printf "    ${C_GRAY}%s${C_RESET}\n" "$1"; }
die()   { printf "\n${C_YELLOW}ERROR: %s${C_RESET}\n\n" "$1" >&2; exit 1; }

run() {
  # Run a command unless --dry-run. Exits on failure unless ALLOW_FAIL=1.
  local display="$*"
  if [[ $DRY_RUN -eq 1 ]]; then info "[dry-run] $display"; return 0; fi
  info "$display"
  if ! "$@"; then
    [[ "${ALLOW_FAIL:-0}" -eq 1 ]] && return 1
    die "Command failed: $display"
  fi
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)    REPO_NAME="$2"; shift 2 ;;
    --user)    GITHUB_USER="$2"; shift 2 ;;
    --path)    PROJECT_PATH="$2"; shift 2 ;;
    --branch)  BRANCH="$2"; shift 2 ;;
    --message) COMMIT_MESSAGE="$2"; shift 2 ;;
    --tag)     TAG_NAME="$2"; shift 2 ;;
    --private) PRIVATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown option: $1  (try --help)" ;;
  esac
done

printf "\n${C_WHITE}=========================================================${C_RESET}\n"
printf "${C_WHITE}  Publish project to GitHub${C_RESET}\n"
printf "${C_WHITE}=========================================================${C_RESET}\n"
[[ $DRY_RUN -eq 1 ]] && warn2 "DRY RUN - nothing will be changed or uploaded."

# ------------------------------------------------------- 1. prerequisites ---
step "Checking prerequisites"
command -v git >/dev/null 2>&1 || die "git is not installed. Install with: sudo apt install git"
ok "git found: $(git --version | sed 's/git version //')"

HAS_GH=0
if command -v gh >/dev/null 2>&1; then
  HAS_GH=1; ok "GitHub CLI found - can create the remote repository automatically"
else
  warn2 "GitHub CLI (gh) not found. The repository must already exist on GitHub."
  info  "Optional: https://github.com/cli/cli#installation"
fi

# ----------------------------------------------------- 2. resolve project ---
step "Resolving project folder"
[[ -d "$PROJECT_PATH" ]] || die "Project path not found: $PROJECT_PATH"
PROJECT_PATH="$(cd "$PROJECT_PATH" && pwd)"
cd "$PROJECT_PATH" || die "Cannot cd to $PROJECT_PATH"
ok "Project folder: $PROJECT_PATH"

MISSING=()
for item in "JasMAP4.py" "jasmap" "requirements.txt"; do
  [[ -e "$PROJECT_PATH/$item" ]] || MISSING+=("$item")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  warn2 "Not all expected items found here (missing: ${MISSING[*]})."
  read -r -p "    Continue publishing this folder anyway? (y/N) " ans
  [[ "$ans" =~ ^[Yy]([Ee][Ss])?$ ]] || { printf "\nAborted.\n"; exit 0; }
else
  ok "JasMAP4 project structure detected"
fi

REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
info "Target repository: https://github.com/${GITHUB_USER}/${REPO_NAME}"

# ---------------------------------------------------------- 3. .gitignore ---
step "Writing .gitignore"
GITIGNORE_PATH="$PROJECT_PATH/.gitignore"

read -r -d '' GITIGNORE_BODY << 'GITIGNORE_EOF' || true
# ---- Python ----
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/

# ---- Virtual environments ----
venv/
venv*/
.venv/
env/
ENV/

# ---- JasMAP generated output ----
# NOTE: --consgwas writes its output to a SUBFOLDER literally named
# "consensus_gwas" inside whatever --out directory you pass (e.g.
# out_foo/consensus_gwas/). That is the SAME NAME as the source package
# jasmap/consensus_gwas/ (the actual 8-step ConsensusGWAS engine). A glob
# pattern like "*/consensus_gwas/" matches BOTH, and git ignores an entire
# directory tree once the folder itself matches -- so a naive pattern here
# would silently exclude the engine's source code from every commit with no
# warning. The patterns below only ever match generated *top-level* output
# folders (jasmap_results/, out_*/, results/), which already recursively
# cover their own nested consensus_gwas/rvat_results/etc. subfolders --
# no separate pattern for those subfolder names is needed or safe to add.
jasmap_results/
out_*/
results/

# Explicit safety net: never let any rule above accidentally exclude the
# actual jasmap/ source package, even if --out is pointed at the project root.
!/jasmap/
!/jasmap/**

# ---- JasMAP intermediate + log files ----
*.tmp.tped
jasmap-*.log
*.assoc
*.RVAT.assoc
*.RVAT.summary

# ---- Generated test data (regenerate with make_test_data.py) ----
test_data/
test_data_*/
test_small/
test_large/

# ---- Generated figures ----
# plots.py is source and IS tracked; these are run outputs only.
out_*/**/*.png
jasmap_results/**/*.png

# ---- OS / editor ----
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
*.swp
*~

# ---- Large data files (never commit genotype data) ----
# NOTE: *.bed is intended for PLINK BINARY GENOTYPE files, but ".bed" is
# ALSO a completely unrelated plain-text genome-annotation format (chrom/
# start/stop/name) -- which is exactly what --gene-annot consumes and what
# --annot writes. A bare "*.bed" therefore silently excludes annotation
# files too, with no warning: the same class of hazard as the
# consensus_gwas/ collision documented above, and verified with a real
# `git add -A` (a gene_annotation.bed was being dropped from every commit).
# PLINK genotype .bed files always travel with .bim/.fam siblings;
# annotation BEDs never do. The negations below re-include the
# annotation-style names this project actually produces, so a curated
# annotation can be version-controlled while real genotype data still
# cannot be committed by accident.
*.bed
!*annotation*.bed
!*annot*.bed
!*_regions.bed
*.bim
*.fam
*.vcf
*.vcf.gz
*.bgen
*.h5
*.hdf5
GITIGNORE_EOF

if [[ $DRY_RUN -eq 1 ]]; then
  info "[dry-run] would write $GITIGNORE_PATH"
elif [[ -f "$GITIGNORE_PATH" ]]; then
  warn2 ".gitignore already exists - leaving it untouched"
  info  "Delete it and re-run if you want the recommended one."
else
  printf '%s\n' "$GITIGNORE_BODY" > "$GITIGNORE_PATH"
  ok "Created .gitignore"
fi

warn2 "Genotype files (.bed/.bim/.fam/.vcf) are excluded by default."
info  "Remove those lines from .gitignore only if the data is safe to publish."

# ------------------------------------------------------ 4. init repository ---
step "Preparing local Git repository"
if [[ -d "$PROJECT_PATH/.git" ]]; then
  ok "Existing Git repository found - reusing it"
else
  run git init -q
  ok "Initialised a new Git repository"
fi

USER_NAME="$(git config user.name || true)"
USER_EMAIL="$(git config user.email || true)"
if [[ -z "$USER_NAME" || -z "$USER_EMAIL" ]]; then
  warn2 "Git identity is not configured."
  if [[ $DRY_RUN -eq 0 ]]; then
    read -r -p "    Your name for commits: " IN_NAME
    read -r -p "    Your email for commits: " IN_EMAIL
    run git config user.name  "$IN_NAME"
    run git config user.email "$IN_EMAIL"
    ok "Identity set for this repository"
  fi
else
  ok "Commit identity: $USER_NAME <$USER_EMAIL>"
fi

ALLOW_FAIL=1 run git checkout -B "$BRANCH" -q
ok "On branch '$BRANCH'"

# ---------------------------------------------------- 5. create remote repo ---
step "Configuring GitHub remote"
EXISTING_REMOTE="$(git remote get-url origin 2>/dev/null || true)"
if [[ -n "$EXISTING_REMOTE" ]]; then
  if [[ "$EXISTING_REMOTE" != "$REMOTE_URL" ]]; then
    warn2 "Remote 'origin' currently points to: $EXISTING_REMOTE"
    read -r -p "    Replace it with $REMOTE_URL ? (y/N) " ans
    if [[ "$ans" =~ ^[Yy]([Ee][Ss])?$ ]]; then
      run git remote set-url origin "$REMOTE_URL"; ok "Remote updated"
    else
      info "Keeping the existing remote"; REMOTE_URL="$EXISTING_REMOTE"
    fi
  else
    ok "Remote 'origin' already correct"
  fi
else
  if [[ $HAS_GH -eq 1 ]]; then
    if gh repo view "${GITHUB_USER}/${REPO_NAME}" >/dev/null 2>&1; then
      ok "Repository already exists on GitHub"
      run git remote add origin "$REMOTE_URL"
    else
      info "Repository does not exist yet - creating it via GitHub CLI"
      VIS="--public"; [[ $PRIVATE -eq 1 ]] && VIS="--private"
      run gh repo create "${GITHUB_USER}/${REPO_NAME}" "$VIS" \
          --source . --remote origin \
          --description "JasMAP4: Joint Ancestry and SNP Mapping for Multi-way Admixed Populations"
      ok "Created https://github.com/${GITHUB_USER}/${REPO_NAME}"
    fi
  else
    run git remote add origin "$REMOTE_URL"
    ok "Remote 'origin' added"
    warn2 "Create the repository on GitHub first if it does not exist:"
    info  "https://github.com/new  (owner: $GITHUB_USER, name: $REPO_NAME)"
  fi
fi

# ------------------------------------------------------------- 6. commit -----
step "Staging and committing"
run git add -A

STATUS="$(git status --porcelain 2>/dev/null || true)"
if [[ -z "$STATUS" && $DRY_RUN -eq 0 ]]; then
  ok "Nothing to commit - working tree is clean"
else
  if [[ $DRY_RUN -eq 0 ]]; then
    FILE_COUNT="$(git diff --cached --name-only | wc -l | tr -d ' ')"
    info "$FILE_COUNT file(s) staged"
    # Safety check that the .ps1 version does not do: confirm the source
    # package actually made it into the staging area. A .gitignore mistake
    # here is silent and only discovered much later, so it is worth one
    # explicit assertion before anything is published.
    SRC_STAGED="$(git diff --cached --name-only | grep -c '^jasmap/' || true)"
    if [[ "$SRC_STAGED" -lt 5 ]]; then
      warn2 "Only $SRC_STAGED file(s) from jasmap/ are staged - that looks wrong."
      warn2 "A .gitignore rule may be excluding source code. Check with:"
      info  "    git check-ignore -v jasmap/consensus_gwas/consensus_gwas.py"
      read -r -p "    Continue anyway? (y/N) " ans
      [[ "$ans" =~ ^[Yy]([Ee][Ss])?$ ]] || { printf "\nAborted.\n"; exit 1; }
    else
      ok "$SRC_STAGED source file(s) from jasmap/ staged"
    fi
  fi
  [[ -z "$COMMIT_MESSAGE" ]] && COMMIT_MESSAGE="JasMAP4 update - $(date '+%Y-%m-%d %H:%M')"
  ALLOW_FAIL=1 run git commit -m "$COMMIT_MESSAGE" -q
  ok "Committed: $COMMIT_MESSAGE"
fi

# --------------------------------------------------------------- 7. push -----
step "Pushing to GitHub"
if [[ $HAS_GH -eq 0 && $DRY_RUN -eq 0 ]]; then
  info "You may be prompted to sign in."
  info "If asked for a password, use a Personal Access Token, not your"
  info "account password: https://github.com/settings/tokens (scope: repo)"
fi
run git push -u origin "$BRANCH"

if [[ -n "$TAG_NAME" ]]; then
  step "Tagging this state as '$TAG_NAME'"
  ALLOW_FAIL=1 run git tag -a "$TAG_NAME" -m "$TAG_NAME"
  ALLOW_FAIL=1 run git push origin "$TAG_NAME"
  ok "Tag '$TAG_NAME' pushed - this state is now recoverable by name"
fi

# -------------------------------------------------------------- 8. summary ---
printf "\n${C_GREEN}=========================================================${C_RESET}\n"
if [[ $DRY_RUN -eq 1 ]]; then
  printf "${C_YELLOW}  DRY RUN COMPLETE - no changes were made${C_RESET}\n"
else
  printf "${C_GREEN}  SUCCESS${C_RESET}\n\n"
  printf "${C_WHITE}  https://github.com/${GITHUB_USER}/${REPO_NAME}${C_RESET}\n"
fi
printf "${C_GREEN}=========================================================${C_RESET}\n\n"

if [[ $DRY_RUN -eq 0 ]]; then
  info "For subsequent updates, just run:"
  info "    git add -A && git commit -m \"your message\" && git push"
  printf "\n"
fi
