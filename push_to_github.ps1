<#
.SYNOPSIS
    Initialise the JasMAP4 project as a Git repository and push it to GitHub.

.DESCRIPTION
    Safe, idempotent publishing script:
      - Verifies git (and optionally GitHub CLI) are installed
      - Writes a .gitignore suited to this project (excludes generated
        outputs, test data, virtualenvs, __pycache__, logs)
      - Initialises the repo if needed, or reuses an existing one
      - Creates the GitHub repository if GitHub CLI is available
      - Commits and pushes to the branch of your choice

    NO credentials are stored in or read by this script. Authentication is
    delegated to GitHub CLI (`gh auth login`) or Git Credential Manager,
    both of which prompt you securely and cache the token in the OS
    credential store.

.PARAMETER RepoName
    Name of the GitHub repository. Default: JasMAP4

.PARAMETER GitHubUser
    GitHub account name. Default: echimusa

.PARAMETER ProjectPath
    Local folder to publish. Default: the folder containing this script.

.PARAMETER Branch
    Branch name to push. Default: main

.PARAMETER CommitMessage
    Commit message. Default: an auto-generated message with a timestamp.

.PARAMETER Tag
    Optional. Create an annotated git tag with this name after a successful
    push (e.g. -Tag v4.0), so the exact published state stays recoverable by
    name even as development continues past it.

.PARAMETER Private
    Create the GitHub repository as private (only used when the repo is
    created by this script via GitHub CLI).

.PARAMETER DryRun
    Show every action without changing anything or contacting GitHub.

.EXAMPLE
    .\push_to_github.ps1
    Publish the current folder to https://github.com/echimusa/JasMAP4

.EXAMPLE
    .\push_to_github.ps1 -RepoName JasMAP -Private -CommitMessage "Initial release"

.EXAMPLE
    .\push_to_github.ps1 -DryRun
    Preview everything without making changes.

.NOTES
    If PowerShell blocks this script, run once in the same session:
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
#>

[CmdletBinding()]
param(
    [string] $RepoName      = "JasMAP4",
    [string] $GitHubUser    = "echimusa",
    [string] $ProjectPath   = $PSScriptRoot,
    [string] $Branch        = "main",
    [string] $CommitMessage = "",
    [string] $Tag           = "",
    [switch] $Private,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------- helpers ---

function Write-Step  { param([string]$m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok    { param([string]$m) Write-Host "    [OK] $m"   -ForegroundColor Green }
function Write-Warn2 { param([string]$m) Write-Host "    [!]  $m"   -ForegroundColor Yellow }
function Write-Info  { param([string]$m) Write-Host "    $m"        -ForegroundColor Gray }

function Invoke-Step {
    <# Run a command unless -DryRun. Throws on non-zero exit code. #>
    param(
        [Parameter(Mandatory)][string]   $Exe,
        [Parameter(Mandatory)][string[]] $Arguments,
        [switch] $AllowFailure
    )
    $display = "$Exe $($Arguments -join ' ')"
    if ($DryRun) { Write-Info "[dry-run] $display"; return "" }

    Write-Info $display
    $output = & $Exe @Arguments 2>&1
    if ($LASTEXITCODE -ne 0 -and -not $AllowFailure) {
        throw "Command failed (exit $LASTEXITCODE): $display`n$output"
    }
    return ($output | Out-String).Trim()
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# ------------------------------------------------------------------ banner ---

Write-Host ""
Write-Host "=========================================================" -ForegroundColor White
Write-Host "  Publish project to GitHub" -ForegroundColor White
Write-Host "=========================================================" -ForegroundColor White

if ($DryRun) { Write-Warn2 "DRY RUN - nothing will be changed or uploaded." }

# --------------------------------------------------------- 1. prerequisites ---

Write-Step "Checking prerequisites"

if (-not (Test-Command git)) {
    throw "git is not installed or not on PATH. Install from https://git-scm.com/download/win then reopen PowerShell."
}
Write-Ok "git found: $((& git --version) -replace 'git version ','')"

$HasGh = Test-Command gh
if ($HasGh) {
    Write-Ok "GitHub CLI found - can create the remote repository automatically"
} else {
    Write-Warn2 "GitHub CLI (gh) not found. The repository must already exist on GitHub."
    Write-Info  "Optional: install from https://cli.github.com/ to have it created for you."
}

# ------------------------------------------------------- 2. resolve project ---

Write-Step "Resolving project folder"

if ([string]::IsNullOrWhiteSpace($ProjectPath)) { $ProjectPath = (Get-Location).Path }
if (-not (Test-Path -LiteralPath $ProjectPath)) { throw "Project path not found: $ProjectPath" }

$ProjectPath = (Resolve-Path -LiteralPath $ProjectPath).Path
Set-Location -LiteralPath $ProjectPath
Write-Ok "Project folder: $ProjectPath"

# Sanity check: does this look like the JasMAP project?
$expected = @("JasMAP4.py", "jasmap", "requirements.txt")
$missing  = $expected | Where-Object { -not (Test-Path -LiteralPath (Join-Path $ProjectPath $_)) }
if ($missing.Count -gt 0) {
    Write-Warn2 "Not all expected items found here (missing: $($missing -join ', '))."
    $ans = Read-Host "    Continue publishing this folder anyway? (y/N)"
    if ($ans -notmatch '^(y|yes)$') { Write-Host "`nAborted."; return }
} else {
    Write-Ok "JasMAP4 project structure detected"
}

$RemoteUrl = "https://github.com/$GitHubUser/$RepoName.git"
Write-Info "Target repository: https://github.com/$GitHubUser/$RepoName"

# ---------------------------------------------------------- 3. .gitignore ----

Write-Step "Writing .gitignore"

$GitIgnorePath = Join-Path $ProjectPath ".gitignore"

$GitIgnore = @'
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
'@

if ($DryRun) {
    Write-Info "[dry-run] would write $GitIgnorePath"
} elseif (Test-Path -LiteralPath $GitIgnorePath) {
    Write-Warn2 ".gitignore already exists - leaving it untouched"
    Write-Info  "Delete it and re-run if you want the recommended one."
} else {
    $GitIgnore | Set-Content -LiteralPath $GitIgnorePath -Encoding UTF8
    Write-Ok "Created .gitignore"
}

Write-Warn2 "Genotype files (.bed/.bim/.fam/.vcf) are excluded by default."
Write-Info  "Remove those lines from .gitignore only if the data is safe to publish."

# ------------------------------------------------------ 4. init repository ---

Write-Step "Preparing local Git repository"

if (Test-Path -LiteralPath (Join-Path $ProjectPath ".git")) {
    Write-Ok "Existing Git repository found - reusing it"
} else {
    Invoke-Step git @("init") | Out-Null
    Write-Ok "Initialised a new Git repository"
}

# Ensure identity is configured (repo-local fallback only; never overwrites global)
$userName  = (& git config user.name)  2>$null
$userEmail = (& git config user.email) 2>$null

if ([string]::IsNullOrWhiteSpace($userName) -or [string]::IsNullOrWhiteSpace($userEmail)) {
    Write-Warn2 "Git identity is not configured."
    if (-not $DryRun) {
        $inName  = Read-Host "    Your name for commits"
        $inEmail = Read-Host "    Your email for commits"
        Invoke-Step git @("config","user.name",  $inName)  | Out-Null
        Invoke-Step git @("config","user.email", $inEmail) | Out-Null
        Write-Ok "Identity set for this repository"
    }
} else {
    Write-Ok "Commit identity: $userName <$userEmail>"
}

# Branch
Invoke-Step git @("checkout","-B",$Branch) -AllowFailure | Out-Null
Write-Ok "On branch '$Branch'"

# ---------------------------------------------------- 5. create remote repo ---

Write-Step "Configuring GitHub remote"

$existingRemote = (& git remote get-url origin) 2>$null
if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($existingRemote)) {
    $existingRemote = $existingRemote.Trim()
    if ($existingRemote -ne $RemoteUrl) {
        Write-Warn2 "Remote 'origin' currently points to: $existingRemote"
        $ans = Read-Host "    Replace it with $RemoteUrl ? (y/N)"
        if ($ans -match '^(y|yes)$') {
            Invoke-Step git @("remote","set-url","origin",$RemoteUrl) | Out-Null
            Write-Ok "Remote updated"
        } else {
            Write-Info "Keeping the existing remote"
            $RemoteUrl = $existingRemote
        }
    } else {
        Write-Ok "Remote 'origin' already correct"
    }
} else {
    if ($HasGh) {
        # Does the repo already exist on GitHub?
        & gh repo view "$GitHubUser/$RepoName" 2>&1 | Out-Null
        $repoExists = ($LASTEXITCODE -eq 0)

        if ($repoExists) {
            Write-Ok "Repository already exists on GitHub"
            Invoke-Step git @("remote","add","origin",$RemoteUrl) | Out-Null
        } else {
            Write-Info "Repository does not exist yet - creating it via GitHub CLI"
            $visibility = if ($Private) { "--private" } else { "--public" }
            Invoke-Step gh @("repo","create","$GitHubUser/$RepoName",
                             $visibility,"--source",".","--remote","origin",
                             "--description","JasMAP4: Joint Ancestry and SNP Mapping for Multi-way Admixed Populations") | Out-Null
            Write-Ok "Created https://github.com/$GitHubUser/$RepoName"
        }
    } else {
        Invoke-Step git @("remote","add","origin",$RemoteUrl) | Out-Null
        Write-Ok "Remote 'origin' added"
        Write-Warn2 "Create the repository on GitHub first if it does not exist:"
        Write-Info  "https://github.com/new  (owner: $GitHubUser, name: $RepoName)"
    }
}

# ------------------------------------------------------------- 6. commit -----

Write-Step "Staging and committing"

Invoke-Step git @("add","-A") | Out-Null

$status = (& git status --porcelain 2>$null) | Out-String
if ([string]::IsNullOrWhiteSpace($status) -and -not $DryRun) {
    Write-Ok "Nothing to commit - working tree is clean"
} else {
    if (-not $DryRun) {
        $fileCount = ((& git diff --cached --name-only) | Measure-Object -Line).Lines
        Write-Info "$fileCount file(s) staged"

        # Safety check: confirm the source package actually made it into the
        # staging area. A .gitignore mistake here is SILENT -- git simply
        # omits the files with no error -- and is typically only discovered
        # much later, after the missing code has already been "published".
        # Worth one explicit assertion before anything leaves the machine.
        $srcStaged = ((& git diff --cached --name-only) |
                      Where-Object { $_ -like 'jasmap/*' } |
                      Measure-Object).Count
        if ($srcStaged -lt 5) {
            Write-Warn2 "Only $srcStaged file(s) from jasmap/ are staged - that looks wrong."
            Write-Warn2 "A .gitignore rule may be excluding source code. Check with:"
            Write-Info  "    git check-ignore -v jasmap/consensus_gwas/consensus_gwas.py"
            $ans = Read-Host "    Continue anyway? (y/N)"
            if ($ans -notmatch '^(y|yes)$') { Write-Host "`nAborted."; return }
        } else {
            Write-Ok "$srcStaged source file(s) from jasmap/ staged"
        }
    }
    if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
        $CommitMessage = "JasMAP4 update - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    }
    Invoke-Step git @("commit","-m",$CommitMessage) -AllowFailure | Out-Null
    Write-Ok "Committed: $CommitMessage"
}

# --------------------------------------------------------------- 7. push -----

Write-Step "Pushing to GitHub"

if (-not $HasGh -and -not $DryRun) {
    Write-Info "You may be prompted to sign in (Git Credential Manager)."
    Write-Info "If asked for a password, use a Personal Access Token, not your"
    Write-Info "account password: https://github.com/settings/tokens (scope: repo)"
}

Invoke-Step git @("push","-u","origin",$Branch) | Out-Null

# ---------------------------------------------------------------- 7b. tag ---

if (-not [string]::IsNullOrWhiteSpace($Tag)) {
    Write-Step "Tagging this state as '$Tag'"
    Invoke-Step git @("tag","-a",$Tag,"-m",$Tag) -AllowFailure | Out-Null
    Invoke-Step git @("push","origin",$Tag) -AllowFailure | Out-Null
    Write-Ok "Tag '$Tag' pushed - this state is now recoverable by name"
}

# -------------------------------------------------------------- 8. summary ---

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
if ($DryRun) {
    Write-Host "  DRY RUN COMPLETE - no changes were made" -ForegroundColor Yellow
} else {
    Write-Host "  SUCCESS" -ForegroundColor Green
    Write-Host ""
    Write-Host "  https://github.com/$GitHubUser/$RepoName" -ForegroundColor White
}
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""

if (-not $DryRun) {
    Write-Info "For subsequent updates, just run:"
    Write-Info "    git add -A"
    Write-Info "    git commit -m ""your message"""
    Write-Info "    git push"
    Write-Host ""
}
