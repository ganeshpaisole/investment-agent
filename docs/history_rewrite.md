# Repository history rewrite — March 6, 2026

What I did
- Removed plaintext secret files from the repository history: 
  - `FMP Key - BtItM5kH3USHa9aAWthBdbrTb.txt`
  - `Google Studio API key.txt`
  - `sk-proj-myWCF7iUScW26V2xh15Ax45J0hD.txt`

- Performed a destructive history rewrite using `git-filter-repo` in a
  mirror clone and force-pushed the rewritten history to `origin`.

Why this matters
- The sensitive files were deleted from the current tree earlier, but
  they remained in prior commits. The rewrite removes them from all
  commits, minimizing the risk of accidental disclosure.

Important: follow these steps to update your local clones

1. Backup any local changes (stash or patch them).
2. Re-clone the repository (recommended):

```
git clone https://github.com/ganeshpaisole/investment-agent.git
```

OR, if you prefer to update an existing clone:

```
git fetch origin --prune
git checkout main
git reset --hard origin/main
git clean -fdx
```

Notes & gotchas
- Pull request refs (the hidden `refs/pull/*` refs) are not rewriteable
  on GitHub; the rewrite updated branches and tags but GitHub still
  maintains historical PR refs. See GitHub's documentation for details.
- Because history was rewritten, any collaborators who have forks or
  local branches based on the old history must rebase or re-clone.

If you need help rotating any of the exposed keys at their issuing
provider (FMP, Google, etc.), tell me which provider(s) and I will
provide a rotation checklist and, where possible, API commands.
