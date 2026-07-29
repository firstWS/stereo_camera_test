# Object Anchor R&D archive

- Purpose: separate historical, non-runtime R&D artifacts from the active MVP workspace.
- Archived at: 2026-07-29T06:06:57.567175+00:00
- Files moved: 1402
- Bytes moved: 590729569
- This is a preservation move, not permanent deletion.
- The phase-1 reference graph classified every moved file as ARCHIVE_RND with no current reference.

## Restore

For each row in `archive_manifest.csv`, move `archived_path` back to
`original_path`, preserving the recorded SHA-256. Restore only on the cleanup
branch or after creating another backup point.
