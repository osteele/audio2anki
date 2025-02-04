# Issue: Audio Cleaning Flag Design

## Current Implementation

The current implementation of audio cleaning flags has some potential usability issues:

### Flag Design
- Uses `--clean/--no-clean` where `--clean` means "force cleaning"
- Internal implementation uses modes "force"/"skip"/None which don't directly map to flag names
- `--force-clean` might better indicate that it will error if cleaning isn't possible

### Cache Behavior Issues
1. With `--no-clean`: Ignores cached files even if they exist
2. Without flags: Uses cached files even if HF_TOKEN is present
3. Cannot force re-cleaning of a file that's already cached without first deleting the cache file

### Default Behavior Complexity
- Clean if HF_TOKEN exists OR cached file exists
- Behavior changes based on presence of cache files
- Could lead to inconsistent results across different machines or runs

## Proposed Solution

Consider replacing the current flags with:

```
--force-clean  Force cleaning. Error if HF_TOKEN is missing.
--use-cache    Use cached clean files if they exist. Don't clean if no cache exists.
--no-clean     Skip cleaning entirely.
(default)      Clean if HF_TOKEN exists, otherwise skip.
```

Benefits:
1. More predictable behavior
2. Separates caching policy from cleaning policy
3. Simpler default that only depends on HF_TOKEN
4. Better communicates the "force" nature of the strict cleaning option

## Impact

This would be a breaking change for any scripts that use the current flags.
Consider implementing in next major version bump.
