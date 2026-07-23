#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$PWD}"

if [[ ! -d "$TARGET" ]]; then
  echo "Target repository does not exist: $TARGET" >&2
  exit 1
fi

if [[ ! -d "$TARGET/docs" ]]; then
  echo "Target does not look like the Helicopter repository: $TARGET" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$TARGET/backups/readme_before_${STAMP}"
mkdir -p "$BACKUP_DIR"

if [[ -f "$TARGET/README.md" ]]; then
  cp "$TARGET/README.md" "$BACKUP_DIR/README.md"
fi

mkdir -p "$TARGET/docs/assets/readme"
cp "$SOURCE_DIR/README.md" "$TARGET/README.md"
cp -a "$SOURCE_DIR/docs/assets/readme/." "$TARGET/docs/assets/readme/"

echo "README patch applied."
echo "Target: $TARGET/README.md"
echo "Assets: $TARGET/docs/assets/readme"
echo "Backup: $BACKUP_DIR"
