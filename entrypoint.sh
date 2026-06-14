#!/bin/sh
set -e

TARGET="${SEPSIS_CHECKPOINT_PATH:-/app/model.pt}"

if [ -n "$SEPSIS_CHECKPOINT_URL" ]; then
  echo "Fetching checkpoint -> $TARGET"
  case "$SEPSIS_CHECKPOINT_URL" in
    s3://*)
      if command -v aws >/dev/null 2>&1; then
        aws s3 cp "$SEPSIS_CHECKPOINT_URL" "$TARGET"
      else
        echo "aws CLI missing"
        exit 1
      fi
      ;;
    gs://*)
      if command -v gsutil >/dev/null 2>&1; then
        gsutil cp "$SEPSIS_CHECKPOINT_URL" "$TARGET"
      else
        echo "gsutil missing"
        exit 1
      fi
      ;;
    *)
      curl -fSL "$SEPSIS_CHECKPOINT_URL" -o "$TARGET"
      ;;
  esac
fi

if [ ! -f "$TARGET" ]; then
  echo "Checkpoint missing at $TARGET"
  exit 1
fi

exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
