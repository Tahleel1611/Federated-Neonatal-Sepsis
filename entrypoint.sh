#!/bin/sh
set -e

TARGET="${SEPSIS_CHECKPOINT_PATH:-/app/model.pt}"

# Try to download checkpoint if URL is set and looks like a real URL
if [ -n "$SEPSIS_CHECKPOINT_URL" ] && echo "$SEPSIS_CHECKPOINT_URL" | grep -qE '^https?://'; then
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

# If no checkpoint file exists, generate a dummy one for demo purposes
if [ ! -f "$TARGET" ]; then
  echo "No checkpoint found at $TARGET - generating dummy checkpoint for demo..."
  python3 - <<'PYEOF'
import torch, os, sys
sys.path.insert(0, '/app')
from src.models.transformer_lstm import TransformerLSTMSepsisModel
from src.constants import MODEL_FEATURE_COLUMNS

target = os.environ.get('SEPSIS_CHECKPOINT_PATH', '/app/model.pt')
os.makedirs(os.path.dirname(target) if os.path.dirname(target) else '.', exist_ok=True)

input_size = len(MODEL_FEATURE_COLUMNS)
model = TransformerLSTMSepsisModel(input_size=input_size)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'd_model': 64,
    'num_heads': 4,
    'transformer_layers': 2,
    'lstm_hidden': 64,
    'dropout': 0.1,
    'seq_len_steps': 24,
    'best_threshold': 0.5,
}
torch.save(checkpoint, target)
print(f'Dummy checkpoint saved to {target} (input_size={input_size})')
PYEOF
fi

exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
