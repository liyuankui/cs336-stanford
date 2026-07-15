#!/usr/bin/env bash
# Build contact sheets using a bash array of frame paths (bash 3.2 compatible).
# Layout: 8 columns x 10 rows = 80 frames per sheet, labelled with frame id + time.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
FRAMES_DIR="$DIR/frames"
OUT_DIR="$DIR/contacts"
mkdir -p "$OUT_DIR"

PER_SHEET=80
COLS=8
ROWS=10

# Load all frame paths into a bash array (bash 3.2: no mapfile, use while-read)
FRAMES=()
while IFS= read -r line; do
  FRAMES+=("$line")
done < <(ls "$FRAMES_DIR"/frame_*.png | sort)
TOTAL=${#FRAMES[@]}
echo "Total frames: $TOTAL"

nsheets=$(( (TOTAL + PER_SHEET - 1) / PER_SHEET ))
echo "Sheets: $nsheets"

for (( s=0; s<nsheets; s++ )); do
  start=$(( s * PER_SHEET ))
  end=$(( start + PER_SHEET ))
  (( end > TOTAL )) && end=$TOTAL

  # Build montage args: for each frame, emit "-label <text> <file>"
  args=()
  for (( i=start; i<end; i++ )); do
    fidx=$(( i + 1 ))              # 1-based frame number
    secs=$(( (fidx - 1) * 15 ))    # frame 1 at 0:00
    mm=$(( secs / 60 ))
    ss=$(( secs % 60 ))
    label=$(printf "f%d %02d:%02d" "$fidx" "$mm" "$ss")
    args+=(-label "$label" "${FRAMES[$i]}")
  done
  n=$(( end - start ))
  batch=$(printf "%02d" $((s+1)))
  out="$OUT_DIR/contact_batch_${batch}.png"

  montage "${args[@]}" \
    -tile "${COLS}x${ROWS}" \
    -geometry 220x124+4+4 \
    -background "#101010" \
    -fill "#dddddd" \
    -font "/System/Library/Fonts/Menlo.ttc" \
    -pointsize 11 \
    "$out"

  echo "Sheet $batch: $n frames -> $out"
done

echo "Done."
