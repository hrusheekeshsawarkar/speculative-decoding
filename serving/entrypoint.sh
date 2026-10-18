#!/bin/sh
# Install the server and core dependencies into the PVC on first start, then run the server.
# The requirements hash invalidates this install automatically when dependencies change.
set -eu

LIBS=/cache/pylibs
STAMP="$LIBS/.installed-$(sha256sum /app/requirements.txt | cut -d' ' -f1)"

if [ ! -f "$STAMP" ]; then
  echo "Installing inference dependencies into $LIBS (first start, takes a few minutes) ..."
  rm -rf "$LIBS"
  mkdir -p "$LIBS" /cache/tmp
  export TMPDIR=/cache/tmp
  pip install --no-cache-dir --target "$LIBS" -r /app/requirements.txt
  touch "$STAMP"
fi

export PYTHONPATH="$LIBS:${PYTHONPATH:-}"
exec python3 /app/server.py
