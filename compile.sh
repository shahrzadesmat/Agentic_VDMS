#!/bin/bash
# Local compile script — works around tectonic's changes.sty conflict with acmart.
# Compiles a temporary copy with [final] mode (strips markup) and \@undefined fix.
# The real main.tex is NOT modified.

set -e
TECTONIC=/work/hdd/bdjd/tectonic_bin/tectonic
PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

cp "$PAPER_DIR"/main.tex "$TMPDIR"/main.tex
cp "$PAPER_DIR"/*.bib "$TMPDIR"/ 2>/dev/null || true
cp "$PAPER_DIR"/*.cls "$TMPDIR"/ 2>/dev/null || true
cp "$PAPER_DIR"/*.bst "$TMPDIR"/ 2>/dev/null || true
cp "$PAPER_DIR"/*.sty "$TMPDIR"/ 2>/dev/null || true
cp -r "$PAPER_DIR"/figures "$TMPDIR"/figures

# Fix 1: undefine \comment before changes.sty redefines it (version conflict)
# Fix 2: use [final] to strip markup (avoids \added etc. in output)
sed -i \
    -e 's/\\usepackage\[markup=colored\]{changes}/\\makeatletter\\expandafter\\let\\csname comment\\endcsname\\@undefined\\makeatother\n\\usepackage[final]{changes}/' \
    "$TMPDIR/main.tex"

cd "$TMPDIR"
echo "Compiling..."
$TECTONIC main.tex 2>&1 | grep -v "^note:"
if [ -f main.pdf ]; then
    cp main.pdf "$PAPER_DIR/main_local.pdf"
    echo ""
    echo "PDF written to: $PAPER_DIR/main_local.pdf"
else
    echo "FAILED — no PDF produced"
    exit 1
fi
