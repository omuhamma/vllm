#!/bin/bash
# Inside the container: copy the K3.1 aiter + vLLM changes over the image's site-packages (backing up the originals).
# These 129 files are exactly what differs between the reference container and rocm/vllm-dev:nightly_455_wip.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); SP=${SP:-/opt/venv/lib/python3.12/site-packages}
want="0.9.2rc2.dev18209+g17a4dcc5d.rocm103"; have=$(python3 -c "import importlib.metadata as m; print(m.version('vllm'))")
[ "$have" = "$want" ] || { echo "vLLM in this image is $have, expected $want (wrong image): abort"; exit 1; }
BK=$HERE/backup_$(date +%Y%m%d_%H%M%S); mkdir -p $BK
cd $SP; tar cf $BK/site_packages_orig.tar --ignore-failed-read $(cat $HERE/overlay_files.txt) 2>/dev/null || true
cd $HERE/overlay; tar cf - $(cat $HERE/overlay_files.txt) | (cd $SP && tar xf -)
cd $SP && sha256sum -c --quiet $HERE/overlay.sha256 && echo "overlay installed: $(wc -l < $HERE/overlay_files.txt) files (originals in $BK)"
