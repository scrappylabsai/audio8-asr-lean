#!/usr/bin/env bash
# LibriSpeech test-clean + test-other (CC BY 4.0, openslr.org/12) into ./data, plus the
# telephone-quality copy and the long-form streams used in the README.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data
for s in test-clean test-other; do
  [ -d "LibriSpeech/$s" ] || curl -sSL "https://www.openslr.org/resources/12/$s.tar.gz" | tar xz
done
cd ..
python make_phone.py test-clean test-other
python make_longform.py --split test-clean --minutes 1 --seed 2
for s in 1 3 4; do python make_longform.py --split test-clean --minutes 10 --seed $s; done
for s in 5 6 7 8; do python make_longform.py --split test-clean --minutes 3 --seed $s; done
