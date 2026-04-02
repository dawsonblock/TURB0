#!/bin/bash
git init
git checkout -b main
git remote add origin https://github.com/dawsonblock/TurboQuantX1.git || true
git add .
git commit -m "chore: initial commit including benchmarking fixes and updated README for TurboQuantX1"
git push -u origin main
