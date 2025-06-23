#!/bin/bash
set -e

git submodule init
if [ ! -d "third_party/eigen" ]; then
  git submodule add https://gitlab.com/libeigen/eigen.git third_party/eigen
fi
git submodule update --recursive --remote
