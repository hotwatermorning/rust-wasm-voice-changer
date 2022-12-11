#! /bin/bash

set -e -u -o pipefail

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

SCRIPT_NAME="$(basename "$(realpath "$0")")"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

cd "$SCRIPT_DIR"

wasm-pack build --target web

echo "Copy into ../public/wasm-audio" 1>&2
rm -rf ../public/wasm-audio
cp -R ./pkg ../public/wasm-audio

echo "Finished." 1>&2