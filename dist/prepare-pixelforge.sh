#!/usr/bin/env bash
# Materialize .pixelforge-patch/ from the upstream pixelforge tag plus the
# patch in patches/pixelforge.patch.
#
# Cargo's [patch] table can only point at a path that exists when the manifest
# is parsed, so the patched sources have to be on disk before cargo runs. This
# script fetches the tag, applies the patch, and leaves the result in
# .pixelforge-patch/ (gitignored, regenerated on demand).
#
# Run it before building on a fresh clone:
#   ./dist/prepare-pixelforge.sh && cargo build --release --workspace
set -euo pipefail

tag="v0.9.1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch_file="$repo_root/patches/pixelforge.patch"
out_dir="$repo_root/.pixelforge-patch"

if [[ ! -f "$patch_file" ]]; then
	echo "error: missing $patch_file" >&2
	exit 1
fi

# Reuse the tree cargo already fetched when possible, but export it with
# `git archive` so only committed content is copied — a checkout with stray
# local edits (a leftover debug dump, say) must not leak into the patch.
cache="${CARGO_HOME:-$HOME/.cargo}/git/checkouts"
src=""
if [[ -d "$cache" ]]; then
	for dir in "$cache"/pixelforge-*/*/; do
		if [[ -d "$dir/.git" ]]; then
			src="$dir"
			break
		fi
	done
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/pixelforge"

if [[ -n "$src" ]]; then
	git -C "$src" archive HEAD | tar -x -C "$tmp/pixelforge"
else
	git clone --quiet --depth 1 --branch "$tag" https://github.com/hgaiser/pixelforge "$tmp/pixelforge"
fi

git -C "$tmp/pixelforge" apply "$patch_file"

rm -rf "$out_dir"
cp -r "$tmp/pixelforge" "$out_dir"

echo "prepared $out_dir from pixelforge $tag + patches/pixelforge.patch"