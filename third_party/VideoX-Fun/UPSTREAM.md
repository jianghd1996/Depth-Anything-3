# Vendored VideoX-Fun inference subset

- Upstream: https://github.com/jianghd1996/VideoX-Fun
- Commit: `18b9b78d85b69edf483e9eeebaa057b39716aba1`
- Purpose: single-GPU Wan2.2-Fun-5B-Control inference with mask-aware control.

Only the transitive source files needed by
`scripts/world_model/generate_with_videox_fun.py` are vendored. The upstream
license is preserved in this directory. Local `__init__.py` files intentionally
export only the required Wan components so importing the subset does not pull in
unrelated VideoX-Fun model families.
