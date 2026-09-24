#!/usr/bin/env python3
"""Run the three-view cinematic pipeline sequentially over object case folders."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path('/home/z00566689/dev/mnt/jiang_test/B004_物体')
DEFAULT_OUTPUT = Path('/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/output')
DEFAULT_PROMPT = (
    'A cinematic camera move around {object}. Preserve the subject identity, '
    'shape, materials, colors, and lighting across the entire shot. '
    'Smooth realistic camera motion and coherent background.'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path,
                        default=SCRIPT_DIR / 'configs/three_view_demo.json')
    parser.add_argument('--batch-seed', type=int, default=42)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--limit', type=int, help='Run only the first N cases for a smoke test')
    return parser.parse_args()


def case_options(name: str, batch_seed: int) -> dict:
    digest = hashlib.sha256(f'{batch_seed}:{name}'.encode('utf-8')).digest()
    rng = random.Random(int.from_bytes(digest[:8], 'big'))
    return {'seed': rng.randrange(2**31)}


def main():
    args = parse_args()
    if not args.batch_root.is_dir():
        raise FileNotFoundError(f'Batch root missing: {args.batch_root}')
    if not args.config.is_file():
        raise FileNotFoundError(f'Config missing: {args.config}')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    # A batch must select the three images from each case, not explicit paths
    # left over from a previous single-case run.
    configured = json.loads(args.config.read_text(encoding='utf-8'))
    if any(configured.get(key) for key in ('left', 'middle', 'right')):
        raise ValueError('Clear left/middle/right in the config before batch mode')
    cases = sorted(case for case in args.batch_root.iterdir()
                   if case.is_dir() and (case / 'image').is_dir())
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError(f'No case/image directories found in {args.batch_root}')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    batch_output = args.output_root / f'{args.batch_root.name}_{run_id}'
    batch_output.mkdir(parents=True, exist_ok=False)
    print(f'Batch output: {batch_output}', flush=True)
    results = []
    from run_three_view_demo import main as run_case
    da3_cache, video_cache = {}, {}
    for index, case in enumerate(cases, 1):
        destination = batch_output / case.name
        destination.mkdir(parents=True, exist_ok=False)
        options = case_options(case.name, args.batch_seed)
        prompt = case / 'prompt.txt'
        if not prompt.is_file():
            prompt = destination / 'prompt.txt'
            object_name = case.name.split('_', 1)[-1].replace('_', ' ').replace('-', ' ')
            prompt.write_text(DEFAULT_PROMPT.format(object=object_name) + '\n', encoding='utf-8')
        entry = {'case': case.name, 'image_dir': str(case / 'image'),
                 'output_dir': str(destination), 'prompt': str(prompt),
                 'options': options}
        command = [sys.executable, str(SCRIPT_DIR / 'run_three_view_demo.py'),
            '--config', str(args.config), '--images-dir', str(case / 'image'),
            '--output-dir', str(destination), '--prompt', str(prompt)]
        for key, value in options.items():
            command.extend(['--' + key.replace('_', '-'), str(value)])
        if args.render_only:
            command.append('--render-only')
        print(f'[{index}/{len(cases)}] Running {case.name}: {options}', flush=True)
        try:
            run_case(command[2:], da3_cache=da3_cache, video_cache=video_cache)
            entry['status'] = 'ok'
            entry['returncode'] = 0
        except Exception as exc:
            import traceback
            traceback.print_exc()
            entry['status'] = 'failed'
            entry['returncode'] = 1
            entry['error'] = str(exc)
        results.append(entry)
        (batch_output / 'batch_summary.json').write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    failures = sum(entry['status'] == 'failed' for entry in results)
    print(f'Finished {len(results)} cases, {failures} failed. Summary: '
          f'{batch_output / "batch_summary.json"}')
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
