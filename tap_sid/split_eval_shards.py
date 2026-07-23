from __future__ import annotations

import argparse
import json
from pathlib import Path

from tap_sid.eval_sharding import write_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="将测试 JSON 均匀拆分为多个 GPU 评估分片。")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    manifest = write_shards(args.dataset, args.output_dir, args.num_shards)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
