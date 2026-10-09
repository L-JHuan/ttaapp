"""用小型随机 Llama 和真实 Llama tokenizer 验证双卡全链路及续跑。"""

import argparse
import hashlib
import json
from pathlib import Path


def hashes(root):
    paths = [root / "p2p/refined_embeddings.npz", root / "sid/initial_rq_sid.csv"]
    paths += list(root.glob("**/adapter_model.*"))
    return {str(path.relative_to(root)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "mtime_ns": path.stat().st_mtime_ns} for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "check", "check_resume", "prepare_eval_retry", "check_eval_retry"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    root, run = args.root, args.root / "run"
    if args.mode == "prepare":
        import pandas as pd
        import torch
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

        root.mkdir(parents=True, exist_ok=False)
        base, data = root / "base", root / "processed"
        (data / "v1_sequence").mkdir(parents=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(base)
        torch.manual_seed(42)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=4096,
        ))
        model.save_pretrained(base)
        pd.DataFrame({"pid": range(12), "latitude": [31 + i * .001 for i in range(12)],
                      "longitude": [121 + i * .001 for i in range(12)]}).to_csv(data / "poi_info.csv", index=False)
        pd.DataFrame({"pid": range(12), "l1_label": [0] * 12, "l2_label": [0] * 12}).to_csv(data / "role_priors.csv", index=False)
        sequence = str(list(range(12)))
        times = str([f"2026-01-01 {i:02}:00:00" for i in range(12)])
        for split, count in (("train", 8), ("test", 4)):
            pd.DataFrame({"UserId": range(count), "sequence_PoiId": [sequence] * count,
                          "sequence_UTCTimeOffset": [times] * count}).to_csv(data / f"v1_sequence/{split}_poi_sequence.csv", index=False)
        checkpoint = run / "em/iteration_1/checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "interrupted.txt").write_text("preserve_this_failure", encoding="utf-8")
        config = {"REPO_ROOT": args.repo, "PROCESSED_ROOT": data, "RUN_ROOT": run,
                  "BASE_MODEL": base, "CODEBOOK_SIZE": 4, "GPUS": "0,1", "REUSE_PREPROCESSED": 1,
                  "NO_VALIDATION": 1, "P2P_BATCH_SIZE": 2, "P2P_GRAD_ACCUM": 1, "P2P_EPOCHS": 1,
                  "EM_ITERATIONS": 2, "EM_EPOCHS": 1, "EM_BEAMS": 20, "SFT_GRAD_ACCUM": 1,
                  "CPT_EPOCHS": 1, "CPT_GRAD_ACCUM": 1, "SFT_EPOCHS": 1, "TEST_BEAMS": 10,
                  "PYTHON_BIN": "python", "WAIT_FOR_GPUS": 0}
        (root / "smoke.env").write_text("\n".join(f"{k}='{v}'" for k, v in config.items()) + "\n", encoding="utf-8")
    elif args.mode == "check":
        for iteration in (1, 2):
            report = json.loads((run / f"em/iteration_{iteration}/em_refinement_report.json").read_text())
            assert report["catalog_pois"] == report["final_unique_paths"] == 12
        metrics = json.loads((run / "cpt_sft/eval/test_metrics.json").read_text())
        assert metrics["samples"] == 4 and metrics["num_shards"] == 2
        assert metrics["illegal_sid_predictions"] == 0
        assert list(run.glob("em/iteration_1/checkpoint.incomplete_*/interrupted.txt"))
        logs = list((run / "logs").glob("*.log"))
        assert all(path.stat().st_size > 0 for path in logs)
        for name in ("p2p_train", "initial_rq", "em_1_train", "em_2_train", "cpt_train", "cpt_sft_train", "cpt_sft_eval"):
            assert any("DONE exit=0" in path.read_text() for path in (run / "logs").glob(name + "_*.log")), name
        for stage in ("em_1_train", "em_2_train", "cpt_train", "cpt_sft_train"):
            for rank in (0, 1):
                paths = list((run / "logs").glob(f"ddp_*/{stage}.rank_{rank}.jsonl"))
                assert len(paths) == 1, (stage, rank, paths)
                snapshots = [json.loads(line) for line in paths[0].read_text().splitlines()]
                assert {"before_ddp", "after_empty_cache", "after_ddp"} <= {s["phase"] for s in snapshots}
                for snapshot in snapshots:
                    assert int(snapshot["rank"]) == rank
                    assert snapshot["parameter_bytes"] > 0
                    assert snapshot["device_free_bytes"] > 0
                    assert snapshot["transport"]["NCCL_P2P_DISABLE"] == "1"
                    assert snapshot["transport"]["NCCL_SHM_DISABLE"] == "1"
        assert any("DONE exit=0" in path.read_text() for path in (run / "logs").glob("nccl_preflight_compat_*.log"))
        (root / "before_resume.json").write_text(json.dumps(hashes(run)), encoding="utf-8")
        print("FULL_PIPELINE_SMOKE_OK: P2P/RQ/two-EM/CPT/SFT/2-shard-beam10; 4 test samples; all stage logs nonempty", flush=True)
    elif args.mode == "check_resume":
        assert hashes(run) == json.loads((root / "before_resume.json").read_text())
        print("RESUME_REUSES_ALL_COMPLETED_WEIGHTS_AND_RQ_OK", flush=True)
    elif args.mode == "prepare_eval_retry":
        path = run / "cpt_sft/eval/test_metrics.json"
        path.rename(path.with_name("test_metrics.completed_fixture.json"))
        print("模拟评估仅预测落盘：保留旧分片及全部训练权重", flush=True)
    else:
        assert hashes(run) == json.loads((root / "before_resume.json").read_text())
        metrics = json.loads((run / "cpt_sft/eval/test_metrics.json").read_text())
        assert metrics["samples"] == 4
        assert len(list((run / "cpt_sft/eval/shards").iterdir())) == 2
        assert list((run / "cpt_sft/eval").glob("test_predictions.json.incomplete_*"))
        print("EVAL_RETRY_PRESERVES_OLD_SHARDS_AND_TRAINING_OK", flush=True)


if __name__ == "__main__":
    main()
