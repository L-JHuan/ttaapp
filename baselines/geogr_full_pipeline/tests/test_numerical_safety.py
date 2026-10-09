"""复现单卡非有限值污染更新、以及多卡不能一致退出的问题。"""

import json
import contextlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.bad_loss = False

    def forward(self, input_ids, labels, **kwargs):
        loss = (self.weight * input_ids.float()).square().mean()
        if self.bad_loss:
            loss = loss * float("nan")
        output = {"loss": loss, "lm_loss": loss.detach()}
        for key in ("prefix_loss", "prefix_loss_unweighted", "z1_accuracy", "z2_loss",
                    "z2_accuracy", "z12_loss", "z12_accuracy", "sibling_z2_loss"):
            output[key] = loss.detach()
        return output


def distributed_failure_worker(rank, root, failure):
    from geogr_full_pipeline.numerical_safety import NonFiniteTrainingError, guard_training

    cuda = os.environ.get("GEOGR_TEST_CUDA") == "1"
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if cuda else "gloo", rank=rank, world_size=2,
        init_method=Path(root, "init").as_uri(),
    )
    os.environ["GEOGR_DIAGNOSTIC_ROOT"] = root
    os.environ["GEOGR_STAGE"] = "test_em"
    try:
        base = TinyModel().to(device)
        base.bad_loss = failure == "loss" and rank == 1
        model = torch.nn.parallel.DistributedDataParallel(
            base, device_ids=[rank] if cuda else None)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        result = {"caught": False}
        try:
            with guard_training(model, optimizer, device, epoch=1, global_step=7):
                count = 4 if failure == "finite_accum" else 1
                for micro in range(count):
                    # 工业配置使用四步累积，检查保护规约不会破坏 DDP.no_sync。
                    context = model.no_sync() if micro < count - 1 else contextlib.nullcontext()
                    with context:
                        output = model(input_ids=torch.ones(1, 3, device=device),
                                       labels=torch.ones(1, 3, dtype=torch.long, device=device))
                        (output["loss"] / count).backward()
                # 在真实 DDP 归约之后，仅损坏一个 rank 的梯度，检查更新前一致退出。
                if failure == "gradient" and rank == 1:
                    base.weight.grad.fill_(float("inf"))
                optimizer.step()
        except NonFiniteTrainingError as error:
            result.update(caught=True, message=str(error))
        result["weight"] = float(base.weight.detach().cpu())
        Path(root, f"result_{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class NumericalSafetyTest(unittest.TestCase):
    def test_nonfinite_loss_stops_before_backward_and_update(self):
        from geogr_full_pipeline.numerical_safety import NonFiniteTrainingError, guard_training

        model = TinyModel()
        model.bad_loss = True
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(NonFiniteTrainingError, "loss"):
            with guard_training(model, optimizer, torch.device("cpu"), epoch=1, global_step=0):
                model(input_ids=torch.ones(1, 3), labels=torch.ones(1, 3, dtype=torch.long))
        self.assertIsNone(model.weight.grad)
        self.assertEqual(float(model.weight.detach()), 1.0)
        self.assertFalse(model._forward_hooks)
        self.assertFalse(optimizer._optimizer_step_pre_hooks)

    def test_nonfinite_gradient_does_not_update_weight(self):
        from geogr_full_pipeline.numerical_safety import NonFiniteTrainingError, guard_training

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(NonFiniteTrainingError, "gradient"):
            with guard_training(model, optimizer, torch.device("cpu"), epoch=1, global_step=0):
                model(input_ids=torch.ones(1, 3), labels=torch.ones(1, 3, dtype=torch.long))["loss"].backward()
                model.weight.grad.fill_(float("nan"))
                optimizer.step()
        self.assertEqual(float(model.weight.detach()), 1.0)

    def test_missing_supervised_target_is_rejected(self):
        from geogr_full_pipeline.numerical_safety import NonFiniteTrainingError, guard_training

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(NonFiniteTrainingError, "labels"):
            with guard_training(model, optimizer, torch.device("cpu"), epoch=1, global_step=0):
                model(input_ids=torch.ones(1, 3), labels=torch.full((1, 3), -100))

    def test_finite_gradient_accumulation_is_unchanged(self):
        from geogr_full_pipeline.numerical_safety import guard_training

        reference, guarded = TinyModel(), TinyModel()
        for model, enabled in ((reference, False), (guarded, True)):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            from contextlib import nullcontext
            context = guard_training(model, optimizer, torch.device("cpu"), epoch=1, global_step=0) if enabled else nullcontext()
            with context:
                for step in range(4):
                    loss = model(input_ids=torch.ones(1, 3), labels=torch.ones(1, 3, dtype=torch.long))["loss"]
                    (loss / 2).backward()
                    if step % 2 == 1:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(reference.weight, guarded.weight))

    def test_original_loop_reproduces_nan_and_isolated_wrapper_stops_it(self):
        from tap_sid.train_tap_sid import train_epoch
        from geogr_full_pipeline.numerical_safety import NonFiniteTrainingError, install_training_guard

        args = SimpleNamespace(grad_accum=1, lm_loss_weight=1.0, alpha_prefix=0.0,
                               logging_steps=120, max_steps=0)
        batch = {"input_ids": torch.ones(1, 3), "labels": torch.ones(1, 3, dtype=torch.long)}
        original = TinyModel()
        original.bad_loss = True
        optimizer = torch.optim.SGD(original.parameters(), lr=0.1)
        # 旧循环会正常返回，但已把参数更新为 NaN，明确复现原缺陷。
        train_epoch(original, [batch], optimizer, torch.device("cpu"), args, 1, 0, False, 0)
        self.assertTrue(torch.isnan(original.weight).all())
        trainer = SimpleNamespace(train_epoch=train_epoch)
        install_training_guard(trainer)
        guarded = TinyModel()
        guarded.bad_loss = True
        optimizer = torch.optim.SGD(guarded.parameters(), lr=0.1)
        with self.assertRaises(NonFiniteTrainingError):
            trainer.train_epoch(guarded, [batch], optimizer, torch.device("cpu"), args, 1, 0, False, 0)
        self.assertEqual(float(guarded.weight.detach()), 1.0)
        self.assertIsNone(guarded.weight.grad)

    def test_nonfinite_parameter_after_optimizer_is_rejected(self):
        from geogr_full_pipeline.numerical_safety import NonFiniteTrainingError, guard_training

        class BrokenOptimizer(torch.optim.SGD):
            def step(self, closure=None):
                result = super().step(closure)
                self.param_groups[0]["params"][0].data.fill_(float("nan"))
                return result

        model = TinyModel()
        optimizer = BrokenOptimizer(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(NonFiniteTrainingError, "parameters_after_update"):
            with guard_training(model, optimizer, torch.device("cpu"), epoch=1, global_step=0):
                model(input_ids=torch.ones(1, 3), labels=torch.ones(1, 3, dtype=torch.long))["loss"].backward()
                optimizer.step()

    def run_distributed_failure(self, failure):
        with tempfile.TemporaryDirectory() as root:
            processes = mp.spawn(distributed_failure_worker, args=(root, failure), nprocs=2, join=False)
            deadline = time.monotonic() + 45
            try:
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("多卡非有限值检测未在45秒内一致退出")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            for rank in range(2):
                result = json.loads(Path(root, f"result_{rank}.json").read_text())
                if failure == "finite_accum":
                    self.assertFalse(result["caught"])
                    self.assertAlmostEqual(result["weight"], 0.8, places=6)
                    continue
                self.assertTrue(result["caught"])
                self.assertEqual(result["weight"], 1.0)
                self.assertIn("bad_ranks=[1]", result["message"])
                records = [json.loads(line) for line in Path(root, f"test_em.rank_{rank}.jsonl").read_text().splitlines()]
                record = records[-1]
                self.assertEqual(record["bad_ranks"], [1])
                self.assertEqual(record["global_step"], 7)
                self.assertEqual(record["local_bad"], rank == 1)
                self.assertNotIn("input_ids", json.dumps(record))

    def test_one_rank_nan_loss_stops_all_ranks(self):
        self.run_distributed_failure("loss")

    def test_one_rank_inf_gradient_stops_all_ranks(self):
        self.run_distributed_failure("gradient")

    def test_ddp_four_micro_steps_preserve_no_sync_accumulation(self):
        self.run_distributed_failure("finite_accum")

    def test_em_sft_and_cpt_are_all_guarded(self):
        root = Path(__file__).resolve().parents[1] / "geogr_full_pipeline"
        self.assertIn("install_training_guard", (root / "train_matched_sft.py").read_text())
        self.assertIn("guard_training", (root / "train_cpt.py").read_text())

    def test_old_nan_training_summary_is_not_reused(self):
        from geogr_full_pipeline.runtime_checks import checkpoint_ready

        with tempfile.TemporaryDirectory() as output:
            checkpoint = Path(output)
            adapter = checkpoint / "final_adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text('{"r": 16}')
            (adapter / "tokenizer_config.json").write_text('{}')
            (adapter / "adapter_model.bin").write_bytes(b"fixture")
            summary = checkpoint / "training_summary.json"
            summary.write_text(json.dumps({"global_step": 5, "epoch_losses": [0.5]}))
            self.assertTrue(checkpoint_ready(checkpoint, "final_adapter"))
            for value in (float("nan"), float("inf")):
                summary.write_text(json.dumps({"global_step": 5, "epochs": [{"train": {"lm_loss": value}}]}))
                self.assertFalse(checkpoint_ready(checkpoint, "final_adapter"))

    def test_numerical_error_survives_long_launcher_summary(self):
        from geogr_full_pipeline.summarize_stage_error import summarize

        with tempfile.TemporaryDirectory() as output:
            log = Path(output) / "failure.log"
            log.write_text('GEOGR_NONFINITE phase=loss bad_ranks=[1]\n' + 'launcher summary\n' * 180)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                summarize(log)
            self.assertIn("GEOGR_NONFINITE phase=loss bad_ranks=[1]", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
