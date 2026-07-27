from types import SimpleNamespace
import unittest

from gnpr_baseline.export_codebook import (
    add_collision_leaf,
    model_config_from_checkpoint,
    sid_tokens,
)


class CodebookExportTest(unittest.TestCase):
    def test_collision_leaf_is_deterministic(self) -> None:
        raw = {
            8: [1, 2, 3],
            2: [1, 2, 3],
            5: [4, 5, 6],
            9: [1, 2, 3],
        }
        result = add_collision_leaf(raw)
        self.assertEqual(
            result,
            {
                2: [1, 2, 3, 0],
                5: [4, 5, 6],
                8: [1, 2, 3, 1],
                9: [1, 2, 3, 2],
            },
        )
        self.assertEqual(sid_tokens(result[2]), "<a_1><b_2><c_3><d_0>")

    def test_old_checkpoint_args_are_supported(self) -> None:
        old = SimpleNamespace(
            num_emb_list=[64, 64, 64],
            e_dim=64,
            layers=[512, 256, 128],
            dropout_prob=0.1,
            bn=True,
            loss_type="mse",
            quant_loss_weight=0.5,
            beta=0.25,
            kmeans_init=True,
            kmeans_iters=100,
            sk_epsilons=[0.1, 0.1, 0.1],
            sk_iters=50,
            use_linear=1,
        )
        config = model_config_from_checkpoint({"args": old}, 79)
        self.assertEqual(config["in_dim"], 79)
        self.assertEqual(config["num_emb_list"], [64, 64, 64])


if __name__ == "__main__":
    unittest.main()
