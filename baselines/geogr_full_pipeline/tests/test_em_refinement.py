import unittest

from geogr_full_pipeline.em_refine_sid import (
    Candidate,
    CombinationTrie,
    complete_unique_assignment,
    parse_code,
)
from geogr_full_pipeline.merge_em_candidates import merge_candidate_shards


class BoundaryMergingTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        table = {
            "<a_0>": [1],
            "<b_0>": [2],
            "<c_0>": [3],
            "<a_0><b_0>": [4],
            "<b_0><c_0>": [5],
            "<a_0><b_0><c_0>": [6, 7],
        }
        return table[text]


class EmRefinementTest(unittest.TestCase):
    def test_merge_candidate_shards_requires_disjoint_complete_coverage(self):
        first = {1: [Candidate((0, 0, 0), 1.0, 1)]}
        second = {2: [Candidate((0, 0, 1), 1.0, 1)]}
        import tempfile
        from pathlib import Path
        from geogr_full_pipeline.common import write_json

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, shard in enumerate((first, second)):
                path = Path(directory) / f"shard_{index}.json"
                write_json(
                    path,
                    {
                        "candidates": {
                            str(pid): [
                                {
                                    "code": list(row.code),
                                    "score": row.score,
                                    "rank": row.rank,
                                }
                                for row in rows
                            ]
                            for pid, rows in shard.items()
                        }
                    },
                )
                paths.append(path)
            merged = merge_candidate_shards(paths, {1, 2})
        self.assertEqual(set(merged), {1, 2})

    def test_combination_trie_falls_back_for_boundary_merging_tokenizer(self):
        trie = CombinationTrie(BoundaryMergingTokenizer(), 1)
        self.assertEqual(trie.allowed([]), [6])
        self.assertEqual(trie.allowed([6]), [7])
        self.assertEqual(trie.allowed([6, 7]), [99])

    def test_parse_code_checks_three_layers_and_range(self):
        self.assertEqual(parse_code("<a_1><b_2><c_3>", 4), (1, 2, 3))
        self.assertIsNone(parse_code("<a_1><b_2><c_4>", 4))
        self.assertIsNone(parse_code("<a_1><b_2>", 4))

    def test_assignment_resolves_candidate_collision_by_score(self):
        candidates = {
            1: [Candidate((0, 0, 0), 0.9, 1), Candidate((0, 0, 1), 0.7, 2)],
            2: [Candidate((0, 0, 0), 0.8, 1), Candidate((0, 0, 2), 0.6, 2)],
        }
        prior = {1: (0, 0, 0), 2: (0, 0, 0)}
        assigned, report = complete_unique_assignment(candidates, prior, 3)
        self.assertEqual(assigned[1], (0, 0, 0))
        self.assertEqual(assigned[2], (0, 0, 2))
        self.assertEqual(report["beam_assigned"], 2)
        self.assertEqual(len(set(assigned.values())), 2)

    def test_assignment_uses_local_fallback_when_beam_is_exhausted(self):
        candidates = {
            1: [Candidate((0, 0, 0), 0.9, 1)],
            2: [Candidate((0, 0, 0), 0.8, 1)],
        }
        prior = {1: (0, 0, 0), 2: (0, 0, 0)}
        assigned, report = complete_unique_assignment(candidates, prior, 2)
        self.assertEqual(len(set(assigned.values())), 2)
        self.assertEqual(report["local_fallback_assigned"], 1)


if __name__ == "__main__":
    unittest.main()
