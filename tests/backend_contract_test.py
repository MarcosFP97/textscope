import unittest
from unittest import mock

import torch

from textscope.e5_backend import DEFAULT_E5_MODEL_REVISION
from textscope.relevance_analyzer import RelevanceAnalyzer
from textscope.subtheme_analyzer import SubthemeAnalyzer


class _Backend:
    def __init__(self):
        self.tokenizer = object()
        self.model = mock.Mock()
        self.device = torch.device('cpu')
        self.embed_calls = []
        self.iter_calls = []

    def embed(self, texts, batch_size=32):
        self.embed_calls.append((list(texts), batch_size))
        return torch.tensor([[1.0, 0.0]] * len(texts))

    def iter_embeddings(self, texts, batch_size=32):
        self.iter_calls.append((list(texts), batch_size))
        yield torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        yield torch.tensor([[0.5, 0.5]])


class BackendContractTests(unittest.TestCase):
    def test_default_model_revision_is_immutable(self):
        self.assertEqual(
            DEFAULT_E5_MODEL_REVISION,
            '274baa43b0e13e37fafa6428dbc7938e62e5c439',
        )

    def test_subtheme_instruction_uses_e5_query_prefix(self):
        analyzer = SubthemeAnalyzer(backend=_Backend())

        instruction = analyzer._get_detailed_instruct('Find the topic', 'security')

        self.assertEqual(
            instruction,
            'Instruct: Find the topic\nQuery: security',
        )

    def test_analyzers_share_the_injected_backend(self):
        backend = _Backend()
        relevance = RelevanceAnalyzer(backend=backend)
        subthemes = SubthemeAnalyzer(backend=backend)

        self.assertIs(relevance.backend, backend)
        self.assertIs(subthemes.backend, backend)
        self.assertIs(relevance.model, subthemes.model)

    def test_subthemes_stream_sentence_batches_and_keep_only_maxima(self):
        backend = _Backend()
        analyzer = SubthemeAnalyzer(backend=backend, batch_size=2)
        analyzer.subthemes = {'demo': [['one'], ['two']]}
        analyzer._get_kw_cache = mock.Mock(return_value=(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            {0: [0], 1: [1]},
        ))

        with mock.patch(
            'textscope.subtheme_analyzer.sent_tokenize',
            return_value=['one', 'two', 'three'],
        ) as tokenize:
            scores, count = analyzer._analyze_common('text', 'demo')

        self.assertEqual(count, 2)
        self.assertEqual(scores, [100.0, 100.0])
        self.assertEqual(
            backend.iter_calls,
            [(['one', 'two', 'three'], 2)],
        )
        tokenize.assert_called_once_with('text', language='spanish')


if __name__ == '__main__':
    unittest.main()
