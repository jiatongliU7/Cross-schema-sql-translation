import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from sqltranslate.evaluation import write_json, read_json
from sqltranslate.gemini import (GeminiMatcher, TemporaryAPIError, APIConfigurationError,
                                  native_body, parse_mapping)


SOURCE = {'tables': [{'name': 'a', 'columns': [{'name': 'x', 'type': 'INTEGER'}]}]}
TARGET = {'tables': [{'name': 'b', 'columns': [{'name': 'y', 'type': 'INTEGER'}]}]}
MAPPING = {'tables': {'a': 'b'}, 'columns': {'a.x': 'b.y'}}


def response(mapping=MAPPING):
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(mapping)}]}}],
            'usageMetadata': {'totalTokenCount': 50}, 'modelVersion': 'test-version'}


def http_error(code, message='temporary', retry_after='0'):
    return urllib.error.HTTPError('https://example.invalid', code, 'failure',
         {'Content-Type': 'application/json', 'Retry-After': retry_after},
         io.BytesIO(json.dumps({'error': {'message': message}}).encode()))


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'GEMINI_API_KEY': 'secret-test-key'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.now, self.delays = 0, []

    def sleep(self, seconds):
        self.now += seconds
        self.delays.append(seconds)

    def matcher(self, **kwargs):
        return GeminiMatcher(self.root, 'test-model', [], self.root / 'logs',
                             sleep=self.sleep, clock=lambda: self.now, **kwargs)

    def test_native_format_and_truncation(self):
        example = {'source_schema': SOURCE, 'target_schema': TARGET, 'mapping': MAPPING}
        body = native_body(SOURCE, TARGET, [example])
        self.assertEqual([c['role'] for c in body['contents']], ['user', 'model', 'user'])
        self.assertNotIn('mapping', json.loads(body['contents'][-1]['parts'][0]['text']))
        payload = response()
        self.assertEqual(parse_mapping(payload, SOURCE, TARGET), MAPPING)
        payload['candidates'][0]['finishReason'] = 'MAX_TOKENS'
        with self.assertRaises(ValueError): parse_mapping(payload, SOURCE, TARGET)

    def test_retry_then_success_and_cache(self):
        matcher = self.matcher(max_attempts=3, min_interval=15)
        with patch('urllib.request.urlopen', side_effect=[http_error(503), io.BytesIO(json.dumps(response()).encode())]) as urlopen:
            self.assertEqual(matcher('pair', SOURCE, TARGET), MAPPING)
            self.assertEqual(matcher('pair', SOURCE, TARGET), MAPPING)
            self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(self.delays, [20])
        attempts = list((self.root / 'logs').glob('*.attempt_*.json'))
        self.assertEqual(len(attempts), 2)
        self.assertNotIn('secret-test-key', ''.join(p.read_text() for p in attempts))

    def test_exhausted_service_failure_can_resume(self):
        matcher = self.matcher(max_attempts=2)
        with patch('urllib.request.urlopen', side_effect=[http_error(503), http_error(503)]) as urlopen:
            with self.assertRaises(TemporaryAPIError): matcher('pair', SOURCE, TARGET)
            self.assertEqual(urlopen.call_count, 2)
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response()).encode())):
            self.assertEqual(matcher('pair', SOURCE, TARGET), MAPPING)
        self.assertEqual(len(list((self.root / 'logs').glob('*.attempt_*.json'))), 3)

    def test_daily_quota_pauses_without_retry(self):
        with patch('urllib.request.urlopen', side_effect=http_error(429, 'daily quota exceeded')) as urlopen:
            with self.assertRaises(TemporaryAPIError): self.matcher(max_attempts=3)('pair', SOURCE, TARGET)
            self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(self.delays, [])

    def test_invalid_model_output_is_not_retried(self):
        matcher = self.matcher(max_attempts=3)
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response({})).encode())) as urlopen:
            with self.assertRaises(ValueError): matcher('pair', SOURCE, TARGET)
            with self.assertRaises(ValueError): matcher('pair', SOURCE, TARGET)
            self.assertEqual(urlopen.call_count, 1)

    def test_authentication_failure_is_not_model_error(self):
        with patch('urllib.request.urlopen', side_effect=http_error(403)) as urlopen:
            with self.assertRaises(APIConfigurationError): self.matcher(max_attempts=3)('pair', SOURCE, TARGET)
            self.assertEqual(urlopen.call_count, 1)

    def test_groups_are_separate(self):
        from run_gemini_batch import group_pair_ids
        splits = {'train': ['train__v0', 'train__challenge_v0'],
                  'dev': ['dev__v0', 'dev__challenge_v0'], 'test': []}
        self.assertEqual(group_pair_ids(splits, 'dev', 'standard'), ['dev__v0'])
        self.assertEqual(group_pair_ids(splits, 'dev', 'challenge'), ['dev__challenge_v0'])
        splits['dev'].append('explicit_new_variant')
        splits['metadata'] = {'dataset_groups': {'challenge': {'pair_ids': ['explicit_new_variant']}}}
        self.assertEqual(group_pair_ids(splits, 'dev', 'challenge'), ['dev__challenge_v0', 'explicit_new_variant'])

    def test_frozen_examples_reject_changed_contents_and_nontrain(self):
        from run_gemini_batch import frozen_examples
        directory = self.root / 'pairs' / 'train__v0'
        directory.mkdir(parents=True)
        for name, value in [('schema_A.json', SOURCE), ('schema_B.json', TARGET), ('gold_mapping.json', MAPPING)]:
            write_json(directory / name, value)
        examples = [{'pair_id': 'train__v0', 'source_schema': SOURCE, 'target_schema': TARGET, 'mapping': MAPPING}]
        protocol = {'shots': 1, 'example_pair_ids': ['train__v0'],
                    'examples_sha256': hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()}
        splits = {'train': ['train__v0']}
        self.assertEqual(frozen_examples(self.root, splits, protocol), examples)
        with self.assertRaisesRegex(ValueError, 'not in train'):
            frozen_examples(self.root, {'train': []}, protocol)
        write_json(directory / 'schema_A.json', {**SOURCE, 'db_id': 'changed'})
        with self.assertRaisesRegex(ValueError, 'contents have changed'):
            frozen_examples(self.root, splits, protocol)

    def test_batch_checkpoints_and_complete_denominator(self):
        import run_gemini_batch
        root = self.root / 'data'
        ids = ['dev1__v0', 'dev2__v0']
        train = ['train1__v0', 'train2__v0', 'train3__v0']
        for pair in train + ids:
            directory = root / 'pairs' / pair
            directory.mkdir(parents=True)
            write_json(directory / 'schema_A.json', SOURCE)
            write_json(directory / 'schema_B.json', TARGET)
            write_json(directory / 'gold_mapping.json', MAPPING)
            row = {'qid': 'q', 'q_A': 'SELECT x FROM a', 'gold_q_B': 'SELECT y FROM b', 'level': 'L2', 'is_join': False}
            (directory / 'queries.jsonl').write_text(json.dumps(row) + '\n')
            for db_name, sql in [('db_A.sqlite', 'CREATE TABLE a(x INTEGER); INSERT INTO a VALUES(7);'),
                                 ('db_B.sqlite', 'CREATE TABLE b(y INTEGER); INSERT INTO b VALUES(7);')]:
                with sqlite3.connect(directory / db_name) as db: db.executescript(sql)
        write_json(root / 'splits.json', {'train': train, 'dev': ids, 'test': []})
        out = self.root / 'batch'
        examples = [{'pair_id': pair, 'source_schema': SOURCE, 'target_schema': TARGET, 'mapping': MAPPING} for pair in train]
        protocol = self.root / 'protocol.json'
        write_json(protocol, {'model': 'test-model', 'shots': 3, 'example_pair_ids': train,
            'examples_sha256': hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()})
        argv = ['run_gemini_batch.py', '--data-root', str(root), '--outdir', str(out), '--model', 'test-model', '--interval', '0', '--examples-config', str(protocol)]
        with patch('sys.argv', argv), patch.object(GeminiMatcher, '__call__', side_effect=[MAPPING, TemporaryAPIError('503')]):
            self.assertEqual(run_gemini_batch.main(), 2)
        self.assertEqual(read_json(out / 'progress.json')['completed_pairs'], 1)
        self.assertFalse((out / 'summary.json').exists())
        with patch('sys.argv', argv), patch.object(GeminiMatcher, '__call__', return_value=MAPPING) as predict:
            self.assertEqual(run_gemini_batch.main(), 0)
            self.assertEqual(predict.call_count, 1)
        self.assertEqual(read_json(out / 'summary.json')['n_queries'], 2)
        self.assertEqual(read_json(out / 'summary.json')['execution_accuracy'], 1)
        self.assertEqual(read_json(out / 'progress.json')['completed_pairs'], 2)


if __name__ == '__main__':
    unittest.main()
