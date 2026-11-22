import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqltranslate.evaluation import (compare_rows, evaluate_query, execute_readonly,
    load_splits, mapping_counts, run, validate_mapping, write_json)
from sqltranslate.fewshot import FewShotMatcher, build_messages, select_examples


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = {'tables': [{'name': 'people', 'columns': [{'name': 'id', 'type': 'INTEGER'}, {'name': 'name', 'type': 'TEXT'}]}], 'primary_keys': [], 'foreign_keys': []}
        self.target = {'tables': [{'name': 'users', 'columns': [{'name': 'label', 'type': 'TEXT'}, {'name': 'uid', 'type': 'INTEGER'}]}], 'primary_keys': [], 'foreign_keys': []}
        self.gold = {'tables': {'people': 'users'}, 'columns': {'people.id': 'users.uid', 'people.name': 'users.label'}}
        for pair in ('train_db__syn_v0', 'test_db__syn_v0'):
            directory = self.root / 'pairs' / pair
            directory.mkdir(parents=True)
            for filename, obj in [('schema_A.json', self.source), ('schema_B.json', self.target), ('gold_mapping.json', self.gold)]:
                write_json(directory / filename, obj)
            with sqlite3.connect(directory / 'db_A.sqlite') as db:
                db.executescript("CREATE TABLE people(id INTEGER, name TEXT); INSERT INTO people VALUES(1,'A'),(2,'B'),(3,'B');")
            with sqlite3.connect(directory / 'db_B.sqlite') as db:
                db.executescript("CREATE TABLE users(uid INTEGER, label TEXT); INSERT INTO users VALUES(1,'A'),(2,'B'),(3,'B');")
            self.rows = [
                {'qid': 'q1', 'q_A': 'SELECT name FROM people', 'gold_q_B': 'SELECT label FROM users', 'is_join': False, 'level': 'L2'},
                {'qid': 'q2', 'q_A': 'SELECT id FROM people ORDER BY id DESC', 'gold_q_B': 'SELECT uid FROM users ORDER BY uid DESC', 'is_join': False, 'level': 'L3'},
                {'qid': 'q3', 'q_A': 'SELECT name FROM people WHERE id > 100', 'gold_q_B': 'SELECT label FROM users WHERE uid > 100', 'is_join': False, 'level': 'L2'}]
            (directory / 'queries.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        self.splits = {'train': ['train_db__syn_v0'], 'dev': [], 'test': ['test_db__syn_v0']}
        write_json(self.root / 'splits.json', self.splits)
        self.directory = self.root / 'pairs' / 'test_db__syn_v0'

    def test_duplicate_sensitive_and_ordered_results(self):
        self.assertFalse(compare_rows([(1,), (1,)], [(1,)], False))
        self.assertTrue(compare_rows([(1,), (2,)], [(2,), (1,)], False))
        self.assertFalse(compare_rows([(1,), (2,)], [(2,), (1,)], True))

    def test_readonly_and_timeout(self):
        db = self.directory / 'db_A.sqlite'
        with self.assertRaises(sqlite3.OperationalError):
            execute_readonly(db, 'DELETE FROM people')
        self.assertEqual(len(execute_readonly(db, 'SELECT * FROM people')), 3)
        with self.assertRaises(sqlite3.OperationalError):
            execute_readonly(db, 'WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x', timeout=.01)
        with self.assertRaises(ValueError):
            execute_readonly(db, 'SELECT * FROM people', max_rows=1)

    def test_validation_and_changed_metrics(self):
        validate_mapping(self.gold, self.source, self.target)
        invalid = copy.deepcopy(self.gold)
        invalid['columns']['people.id'] = 'users.label'
        with self.assertRaises(ValueError):
            validate_mapping(invalid, self.source, self.target)
        gold = {'tables': {'a': 'a'}, 'columns': {'a.id': 'a.id', 'a.name': 'a.label'}}
        pred = {'tables': {'a': 'a'}, 'columns': {'a.id': 'a.id', 'a.name': 'a.id'}}
        counts = mapping_counts(pred, gold)
        self.assertEqual(counts['column_correct'], 1)
        self.assertEqual(counts['changed_column_correct'], 0)
        self.assertEqual(counts['changed_column_total'], 1)

    def test_base_database_leakage_rejected(self):
        self.splits['train'] = ['test_db__abbr_v1']
        write_json(self.root / 'splits.json', self.splits)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            load_splits(self.root)

    def test_training_examples_and_prompt(self):
        examples = select_examples(self.root, 1, 4705)
        self.assertEqual(examples[0]['pair_id'], 'train_db__syn_v0')
        messages = build_messages(self.source, self.target, examples)
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
        self.assertNotIn('mapping', json.loads(messages[-1]['content']))
        with self.assertRaises(ValueError):
            select_examples(self.root, 2, 4705)

    def test_llm_cache_and_invalid_output(self):
        class Response:
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps(self.payload).encode()
        matcher = FewShotMatcher(self.root, 'test-model', [], self.root / 'logs')
        payload = {'model': 'test-model-v1', 'usage': {'total_tokens': 42}, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.gold)}}]}
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-key'}), patch('urllib.request.urlopen', return_value=Response(payload)) as request:
            self.assertEqual(matcher('p', self.source, self.target), self.gold)
            self.assertEqual(matcher('p', self.source, self.target), self.gold)
            self.assertEqual(request.call_count, 1)
        log = json.loads(next((self.root / 'logs').glob('*.json')).read_text())
        self.assertEqual(log['usage']['total_tokens'], 42)
        self.assertNotIn('fake-key', json.dumps(log))
        bad = FewShotMatcher(self.root, 'test-model', [], self.root / 'badlogs')
        payload['choices'][0]['message']['content'] = '{}'
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-key'}), patch('urllib.request.urlopen', return_value=Response(payload)):
            with self.assertRaises(ValueError): bad('p', self.source, self.target)
        self.assertEqual(json.loads(next((self.root / 'badlogs').glob('*.json')).read_text())['status'], 'failure')

    def test_dry_run_is_repeatable_without_api_calls(self):
        matcher = FewShotMatcher(self.root, 'placeholder', [], self.root / 'preview', dry_run=True)
        with patch('urllib.request.urlopen') as request:
            first = matcher('p', self.source, self.target)
            self.assertEqual(first, matcher('p', self.source, self.target))
            request.assert_not_called()
        self.assertEqual(json.loads(first.read_text())['status'], 'dry_run')

    def test_oracle_saved_mapping_and_failures(self):
        methods = [{'label': 'oracle', 'oracle': True},
                   {'label': 'correct', 'predictor': lambda p, a, b: self.gold},
                   {'label': 'invalid', 'predictor': lambda p, a, b: {}}]
        report = run(self.root, 'test', methods, self.root / 'results')
        self.assertEqual(report['oracle']['execution_accuracy'], 1)
        self.assertEqual(report['correct']['execution_accuracy'], 1)
        self.assertEqual(report['invalid']['execution_accuracy'], 0)
        self.assertEqual(report['invalid']['n_queries'], 3)
        self.assertEqual(report['invalid']['error_counts']['mapping'], 3)
        self.assertEqual(report['oracle']['empty_gold_queries'], 1)
        self.assertTrue((self.root / 'results' / 'queries.csv').exists())
        with self.assertRaises(FileExistsError):
            run(self.root, 'test', methods, self.root / 'results')

    def test_reference_mismatch_is_not_scored_correct(self):
        row = dict(self.rows[0], gold_q_B='SELECT uid FROM users')
        report = evaluate_query(row, self.directory / 'db_A.sqlite', self.directory / 'db_B.sqlite', self.gold)
        self.assertFalse(report['reference_valid'])
        self.assertFalse(report['execution_correct'])
        self.assertEqual(report['error_stage'], 'reference')

    def test_wrong_but_executable_mapping(self):
        wrong = copy.deepcopy(self.gold)
        wrong['columns'] = {'people.id': 'users.label', 'people.name': 'users.uid'}
        report = evaluate_query(self.rows[0], self.directory / 'db_A.sqlite', self.directory / 'db_B.sqlite', wrong)
        self.assertTrue(report['execution_success'])
        self.assertFalse(report['execution_correct'])
        self.assertEqual(report['error_stage'], 'result_mismatch')


if __name__ == '__main__':
    unittest.main()
