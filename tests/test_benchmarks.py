import json, sqlite3, tempfile, unittest
from pathlib import Path
from sqltranslate.core import extract_schema
from sqltranslate.benchmarks import derive_dr_mapping, prepare_dr, prepare_spider, retrieval_metrics, evaluate_paired

class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.d=Path(self.tmp.name); self.a=self.d/'shop.sqlite'; self.b=self.d/'renamed.sqlite'
        with sqlite3.connect(self.a) as c:
            c.executescript("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id)); INSERT INTO users VALUES (1,'Alice'); INSERT INTO orders VALUES (5,1);")
        with sqlite3.connect(self.b) as c:
            c.executescript("CREATE TABLE clients(id INTEGER PRIMARY KEY, full_name TEXT); CREATE TABLE purchases(id INTEGER PRIMARY KEY, client_id INTEGER REFERENCES clients(id)); INSERT INTO clients VALUES (1,'Alice'); INSERT INTO purchases VALUES (5,1);")
        self.pre={'db_id':'shop','table_names_original':['users','orders'],'column_names_original':[[-1,'*'],[0,'id'],[0,'name'],[1,'id'],[1,'user_id']],'column_types':['text','number','text','number','number'],'primary_keys':[1,3],'foreign_keys':[[4,1]]}
        self.post={**self.pre,'table_names_original':['clients','purchases'],'column_names_original':[[-1,'*'],[0,'id'],[0,'full_name'],[1,'id'],[1,'client_id']]}
    def tearDown(self): self.tmp.cleanup()
    def test_candidate_audit_and_evaluation(self):
        m=derive_dr_mapping(self.pre,self.post,self.a,self.b)
        self.assertTrue(m['audit']['table_contents_equal']); self.assertEqual(m['audit']['sqlite_fk_consistency'],1)
        self.assertIn('requires human review',m['provenance'])
        metric=retrieval_metrics(extract_schema(self.a),extract_schema(self.b),m,k=3)
        self.assertEqual(metric['n_columns'],4); self.assertGreaterEqual(metric['topk_recall'],metric['top1_accuracy'])
        rows=[{'id':0,'source_sql':'SELECT name FROM users','target_gold_sql':'SELECT full_name FROM clients'}]
        report=evaluate_paired(self.a,self.b,m,rows)
        self.assertEqual(report['target_gold_execution_equivalence'],1)
        self.assertEqual(report['cross_database_execution_equivalence'],1)
        with sqlite3.connect(self.b) as c: c.execute("UPDATE clients SET full_name='Bob'")
        m=derive_dr_mapping(self.pre,self.post,self.a,self.b)
        self.assertFalse(m['audit']['table_contents_equal'])
        report=evaluate_paired(self.a,self.b,m,rows)
        self.assertEqual(report['target_gold_execution_equivalence'],1)
        self.assertEqual(report['cross_database_execution_equivalence'],0)
    def test_refuse_metadata_mismatch(self):
        wrong={**self.post,'foreign_keys':[]}
        with self.assertRaises(ValueError): derive_dr_mapping(self.pre,wrong,self.a,self.b)
    def test_prepare(self):
        def save(name,data):
            p=self.d/name; p.write_text(json.dumps(data)); return p
        pre=save('pre.json',[self.pre]); post=save('post.json',[self.post]);
        queries=save('train.json',[{'db_id':'shop','query':'SELECT name FROM users'}])
        manifest=prepare_spider(self.a,queries,self.d/'spider'); self.assertEqual(manifest['db_id'],'shop')
        qa=save('preq.json',[{'db_id':'shop','q_id_spider_dev':2,'query':'SELECT name FROM users'}])
        qb=save('postq.json',[{'db_id':'shop','q_id_spider_dev':2,'query':'SELECT full_name FROM clients'}])
        dr=prepare_dr(pre,post,self.a,self.b,'shop',self.d/'dr',qa,qb)
        self.assertEqual(dr['query_count'],1); self.assertTrue(dr['requires_manual_review'])

if __name__=='__main__': unittest.main()
