import json, sqlite3, tempfile, unittest
from pathlib import Path
from sqltranslate.core import extract_schema, predict, fk_consistency, type_score
from sqltranslate.data import generate, make_pairs
from sqltranslate.sql import rewrite_sql, evaluate

class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.d=Path(self.tmp.name); self.a=self.d/'a.db'; self.b=self.d/'b.db'
        with sqlite3.connect(self.a) as c:
            c.executescript('''CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, customer_name TEXT);
                CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), amount REAL);
                INSERT INTO customers VALUES(1,'Alice'),(2,'Bob'); INSERT INTO orders VALUES(1,1,100),(2,2,20);''')
        self.gold=generate(self.a,self.b,self.d/'gold.json'); self.src=extract_schema(self.a); self.tgt=extract_schema(self.b)
    def tearDown(self): self.tmp.cleanup()
    def test_gold_queries(self):
        for sql in ['SELECT c.customer_name, o.amount FROM customers c JOIN orders o ON c.customer_id=o.customer_id WHERE o.amount > 50',
                    'SELECT customer_name FROM customers', 'SELECT * FROM customers',
                    'SELECT COUNT(*) FROM orders', 'SELECT customer_name FROM customers ORDER BY customer_name',
                    'WITH x AS (SELECT customer_name FROM customers) SELECT customer_name FROM x',
                    'SELECT customer_name FROM customers WHERE customer_id IN (SELECT customer_id FROM orders WHERE amount > 50)']:
            with self.subTest(sql=sql):
                translated=rewrite_sql(sql,self.src,self.gold)
                self.assertTrue(evaluate(self.a,self.b,sql,translated)['equivalent'],translated)
    def test_fk_and_matching(self):
        self.assertEqual(fk_consistency(self.gold['columns'],self.src,self.tgt),1)
        wrong=self.gold['columns'].copy(); wrong['orders.customer_id']=wrong['orders.order_id']
        self.assertEqual(fk_consistency(wrong,self.src,self.tgt),0)
        m=predict(self.src,self.tgt); self.assertEqual(set(m['columns']),set(self.gold['columns']))
        self.assertEqual(type_score('INTEGER','TEXT'),0)
        self.assertEqual(len(make_pairs(self.src,self.tgt,self.gold)),7)
    def test_unsafe_and_unsupported(self):
        with self.assertRaises(ValueError): rewrite_sql('DELETE FROM orders',self.src,self.gold)
        with self.assertRaises(Exception): rewrite_sql('SELECT customer_id FROM customers JOIN orders',self.src,self.gold)
        with self.assertRaises(ValueError): generate(self.a,self.a,self.d/'x.json')
    def test_no_fk(self):
        self.src.fks=[]; self.assertIsNone(fk_consistency(self.gold['columns'],self.src,self.tgt))

if __name__=='__main__': unittest.main()
