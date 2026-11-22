import json, subprocess, sys, tempfile, unittest
from pathlib import Path
from evaluate import run
from handoff.interface import load_splits, write_json

ROOT=Path(__file__).resolve().parents[1]

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.base=Path(self.tmp.name); self.data=self.base/'data'; self.pred=self.base/'pred'
        subprocess.run([sys.executable,str(ROOT/'make_toy.py'),'--data',str(self.data),'--pred',str(self.pred)],check=True,capture_output=True)
    def tearDown(self): self.tmp.cleanup()
    def test_correct_and_missing_prediction(self):
        rows,errors,queries=run(self.data,self.pred,'test','mock')
        self.assertEqual(rows[0]['column_accuracy'],1); self.assertEqual(rows[0]['execution_equivalence'],1); self.assertEqual(errors,[])
        rows,errors,_=run(self.data,self.base/'missing','test','missing')
        self.assertEqual(rows[0]['execution_equivalence'],0); self.assertEqual(rows[0]['column_top3_recall'],0)
        self.assertEqual(rows[0]['missing_sql'],2)
    def test_target_comparison_and_gold_error(self):
        file=self.pred/'toy_concert__v1.json'; obj=json.loads(file.read_text()); obj['q_B']['toy_001']='SELECT no_such_column FROM performer'; write_json(file,obj)
        rows,errors,queries=run(self.data,self.pred,'test','mock')
        self.assertEqual(rows[0]['prediction_sql_errors'],1); self.assertEqual(rows[0]['execution_equivalence'],.5)
        self.assertTrue(any(e.get('status')=='prediction_error' for e in errors))
    def test_split_leak(self):
        file=self.data/'splits.json'; write_json(file,{'train':['toy_concert__v1'],'dev':[],'test':['toy_concert__v1']})
        with self.assertRaises(ValueError): load_splits(self.data)

if __name__=='__main__': unittest.main()
