import json, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from train_sbert import prepare_pairs, train
from matcher import build_report
from sqltranslate.core import Schema, Table, Column

class NeuralWorkflowTests(unittest.TestCase):
    def test_pairs_are_validated_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'pairs.json'
            p.write_text(json.dumps([{'anchor':'a','positive':'x'},{'anchor':'a','positive':'x'},{'anchor':'b','positive':'y'}]))
            self.assertEqual(len(prepare_pairs(p)),2)
            p.write_text(json.dumps([{'anchor':'a'}]))
            with self.assertRaises(ValueError): prepare_pairs(p)

    def test_candidates_and_similarity_matrix(self):
        a=Schema([Table('customers',[Column('name','TEXT')])],[])
        b=Schema([Table('clients',[Column('full_name','TEXT'),Column('id','INTEGER')])],[])
        report=build_report(a,b,np.array([[0.9]]),np.array([[0.8,0.1]]),2)
        self.assertEqual(report['columns']['top_k']['customers.name'][0]['target'],'clients.full_name')
        self.assertEqual(report['columns']['similarity_matrix'],[[0.8,0.1]])
        with self.assertRaises(ValueError): build_report(a,b,np.zeros((2,2)),np.zeros((1,2)))

    def test_training_pipeline_with_stubs(self):
        # Tests wiring without claiming that a real model has been trained.
        class Model:
            def __init__(self,name): self.name=name
            def save_pretrained(self,path): Path(path).mkdir(parents=True,exist_ok=True)
        class Trainer:
            def __init__(self,**kw): self.kw=kw
            def train(self): pass
        st=types.ModuleType('sentence_transformers')
        st.SentenceTransformer=Model; st.SentenceTransformerTrainer=Trainer
        st.SentenceTransformerTrainingArguments=lambda **kw:kw
        st.losses=types.SimpleNamespace(MultipleNegativesRankingLoss=lambda model:object())
        argsmod=types.ModuleType('sentence_transformers.training_args')
        argsmod.BatchSamplers=types.SimpleNamespace(NO_DUPLICATES='no_duplicates')
        datasets=types.ModuleType('datasets'); datasets.Dataset=types.SimpleNamespace(from_list=lambda x:x)
        with tempfile.TemporaryDirectory() as directory:
            pairfile=Path(directory)/'pairs.json'; pairfile.write_text(json.dumps([{'anchor':'a','positive':'x'},{'anchor':'b','positive':'y'}]))
            args=types.SimpleNamespace(train_pairs=str(pairfile),output=str(Path(directory)/'model'),model='stub',epochs=1,batch_size=2,learning_rate=2e-5,seed=42)
            with patch.dict('sys.modules',{'sentence_transformers':st,'sentence_transformers.training_args':argsmod,'datasets':datasets}):
                result=train(args)
            metadata=json.loads((Path(args.output)/'training_metadata.json').read_text())
            self.assertEqual(result['training_pairs'],2)
            self.assertEqual(metadata['loss'],'MultipleNegativesRankingLoss')

if __name__=='__main__': unittest.main()
