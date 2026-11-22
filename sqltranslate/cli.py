import argparse, csv, json, os, random, sqlite3, sys, urllib.request
from pathlib import Path
from .core import extract_schema, predict, save_json, load_json, fk_consistency
from .data import generate, make_pairs
from .sql import rewrite_sql, evaluate
from .benchmarks import prepare_spider, prepare_dr, retrieval_metrics, evaluate_paired

def schema(path): return extract_schema(path)
def parser():
    p=argparse.ArgumentParser(prog='sqltranslate'); sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('schema'); s.add_argument('db'); s.add_argument('--out')
    s=sub.add_parser('generate'); s.add_argument('source'); s.add_argument('target'); s.add_argument('mapping'); s.add_argument('--difficulty',choices=['easy','medium','hard'],default='easy'); s.add_argument('--seed',type=int,default=42)
    s=sub.add_parser('match'); s.add_argument('source'); s.add_argument('target'); s.add_argument('--method',choices=['lexical','sbert'],default='lexical'); s.add_argument('--model',default='all-MiniLM-L6-v2'); s.add_argument('--type-weight',type=float,default=.15); s.add_argument('--fk-weight',type=float,default=.3); s.add_argument('--top-k',type=int,default=3); s.add_argument('--out',required=True)
    s=sub.add_parser('rewrite'); s.add_argument('source'); s.add_argument('mapping'); s.add_argument('--sql',required=True); s.add_argument('--out')
    s=sub.add_parser('evaluate'); s.add_argument('source'); s.add_argument('target'); s.add_argument('mapping'); s.add_argument('--queries',required=True); s.add_argument('--gold'); s.add_argument('--out',required=True)
    s=sub.add_parser('pairs'); s.add_argument('manifest'); s.add_argument('--out',required=True)
    s=sub.add_parser('train'); s.add_argument('pairs'); s.add_argument('--model',default='all-MiniLM-L6-v2'); s.add_argument('--out',required=True); s.add_argument('--epochs',type=int,default=1); s.add_argument('--batch-size',type=int,default=8)
    s=sub.add_parser('llm-match'); s.add_argument('source'); s.add_argument('target'); s.add_argument('--examples',required=True,help='JSON list of train schema-pair examples'); s.add_argument('--model',required=True); s.add_argument('--out',required=True); s.add_argument('--endpoint',default='https://api.openai.com/v1/chat/completions')
    s=sub.add_parser('split'); s.add_argument('ids',help='JSON list of database IDs'); s.add_argument('--out',required=True); s.add_argument('--seed',type=int,default=42)
    s=sub.add_parser('spider-queries'); s.add_argument('input',help='Spider train/dev JSON'); s.add_argument('--db-id',required=True); s.add_argument('--out',required=True)
    s=sub.add_parser('prepare-spider'); s.add_argument('db'); s.add_argument('questions'); s.add_argument('--outdir',required=True); s.add_argument('--difficulty',choices=['easy','medium','hard'],default='easy'); s.add_argument('--seed',type=int,default=42)
    s=sub.add_parser('prepare-dr'); s.add_argument('--pre-tables',required=True); s.add_argument('--post-tables',required=True); s.add_argument('--source',required=True); s.add_argument('--target',required=True); s.add_argument('--db-id',required=True); s.add_argument('--pre-questions'); s.add_argument('--post-questions'); s.add_argument('--outdir',required=True)
    s=sub.add_parser('retrieval-eval'); s.add_argument('source'); s.add_argument('target'); s.add_argument('gold'); s.add_argument('--method',choices=['lexical','sbert'],default='lexical'); s.add_argument('--model',default='all-MiniLM-L6-v2'); s.add_argument('--top-k',type=int,default=3); s.add_argument('--out',required=True)
    s=sub.add_parser('paired-eval'); s.add_argument('source'); s.add_argument('target'); s.add_argument('mapping'); s.add_argument('paired_queries'); s.add_argument('--out',required=True)
    s=sub.add_parser('retrieval-summary'); s.add_argument('reports',nargs='+'); s.add_argument('--out',required=True)
    s=sub.add_parser('check-splits'); s.add_argument('--train',required=True); s.add_argument('--dev',required=True); s.add_argument('--test',required=True)
    s=sub.add_parser('demo'); s.add_argument('--dir',default='demo_output')
    return p

def run_eval(args):
    src=schema(args.source); target=schema(args.target); mapping=load_json(args.mapping); queries=load_json(args.queries); results=[]
    for i,q in enumerate(queries):
        sql=q if isinstance(q,str) else q['sql'] if 'sql' in q else q['query']
        try:
            translated=rewrite_sql(sql,src,mapping); item=evaluate(args.source,args.target,sql,translated); item['target_sql']=translated
        except Exception as e: item={'execution_success':False,'equivalent':False,'error':f'{type(e).__name__}: {e}','target_sql':None}
        item.update({'id':i,'source_sql':sql}); results.append(item)
    n=len(results); report={'n_queries':n,'execution_success_rate':sum(r['execution_success'] for r in results)/n if n else None,'execution_equivalence':sum(r['equivalent'] for r in results)/n if n else None,'fk_consistency':fk_consistency(mapping['columns'],src,target),'results':results}
    if args.gold:
        gold=load_json(args.gold)
        for field in ('tables','columns'):
            truth=gold[field]; report[field+'_accuracy']=sum(mapping[field].get(k)==v for k,v in truth.items())/len(truth) if truth else None
    save_json(args.out,report); return {k:v for k,v in report.items() if k!='results'}

def validate_mapping(mapping,source,target):
    for key in ('tables','columns'):
        if not isinstance(mapping.get(key),dict): raise ValueError('LLM response must contain tables and columns objects')
    st={t.name for t in source.tables}; tt={t.name for t in target.tables}
    sc={f'{t.name}.{c.name}' for t in source.tables for c in t.columns}; tc={f'{t.name}.{c.name}' for t in target.tables for c in t.columns}
    if set(mapping['tables'])!=st or set(mapping['tables'].values())!=tt or set(mapping['columns'])!=sc or set(mapping['columns'].values())!=tc: raise ValueError('LLM mapping must be a complete one-to-one mapping')
    if any(mapping['tables'][a.rsplit('.',1)[0]]!=b.rsplit('.',1)[0] for a,b in mapping['columns'].items()): raise ValueError('Column mappings must agree with table mappings')
    return mapping

def main():
    a=parser().parse_args(); result=None
    if a.cmd=='schema':
        result=schema(a.db).to_dict()
        if a.out: save_json(a.out,result)
    elif a.cmd=='generate': result=generate(a.source,a.target,a.mapping,a.difficulty,a.seed)
    elif a.cmd=='match': result=predict(schema(a.source),schema(a.target),a.method,a.model,a.type_weight,a.fk_weight,a.top_k); save_json(a.out,result)
    elif a.cmd=='rewrite':
        result=rewrite_sql(a.sql,schema(a.source),load_json(a.mapping))
        if a.out: Path(a.out).write_text(result+'\n')
    elif a.cmd=='evaluate': result=run_eval(a)
    elif a.cmd=='pairs':
        pairs=[]
        for row in load_json(a.manifest): pairs.extend(make_pairs(schema(row['source']),schema(row['target']),load_json(row['mapping'])))
        save_json(a.out,pairs); result={'pairs':len(pairs)}
    elif a.cmd=='train':
        try:
            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
            from datasets import Dataset
        except ImportError as e: raise RuntimeError('Install requirements-neural.txt first') from e
        pairs=load_json(a.pairs)
        if len(pairs)<2: raise ValueError('Training needs at least two pairs')
        model=SentenceTransformer(a.model)
        # Pairwise cosine regression avoids treating equivalent schema names in a batch as negatives.
        rng=random.Random(42); rows=[]
        for i,pair in enumerate(pairs):
            rows.append({'sentence1':pair['anchor'],'sentence2':pair['positive'],'label':1.0})
            options=[p['positive'] for p in pairs if p['positive'] != pair['positive']]
            if options: rows.append({'sentence1':pair['anchor'],'sentence2':rng.choice(options),'label':0.0})
        # Negative labels may contain unknown semantic equivalents; curate before scientific use.
        model_args=SentenceTransformerTrainingArguments(output_dir=a.out,num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,report_to='none')
        SentenceTransformerTrainer(model=model,args=model_args,train_dataset=Dataset.from_list(rows),loss=losses.CosineSimilarityLoss(model)).train()
        model.save_pretrained(a.out); result={'model':a.out,'examples':len(rows),'note':'Review mined negatives for false negatives before scientific experiments.'}
    elif a.cmd=='split':
        ids=load_json(a.ids)
        if len(set(ids))!=len(ids): raise ValueError('Database IDs must be unique')
        random.Random(a.seed).shuffle(ids); n=len(ids); i=int(.7*n); j=int(.85*n)
        result={'train':ids[:i],'dev':ids[i:j],'test':ids[j:],'seed':a.seed}; save_json(a.out,result)
    elif a.cmd=='spider-queries':
        rows=load_json(a.input); result=[r['query'] for r in rows if r['db_id']==a.db_id]; save_json(a.out,result); result={'queries':len(result)}
    elif a.cmd=='llm-match':
        key=os.environ.get('OPENAI_API_KEY')
        if not key: raise ValueError('Set OPENAI_API_KEY for the configured API endpoint')
        source=schema(a.source); target=schema(a.target)
        examples=load_json(a.examples)
        prompt={'instructions':'Return only a JSON object with tables and columns dictionaries mapping every source identifier to the corresponding target identifier. Columns use table.column. Each mapping is one-to-one and columns must belong to the mapped table.','training_examples':examples,'source_schema':source.to_dict(),'target_schema':target.to_dict()}
        body=json.dumps({'model':a.model,'messages':[{'role':'user','content':json.dumps(prompt)}],'temperature':0,'response_format':{'type':'json_object'}}).encode()
        req=urllib.request.Request(a.endpoint,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response: payload=json.load(response)
        result=validate_mapping(json.loads(payload['choices'][0]['message']['content']),source,target); save_json(a.out,result)
    elif a.cmd=='prepare-spider': result=prepare_spider(a.db,a.questions,a.outdir,a.difficulty,a.seed)
    elif a.cmd=='prepare-dr': result=prepare_dr(a.pre_tables,a.post_tables,a.source,a.target,a.db_id,a.outdir,a.pre_questions,a.post_questions)
    elif a.cmd=='retrieval-eval':
        report=retrieval_metrics(schema(a.source),schema(a.target),load_json(a.gold),a.method,a.model,a.top_k); save_json(a.out,report); result={k:v for k,v in report.items() if k!='details'}
    elif a.cmd=='paired-eval':
        report=evaluate_paired(a.source,a.target,load_json(a.mapping),load_json(a.paired_queries)); save_json(a.out,report); result={k:v for k,v in report.items() if k!='results'}
    elif a.cmd=='retrieval-summary':
        rows=[]
        for path in a.reports:
            report=load_json(path)
            rows.append({'report':path,**{k:report[k] for k in ('method','model','k','n_columns','top1_accuracy','topk_recall')}})
        with open(a.out,'w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result={'out':a.out,'reports':len(rows)}
    elif a.cmd=='check-splits':
        splits={name:{r['db_id'] for r in load_json(getattr(a,name))} for name in ('train','dev','test')}
        overlaps={f'{x}_{y}':sorted(splits[x]&splits[y]) for x,y in (('train','dev'),('train','test'),('dev','test'))}
        if any(overlaps.values()): raise ValueError(f'Database leakage across splits: {overlaps}')
        result={'counts':{k:len(v) for k,v in splits.items()},'overlap':overlaps}
    elif a.cmd=='demo': result=demo(a.dir)
    print(result if isinstance(result,str) else json.dumps(result,indent=2,ensure_ascii=False))

def demo(directory):
    d=Path(directory); d.mkdir(parents=True,exist_ok=True); source=d/'source.db'; target=d/'target.db'; gold=d/'gold.json'
    if source.exists() or target.exists(): raise ValueError('Demo databases already exist; choose a new --dir')
    with sqlite3.connect(source) as c:
        c.executescript('''CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, customer_name TEXT);
        CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), amount REAL);
        INSERT INTO customers VALUES (1,'Alice'),(2,'Bob'); INSERT INTO orders VALUES (10,1,100.0),(11,2,25.0);''')
    generate(source,target,gold,'easy',42)
    queries=['SELECT c.customer_name, o.amount FROM customers c JOIN orders o ON c.customer_id=o.customer_id WHERE o.amount > 50','SELECT COUNT(*) FROM orders','SELECT customer_name FROM customers ORDER BY customer_name']
    save_json(d/'queries.json',queries)
    mapping=predict(schema(source),schema(target),'lexical',type_weight=.15,fk_weight=.3); save_json(d/'predicted.json',mapping)
    class A: pass
    args=A(); args.source=str(source); args.target=str(target); args.queries=str(d/'queries.json'); args.mapping=str(gold); args.gold=str(gold); args.out=str(d/'gold_report.json')
    gold_report=run_eval(args); args.mapping=str(d/'predicted.json'); args.out=str(d/'predicted_report.json'); pred_report=run_eval(args)
    return {'directory':str(d),'gold':gold_report,'predicted':pred_report}

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'Error: {type(e).__name__}: {e}',file=sys.stderr); sys.exit(1)
