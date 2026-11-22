"""Generate a small, explicitly artificial end-to-end handoff fixture."""
import argparse, json, random, sqlite3
from pathlib import Path
from handoff.interface import write_json
from sqltranslate.core import extract_schema


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',default='data'); p.add_argument('--pred',default='predictions/mock'); a=p.parse_args()
    root=Path(a.data); pair='toy_concert__v1'; d=root/'pairs'/pair
    if d.exists() or (root/'splits.json').exists(): raise SystemExit('Destination exists; choose a new --data directory')
    d.mkdir(parents=True)
    with sqlite3.connect(d/'db_A.sqlite') as c:
        c.executescript("""CREATE TABLE singer(id INTEGER PRIMARY KEY,name TEXT,age INTEGER); CREATE TABLE song(id INTEGER PRIMARY KEY,singer_id INTEGER REFERENCES singer(id),title TEXT); INSERT INTO singer VALUES (1,'Alice',35),(2,'Bob',25),(3,'Carol',42); INSERT INTO song VALUES (11,1,'Sky'),(12,3,'Sea');""")
    with sqlite3.connect(d/'db_B.sqlite') as c:
        c.executescript("""CREATE TABLE performer(performer_id INTEGER PRIMARY KEY,full_name TEXT,age_years INTEGER); CREATE TABLE track(track_id INTEGER PRIMARY KEY,performer_ref INTEGER REFERENCES performer(performer_id),track_title TEXT); INSERT INTO performer VALUES (1,'Alice',35),(2,'Bob',25),(3,'Carol',42); INSERT INTO track VALUES (11,1,'Sky'),(12,3,'Sea');""")
    for suffix in ('A','B'):
        obj=extract_schema(d/f'db_{suffix}.sqlite').to_dict()
        if suffix=='B':
            rng=random.Random(42); rng.shuffle(obj['tables'])
            for t in obj['tables']: rng.shuffle(t['columns'])
        write_json(d/f'schema_{suffix}.json',obj)
    gold={'tables':{'singer':'performer','song':'track'},'columns':{'singer.id':'performer.performer_id','singer.name':'performer.full_name','singer.age':'performer.age_years','song.id':'track.track_id','song.singer_id':'track.performer_ref','song.title':'track.track_title'}}
    write_json(d/'gold_mapping.json',gold); write_json(d/'metadata.json',{'db_id':'toy_concert','pair_id':pair,'provenance':'handwritten toy fixture; not Spider','data_correspondence':'equivalent toy data'})
    queries=[{'qid':'toy_001','q_A':'SELECT name FROM singer WHERE age > 30','gold_q_B':'SELECT full_name FROM performer WHERE age_years > 30','is_join':False,'level':'L1'},{'qid':'toy_002','q_A':'SELECT s.name, t.title FROM singer s JOIN song t ON s.id = t.singer_id','gold_q_B':'SELECT p.full_name, t.track_title FROM performer p JOIN track t ON p.performer_id = t.performer_ref','is_join':True,'level':'L2'}]
    (d/'queries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in queries))
    write_json(root/'splits.json',{'train':[],'dev':[],'test':[pair]})
    tables={k:[[v,.9]] for k,v in gold['tables'].items()}; cols={k:[[v,.9]] for k,v in gold['columns'].items()}
    prediction={'tables':tables,'columns':cols,'selected_mapping':gold,'q_B':{q['qid']:q['gold_q_B'] for q in queries}}
    write_json(Path(a.pred)/(pair+'.json'),prediction)
    print(json.dumps({'data':str(root),'pair_id':pair,'mock_prediction':str(Path(a.pred)/(pair+'.json'))},indent=2))
if __name__=='__main__': main()
