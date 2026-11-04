import json,math,statistics,sys
from pathlib import Path
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def load(n):return json.loads((p/(n+'.json')).read_bytes())
ref=load('fixed-inputs')
def series(n):
 d=load(n); assert d['status']=='completed' and len(d['cases'])==len(ref)==8
 vals=[]; gens=[]; outs=[]; genlps=[]
 for c,r in zip(d['cases'],ref):
  z=c['score_response']['choices'][0]; seq=r['prompt_ids']+r['target_ids']
  assert c['score_request']['prompt']==z['prompt_token_ids']==seq
  scores=[z['prompt_logprobs'][k][str(seq[k])]['logprob'] for k in range(len(seq)-8,len(seq))]
  assert scores==c['fixed_logprobs'] and len(scores)==8 and all(math.isfinite(v) for v in scores); vals+=scores
  ch=c['generation']['choices'][0];gens.append(ch['token_ids'])
  outs.append(ch['text'] if 'text' in ch else ch['message']['content'])
  lp=ch['logprobs'];genlps.extend(lp['token_logprobs'] if 'token_logprobs' in lp else [t['logprob'] for t in lp['content']])
 return vals,gens,outs,genlps
def compare(a,b):
 x,ga,ta,la=series(a); y,gb,tb,lb=series(b)
 delta=[abs(v-w) for v,w in zip(x,y)]; idx=max(range(len(delta)),key=delta.__getitem__); i,j=divmod(idx,8)
 return dict(versus=[a,b],n=len(delta),mean=statistics.mean(delta),median=statistics.median(delta),max=delta[idx],nonzero=sum(v!=0 for v in delta),worst=dict(prompt=ref[i]['prompt'],endpoint=ref[i]['endpoint'],position=j+1,id=ref[i]['target_ids'][j],values=[x[idx],y[idx]]),free_generation_identical=sum(g==h and t==u for g,h,t,u in zip(ga,gb,ta,tb)),free_lp_max_same_trajectory=max((abs(la[8*i+j]-lb[8*i+j]) for i in range(8) for j in range(8) if ga[i][:j+1]==gb[i][:j+1]),default=None))
pairs=[('off-a','off-b'),('off-a','off-cold'),('off-a','on'),('off-cold','on')]
print(json.dumps([compare(*pair) for pair in pairs],ensure_ascii=False,indent=1))

