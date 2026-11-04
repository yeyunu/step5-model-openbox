import json,sys,time,math,hashlib,urllib.request
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent
model="step5-head-tail8"
mode=sys.argv[1]
url="http://127.0.0.1:8940"
orig=json.loads((out.parent/"sp-pp-20261104-052954"/"baseline.json").read_text())
def call(ep,b):
 req=urllib.request.Request(url+ep,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=120) as r: return json.load(r)
def write(name,d):
 (out/name).write_text(json.dumps(d,ensure_ascii=False,indent=1))
def save():write(mode+".json",d)
d={"mode":mode,"started":datetime.now(timezone.utc).isoformat(),"cases":[]}
if mode=="on":
 w=call("/collective_rpc",{"method":"functional_topology"})["results"][0]["all_workers"]
 assert len({v["world_rank"] for v in w})==8 and all(v["compilation_enable_sp"] and v["tp"]==4 and v["dp"]==2 and {l["layer_idx"] for l in v["model_layers"] if l["use_moe"] and l["sequence_parallel"]}=={3,4,5,6} for v in w)
 d["workers"]=w
ref=out/"fixed-inputs.json"
cases=[] if mode=="off-a" else json.loads(ref.read_text())
for i,c in enumerate(orig["cases"]):
 b=dict(c["request"],return_token_ids=True)
 raw=call(c["endpoint"],b)
 ch=raw["choices"][0]
 ids=ch["token_ids"]
 prompt=raw.get("prompt_token_ids") or ch.get("prompt_token_ids")
 assert len(ids)==8 and len(prompt)==raw["usage"]["prompt_tokens"]
 if mode=="off-a":cases.append({"endpoint":c["endpoint"],"prompt":c["prompt"],"prompt_ids":prompt,"target_ids":ids})
 f=cases[i];assert prompt==f["prompt_ids"] and f["endpoint"]==c["endpoint"]
 # Score the fixed trajectory using prompt logprobs, independent of sampled outputs.
 seq=prompt+f["target_ids"]
 req={"model":model,"prompt":seq,"max_tokens":1,"temperature":0,"seed":0,"prompt_logprobs":1,"return_token_ids":True,"add_special_tokens":False}
 scored=call("/v1/completions",req)
 z=scored["choices"][0];assert z["prompt_token_ids"]==seq
 scores=[z["prompt_logprobs"][k][str(seq[k])]["logprob"] for k in range(len(prompt),len(seq))]
 assert len(scores)==8 and all(math.isfinite(x) and x<=0 for x in scores)
 d["cases"].append({"generation_request":b,"generation":raw,"score_request":req,"score_response":scored,"fixed_logprobs":scores})
 save()
if mode=="off-a":write(ref.name,cases)
d["status"]="completed";d["ended"]=datetime.now(timezone.utc).isoformat();save()

