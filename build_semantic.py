#!/usr/bin/env python3
"""
Semantic topic graph. No LLM, no conversation grouping.

  1. Collect substantive user messages from all transcripts (flat bag, convo origin ignored).
  2. Embed each locally with all-MiniLM-L6-v2 (sentence-transformers).
  3. Cluster by cosine similarity (kNN + label propagation) -> each cluster is a TOPIC node,
     sized by how many messages fall in it; labeled by its top keywords (TF-IDF, no LLM).
  4. Edges = cosine similarity between cluster centroids (semantic connectivity).
  5. Static force layout + interactive circular-node HTML.
"""
import json, glob, os, re, math
from collections import defaultdict, Counter
import numpy as np

PROJECTS = os.path.expanduser("~/.claude/projects")
OUT_DIR = os.path.expanduser("~/chat-topic-graph")
MODEL_NAME = "all-MiniLM-L6-v2"           # winner: separates short same-domain msgs better than bge-large
INSTRUCT = ""                             # constant prefix just added uniformity; not worth it here
ROLE = "assistant"                        # which side to map: "assistant" (LLM output) or "user" (your prompts)
MAX_UNITS = 4000          # cap embedded messages for speed
KNN = 8                   # neighbors per message when clustering
SIM_CLUSTER = 0.50        # min cosine to be a clustering neighbor
TOPIC_KNN = 4             # semantic edges kept per topic
SIM_EDGE = 0.40           # min cosine between topic centroids for an extra edge
MIN_CLUSTER = 2           # drop clusters smaller than this

STOP = set("the and for with from into your you that this have has had not are was were will "
           "can could would should make made use using used like want need get got add added fix "
           "also just now new one two how what why when where which who out off but our its it's "
           "i'm i've don't can't let lets please thanks yeah ok okay sure good update create change "
           "run running file files code line lines work working try set show give take put keep find "
           "function functions method value data type return test tests app way thing things still "
           "jasonchen macbook air ttys downloads desktop".split())
WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9+#-]{2,}")

def text_of(c):
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(b.get("text","") for b in c if isinstance(b, dict) and b.get("type")=="text")
    return ""

def clean(t):
    """Strip noise (code blocks, paths, shell prompts, banners, urls) so topics reflect intent."""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)           # fenced code blocks
    t = re.sub(r"`[^`]+`", " ", t)                         # inline code
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"(?i)last login:[^\n]*", " ", t)            # terminal login banner
    t = re.sub(r"[\w.\-]+@[\w.\-]+\S*", " ", t)             # emails + user@host shell prompts
    t = re.sub(r"~?(?:/[\w.\-]+){2,}/?", " ", t)            # file paths /Users/jasonchen/...
    t = re.sub(r"\bttys?\d+\b|%\s", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---- 1. collect substantive user messages (ignore which conversation) ----
units = []
for path in glob.glob(os.path.join(PROJECTS, "**", "*.jsonl"), recursive=True):
    try:
        with open(path) as fh:
            for line in fh:
                try: o = json.loads(line)
                except: continue
                m = o.get("message") or {}
                if (m.get("role") or o.get("type")) != ROLE: continue
                t = text_of(m.get("content")).strip()
                if not t or t[0] in "<[" or "<system-reminder>" in t[:40] or "Caveat:" in t[:40]:
                    continue
                t = clean(t)
                if len(t) < 60 or len(t.split()) < 9: continue       # drops confirmations + bare pastes
                units.append(t[:500])
    except Exception:
        continue
# dedup exact repeats but keep a frequency count (repeated work => bigger topic)
freq = Counter(units)
uniq = list(freq.keys())
if len(uniq) > MAX_UNITS:
    uniq = [u for u, _ in freq.most_common(MAX_UNITS)]
print(f"substantive user messages: {len(units)}  unique: {len(uniq)}")

# ---- 2. embed locally (per-model cache: swapping models keeps both, A/B without re-embedding) ----
EMB_CACHE = os.path.join(OUT_DIR, f"emb_{ROLE}_" + re.sub(r"[^a-z0-9]+", "-", MODEL_NAME.lower()) + ".npz")
cached = None
if os.path.exists(EMB_CACHE):
    z = np.load(EMB_CACHE, allow_pickle=True)
    if list(z["texts"]) == uniq:
        cached = z["emb"].astype(np.float32); print(f"embeddings loaded from cache ({MODEL_NAME})")
if cached is None:
    from sentence_transformers import SentenceTransformer
    print(f"loading {MODEL_NAME} + embedding (instruction-steered)...")
    model = SentenceTransformer(MODEL_NAME)
    cached = np.asarray(model.encode([INSTRUCT + t for t in uniq], batch_size=32,
                                     normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    np.savez(EMB_CACHE, emb=cached, texts=np.array(uniq, dtype=object))
emb = cached
n = len(uniq)

# ---- adaptive thresholds: derived from THIS model's own similarity scale (scale-invariant) ----
_rng = np.random.default_rng(0)
_cs = (emb[_rng.integers(0, n, 40000)] * emb[_rng.integers(0, n, 40000)]).sum(1)
SIM_CLUSTER = float(np.percentile(_cs, 99.5))    # "unusually similar" bar for clustering
SIM_EDGE = float(np.percentile(_cs, 99.0))       # slightly looser bar for topic edges
print(f"adaptive thresholds for {MODEL_NAME}: SIM_CLUSTER={SIM_CLUSTER:.2f}  SIM_EDGE={SIM_EDGE:.2f}")

# ---- 3. cluster: kNN cosine graph + label propagation ----
S = emb @ emb.T
np.fill_diagonal(S, 0)
adj = defaultdict(dict)
for i in range(n):
    nbrs = np.argsort(-S[i])[:KNN]
    for j in nbrs:
        if S[i, j] < SIM_CLUSTER: continue
        adj[i][int(j)] = float(S[i, j]); adj[int(j)][i] = float(S[i, j])
rng = np.random.default_rng(7); comp = list(range(n)); order = list(range(n))
for _ in range(80):
    rng.shuffle(order); changed = 0
    for u in order:
        if not adj[u]: continue
        v = defaultdict(float)
        for nb, w in adj[u].items(): v[comp[nb]] += w
        best = max(v.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if best != comp[u]: comp[u] = best; changed += 1
    if changed == 0: break
clusters = defaultdict(list)
for i in range(n): clusters[comp[i]].append(i)
clusters = {c: idx for c, idx in clusters.items() if len(idx) >= MIN_CLUSTER}
print(f"clusters (topics): {len(clusters)}")

# ---- label each cluster by distinctive keywords (TF-IDF, no LLM) ----
def toks(s): return [w.lower() for w in WORD.findall(s) if w.lower() not in STOP and not w.isdigit()]
gdf = Counter()
ctoks = {}
for c, idx in clusters.items():
    cc = Counter()
    for i in idx:
        for w in set(toks(uniq[i])): cc[w] += 1
    ctoks[c] = cc
    for w in cc: gdf[w] += 1
NC = len(clusters)
labels = {}
for c, cc in ctoks.items():
    scored = sorted(cc.items(), key=lambda kv: -(kv[1] * math.log(1 + NC / (1 + gdf[kv[0]]))))
    labels[c] = " ".join(w for w, _ in scored[:4]) or "misc"

# ---- 4. topic nodes + semantic centroid edges ----
cids = list(clusters.keys())
cidx = {c: i for i, c in enumerate(cids)}
cent = np.zeros((len(cids), emb.shape[1]), dtype=np.float32)
size = np.zeros(len(cids), dtype=np.float32)
for c in cids:
    idx = clusters[c]
    cent[cidx[c]] = emb[idx].mean(0)                    # unweighted centroid
    size[cidx[c]] = len(idx)                            # each distinct message counts as 1
cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
CS = cent @ cent.T; np.fill_diagonal(CS, 0)
edges = []; seen = set()
for i in range(len(cids)):
    nbrs = np.argsort(-CS[i])[:TOPIC_KNN]
    for rank, j in enumerate(nbrs):
        j = int(j)
        # always keep each topic's single nearest neighbor (no orphans); extras must clear SIM_EDGE
        if rank > 0 and CS[i, j] < SIM_EDGE: break
        if CS[i, j] <= 0: continue
        key = tuple(sorted((i, j)))
        if key in seen: continue
        seen.add(key); edges.append({"s": key[0], "t": key[1], "w": round(float(CS[i, j]), 3)})
print(f"semantic edges: {len(edges)}")

# ---- bridge disconnected components into ONE graph (best available semantic link) ----
M = len(cids)
parent = list(range(M))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for e in edges: parent[find(e["s"])] = find(e["t"])
groups = defaultdict(list)
for i in range(M): groups[find(i)].append(i)
if len(groups) > 1:
    main = max(groups.values(), key=len)
    for members in groups.values():
        if members is main: continue
        sub = CS[np.ix_(members, main)]
        a, b = np.unravel_index(np.argmax(sub), sub.shape)
        i, j = members[a], main[b]
        edges.append({"s": min(i, j), "t": max(i, j), "w": round(float(CS[i, j]), 3)})
    print(f"bridged {len(groups)-1} islands into the main graph")

# ---- color: communities over the topic graph (label propagation again) ----
M = len(cids)
tadj = defaultdict(dict)
for e in edges: tadj[e["s"]][e["t"]] = e["w"]; tadj[e["t"]][e["s"]] = e["w"]
tcomp = list(range(M)); order = list(range(M))
for _ in range(60):
    rng.shuffle(order); ch = 0
    for u in order:
        if not tadj[u]: continue
        v = defaultdict(float)
        for nb, w in tadj[u].items(): v[tcomp[nb]] += w
        best = max(v.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if best != tcomp[u]: tcomp[u] = best; ch += 1
    if ch == 0: break
remap = {}; tcomp = [remap.setdefault(x, len(remap)) for x in tcomp]

# ---- layout (FR, physics OFF in browser) ----
W = np.zeros((M, M), dtype=np.float32)
for e in edges: W[e["s"], e["t"]] = e["w"]; W[e["t"], e["s"]] = e["w"]
side = 2600.0
pos = (rng.random((M, 2)).astype(np.float32) - 0.5) * side
k = 2.2 * side / math.sqrt(max(M, 1)); temp = side / 6
for _ in range(600):
    d = pos[:, None, :] - pos[None, :, :]; dist = np.sqrt((d**2).sum(-1)) + 1e-2
    u = d / dist[..., None]; rep = (k*k)/dist; np.fill_diagonal(rep, 0)
    disp = (u*rep[..., None]).sum(1) - (u*((dist*dist)/k*W)[..., None]).sum(1) - pos*0.02
    dl = np.sqrt((disp**2).sum(-1, keepdims=True)) + 1e-9
    pos += disp/dl*np.minimum(dl, temp); temp = max(temp*0.99, 1.0)
pos -= pos.mean(0); pos *= 950/(np.percentile(np.abs(pos), 92)+1e-9)

palette = ["#00eaff","#22d3ee","#38bdf8","#0ea5e9","#3b82f6","#60a5fa","#06b6d4","#7dd3fc",
           "#2dd4bf","#2563eb","#67e8f9","#1d4ed8","#0284c7","#5eead4","#93c5fd","#818cf8"]
ccount = Counter(tcomp); corder = [c for c, _ in ccount.most_common()]
ccolor = {c: palette[i % len(palette)] for i, c in enumerate(corder)}
nodes = []
for c in cids:
    i = cidx[c]
    nodes.append({"id": i, "label": labels[c], "count": int(size[i]), "deg": len(tadj[i]),
                  "color": ccolor[tcomp[i]], "x": round(float(pos[i,0]),1), "y": round(float(pos[i,1]),1)})
out = {"nodes": nodes, "edges": [{"source":e["s"],"target":e["t"],"w":e["w"]} for e in edges]}
json.dump(out, open(os.path.join(OUT_DIR, "graph_semantic.json"), "w"))

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Topic Map</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
 html,body{margin:0;height:100%;color:#bfe9ff;overflow:hidden;font:14px system-ui,sans-serif;
   background:#00030a}
 #net{position:absolute;inset:0;z-index:1}
 #panel{position:absolute;top:12px;left:12px;width:230px;z-index:3}
 input{width:100%;box-sizing:border-box;background:#00030a;border:1px solid #0e7490;color:#bfe9ff;
   border-radius:6px;padding:8px 10px;font-size:13px;font-family:inherit}
 input:focus{outline:none;border-color:#22d3ee;box-shadow:0 0 10px rgba(34,211,238,.5)}
 #info{position:absolute;bottom:12px;left:12px;width:340px;z-index:3;background:rgba(2,8,23,.78);
   border:1px solid #0e7490;border-radius:10px;padding:12px 14px;display:none;backdrop-filter:blur(6px);
   box-shadow:0 0 22px rgba(14,165,233,.25)}
 #info b{color:#48e6ff;font-size:15px;text-shadow:0 0 8px rgba(56,189,248,.6)}
 .pill{display:inline-block;background:rgba(14,116,144,.3);border:1px solid #0e7490;border-radius:10px;
   padding:1px 8px;font-size:11px;color:#7dd3fc;margin:5px 4px 0 0}
 .key{color:#3f7aa0;font-size:11px;margin-top:8px;line-height:1.5}
</style></head><body>
<div id="net"></div>
<div id="panel"><input id="q" placeholder="search topics…" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false"></div>
<div id="info"></div>
<script>
const DATA=__DATA__; const byId={}; DATA.nodes.forEach(n=>byId[n.id]=n);
const cnts=DATA.nodes.map(n=>n.count), cmax=Math.max(...cnts), lcache={};
function radiusOf(n){ return 9 + Math.pow(n.count/cmax,0.6)*54; }   // size by #messages in topic
function lay(ctx,n){ if(lcache[n.id]) return lcache[n.id];
  let r=radiusOf(n);
  const wrap=fs=>{ctx.font=fs+'px system-ui,sans-serif';const out=[];let cur='';
    for(const w of n.label.split(' ')){const tt=cur?cur+' '+w:w;
      if(ctx.measureText(tt).width<=1.7*r||!cur)cur=tt;else{out.push(cur);cur=w;}}
    if(cur)out.push(cur);return out;};
  const fits=(fs,ls)=>{const lh=fs*1.12,h=ls.length*lh,wd=Math.max(...ls.map(l=>ctx.measureText(l).width));
    return h<=1.7*r&&wd<=1.85*r;};
  let fs=Math.max(6,Math.min(15,r*0.6)),lines=wrap(fs);
  while(fs>5&&!fits(fs,lines)){fs--;lines=wrap(fs);}
  if(!fits(fs,lines)){const wd=Math.max(...lines.map(l=>ctx.measureText(l).width));
    r=Math.max(r,(lines.length*fs*1.12)/1.6,wd/1.85);lines=wrap(fs);}
  const res={r,fs,lines,lh:fs*1.12};lcache[n.id]=res;return res;}
function txtCol(hex){const c=hex.replace('#','');
  const L=(0.299*parseInt(c.substr(0,2),16)+0.587*parseInt(c.substr(2,2),16)+0.114*parseInt(c.substr(4,2),16))/255;
  return L>0.62?'#0d1117':'#fff';}
const nodes=new vis.DataSet(DATA.nodes.map(n=>({id:n.id,x:n.x,y:n.y,raw:n,shape:'custom',
  ctxRenderer:({ctx,id,x,y,state:{selected,hover}})=>{const nn=byId[id],L=lay(ctx,nn);
    return {drawNode(){ctx.save();
      ctx.shadowColor=nn.color;ctx.shadowBlur=(selected||hover)?26:13;
      ctx.beginPath();ctx.arc(x,y,L.r,0,2*Math.PI);
      ctx.fillStyle=nn.color;ctx.globalAlpha=0.92;ctx.fill();
      ctx.shadowBlur=0;ctx.globalAlpha=1;
      ctx.lineWidth=(selected||hover)?2.5:1;ctx.strokeStyle=(selected||hover)?'#eafcff':'rgba(0,3,10,0.5)';ctx.stroke();
      ctx.fillStyle=txtCol(nn.color);ctx.font=L.fs+'px system-ui,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';
      const sy=y-(L.lines.length-1)*L.lh/2;L.lines.forEach((l,i)=>ctx.fillText(l,x,sy+i*L.lh));ctx.restore();},
      nodeDimensions:{width:2*L.r,height:2*L.r}};}})));
const edges=new vis.DataSet(DATA.edges.map(e=>({from:e.source,to:e.target,
  width:Math.max(0.4,0.4+(e.w-0.4)*5),color:{color:'#1fb6e6',opacity:0.28}})));
const net=new vis.Network(document.getElementById('net'),{nodes,edges},{
  edges:{smooth:false},
  physics:{solver:'barnesHut',barnesHut:{gravitationalConstant:-30000,centralGravity:0.05,
    springLength:140,springConstant:0.03,damping:0.5,avoidOverlap:0.35},
    stabilization:{iterations:300},minVelocity:0.6,timestep:0.4},
  interaction:{hover:true,dragNodes:true}});
net.once('stabilizationIterationsDone',()=>net.fit());
document.getElementById('q').oninput=e=>{const q=e.target.value.toLowerCase().trim();
  nodes.update(DATA.nodes.map(n=>({id:n.id,hidden:q?!n.label.toLowerCase().includes(q):false})));};
const info=document.getElementById('info');
net.on('click',p=>{if(!p.nodes.length){info.style.display='none';return;}
  const n=nodes.get(p.nodes[0]).raw;info.style.display='block';
  info.innerHTML=`<b>${n.label}</b><br><span class="pill">${n.count} responses</span><span class="pill">${n.deg} connections</span>`;});
</script></body></html>"""
HTML=(HTML.replace("__DATA__",json.dumps(out)).replace("__NODES__",str(len(nodes)))
        .replace("__EDGES__",str(len(edges))).replace("__MODEL__",MODEL_NAME)
        .replace("__SIMEDGE__",f"{SIM_EDGE:.2f}"))
open(os.path.join(OUT_DIR,"graph_semantic.html"),"w").write(HTML)
print("wrote graph_semantic.html")
print("biggest topics:", ", ".join(f"{labels[c]}({int(size[cidx[c]])})"
      for c in sorted(cids, key=lambda c:-size[cidx[c]])[:12]))
