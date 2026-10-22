"""Offline semantic inspection: full-mesh labels/predictions and simplified wireframe."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..config import resolve
from ..cohort import read_manifest
from ..train_regions import load_jaws,binary_metrics
from .segmentation import load_model,predict


def export(cfg):
    out=resolve(cfg['out']);model=load_model(cfg); rows=[]
    directory=out/'viewer';directory.mkdir(parents=True,exist_ok=True)
    for split,paths in read_manifest(out/'manifest.json').items():
        for item in load_jaws(paths):
            for name,angles in [('原始',[0,0,0]),('旋转 A',[70,-35,110]),('旋转 B',[-120,60,170])]:
                r=Rotation.from_euler('xyz',angles,degrees=True).as_matrix();pivot=item['xyz'].mean(0)
                transformed=dict(item,xyz=(item['xyz']-pivot)@r.T+pivot,normal=item['normal']@r.T)
                probability,mapping=predict(model,transformed,cfg['segmentation']['sampling'])
                ids=np.linspace(0,len(probability)-1,min(12000,len(probability)),dtype=int)
                prefix='upper' if item['jaw']==0 else 'lower'
                meshfile=out/'regions/cases'/item['case']/(prefix+'_mapping.npz')
                simplified=None
                if meshfile.exists():
                    with np.load(meshfile) as data:
                        simple_ids=data['simplified_vertex_ids']; faces=data['simplified_faces']
                    simplified=dict(xyz=np.round(transformed['xyz'][simple_ids],3).tolist(),faces=faces.tolist())
                rows.append(dict(case=item['case'],split=split,jaw=prefix,pose=name,angles=angles,
                    xyz=np.round(transformed['xyz'][ids],3).tolist(),prob=np.round(probability[ids],4).tolist(),
                    label=item['label'][ids].astype(int).tolist(),
                    voxels=np.round(transformed['xyz'][mapping['ids']],3).tolist(),
                    voxel_probability=np.round(probability[mapping['ids']],4).tolist(),
                    total=len(probability),voxel_count=len(mapping['ids']),voxel_size=mapping['size_mm'],simplified=simplified,
                    metrics=binary_metrics(probability,item['label'],cfg['segmentation']['threshold'])))
                np.savez_compressed(directory/(item['case']+'_'+prefix+'_'+str(len(rows))+'.npz'),
                    original_vertex_probability=probability,voxel_vertex_ids=mapping['ids'],vertex_to_voxel=mapping['inverse'])
    payload=dict(rows=rows,threshold=cfg['segmentation']['threshold'],smoke=bool(cfg.get('smoke')))
    (directory/'data.js').write_text('const DATA='+json.dumps(payload,separators=(',',':'))+';\n')
    (directory/'index.html').write_text(HTML)
    print('Viewer:',directory/'index.html',flush=True)


HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>咬合作用区域 · 语义检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#101923;color:#e4edf6;font:15px system-ui}main{max-width:1440px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 8px}.muted{color:#a9becf}header{margin-bottom:22px}nav{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}select,button{padding:10px;border:1px solid #486078;background:#203246;color:#e4edf6;border-radius:7px}section{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px}canvas{width:100%;height:650px;background:#152431;border-radius:12px;touch-action:none}aside{background:#1c2b39;padding:18px;border-radius:12px;line-height:1.8}.metric{font-size:23px;font-weight:650;color:#56d9c0}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}#status{color:#f9c76c}small{display:block;margin-top:12px;color:#a9becf}@media(max-width:750px){main{padding:12px}section{grid-template-columns:1fr}canvas{height:430px}h1{font-size:23px}}
</style><main><nav id="pipeline-links"><a href="teacher.html">Teacher 评分</a> · <a href="agent.html">Agent 轨迹</a></nav><header><h1>咬合作用区域 · 语义检查</h1><div class="muted">5 mm 原始咬合标签 → 体素代表顶点 → PointTransformer → 原 STL 顶点反映射</div><p id="status"></p></header>
<nav><select id="case" aria-label="病例"></select><select id="jaw" aria-label="上下颌"><option value="upper">上颌</option><option value="lower">下颌</option></select><select id="pose" aria-label="输入旋转"></select><select id="mode" aria-label="显示模式"><option value="pred">原网格语义预测</option><option value="label">原始 5 mm 标签</option><option value="error">预测错误对照</option><option value="voxel">网络体素输入</option><option value="mesh">简面三角网格</option></select><button id="reset">重置视角</button></nav>
<section><div><canvas id="view"></canvas><div id="legend" class="legend"></div><div class="muted">拖动旋转视角 · 滚轮缩放 · 输入旋转会重新体素采样和预测</div></div><aside id="info"></aside></section>
<small>页面为原网格顶点的均匀索引显示子集（最多 12000 点），指标基于全部原始顶点。体素视图显示全部实际网络输入。完整映射保存在旁边的 NPZ；完整彩色原三角网格见 regions/cases/*/*_semantic.ply。简面视图展示原始输入分割后的缓存子网格随刚体旋转，非该旋转的再次简面结果。</small></main><script src="data.js"></script><script>
const $=id=>document.getElementById(id), canvas=$('view'),ctx=canvas.getContext('2d');let yaw=.1,pitch=.4,zoom=1,drag=null,row;
const cases=[...new Set(DATA.rows.map(x=>x.case))];$('case').innerHTML=cases.map(x=>`<option>${x}</option>`).join('');
$('pose').innerHTML=[...new Set(DATA.rows.map(x=>x.pose))].map(x=>`<option>${x}</option>`).join('');
$('status').textContent=DATA.smoke?'缩短版流程测试模型：尚不能作为正式分割质量结论':'正式训练输出：请结合独立测试指标检查旋转稳定性';
function select(){row=DATA.rows.find(r=>r.case==$('case').value&&r.jaw==$('jaw').value&&r.pose==$('pose').value);const m=row.metrics;
$('info').innerHTML=`<b>病例 ${row.case} · ${row.split}</b><p>全网格 IoU<br><span class="metric">${(m.iou*100).toFixed(1)}%</span></p>Dice ${(m.dice*100).toFixed(1)}%<br>精确率 ${(m.precision*100).toFixed(1)}%<br>召回率 ${(m.recall*100).toFixed(1)}%<hr>原始顶点 ${row.total.toLocaleString()}<br>体素输入 ${row.voxel_count.toLocaleString()}<br>实际体素边长 ${row.voxel_size.toFixed(3)} mm<br>预测阳性 ${(m.predicted_fraction*100).toFixed(1)}%<br>标签阳性 ${(m.positive_fraction*100).toFixed(1)}%<br>简面顶点 ${row.simplified?row.simplified.xyz.length:'未生成'}<hr>输入 Euler xyz 角度：<br>${row.angles.join(' / ')}°<small>标签仅用于监督与检查，不作为预测输入。</small>`;draw();}
function draw(){if(!row)return;const mode=$('mode').value,isMesh=mode==='mesh';let points=mode==='voxel'?row.voxels:isMesh?(row.simplified?.xyz||[]):row.xyz;
const rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);
const labels=mode==='error'?[['#f26a84','假阳性'],['#ffbf69','漏分'],['#48d9b0','正确阳性'],['#567084','正确阴性']]:[['#48d9b0','咬合作用区域'],['#567084','其他区域']];$('legend').innerHTML=labels.map(([c,t])=>`<span><i class="dot" style="background:${c}"></i>${t}</span>`).join('');
if(!points.length){ctx.fillStyle='#fff';ctx.fillText('尚未生成简面网格',30,50);return;}
const center=[0,0,0];points.forEach(p=>p.forEach((v,k)=>center[k]+=v/points.length));let radius=0;points.forEach(p=>radius=Math.max(radius,Math.hypot(...p.map((v,k)=>v-center[k]))));const scale=Math.min(w,h)*.43/Math.max(radius,1)*zoom;
const projected=points.map((p,i)=>{const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],xx=x*Math.cos(yaw)+z*Math.sin(yaw),zz=-x*Math.sin(yaw)+z*Math.cos(yaw);return [w/2+xx*scale,h/2+(y*Math.cos(pitch)-zz*Math.sin(pitch))*scale,y*Math.sin(pitch)+zz*Math.cos(pitch),i];});
if(isMesh){ctx.strokeStyle='#48d9b088';ctx.lineWidth=.5;row.simplified.faces.forEach(f=>{ctx.beginPath();f.forEach((i,j)=>j?ctx.lineTo(projected[i][0],projected[i][1]):ctx.moveTo(projected[i][0],projected[i][1]));ctx.closePath();ctx.stroke();});return;}
projected.sort((a,b)=>a[2]-b[2]).forEach(p=>{const i=p[3],positive=(mode==='voxel'?row.voxel_probability[i]:row.prob[i])>=DATA.threshold,truth=!!row.label[i];let c=positive?'#48d9b0':'#567084';if(mode==='label')c=truth?'#48d9b0':'#567084';if(mode==='error')c=positive&&!truth?'#f26a84':!positive&&truth?'#ffbf69':positive?'#48d9b0':'#567084';ctx.fillStyle=c;const r=mode==='voxel'?2.6:1.3;ctx.fillRect(p[0],p[1],r,r);});}
['case','jaw','pose','mode'].forEach(id=>$(id).addEventListener('change',select));$('reset').onclick=()=>{yaw=.1;pitch=.4;zoom=1;draw();};canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw();}};canvas.onpointerup=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw();};window.onresize=draw;select();
</script></html>'''
