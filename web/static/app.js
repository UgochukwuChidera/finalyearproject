let vfState = {
  canvas: null,
  ctx: null,
  img: null,
  fields: [],
  selected: -1,
  dragMode: null,
  startX: 0,
  startY: 0,
};

function vfCanvasPos(e){
  const r = vfState.canvas.getBoundingClientRect();
  return {x: e.clientX - r.left, y: e.clientY - r.top};
}

function vfHitRect(x,y){
  for(let i=vfState.fields.length-1;i>=0;i--){
    const b=vfState.fields[i].bounding_box||{};
    const hx=(b.x||0)+(b.w||0)-8, hy=(b.y||0)+(b.h||0)-8;
    if(x>=hx && x<=hx+16 && y>=hy && y<=hy+16) return {idx:i, mode:'resize'};
    if(x>=b.x && x<=b.x+b.w && y>=b.y && y<=b.y+b.h) return {idx:i, mode:'move'};
  }
  return {idx:-1, mode:null};
}

function vfRedraw(){
  const {ctx,canvas,img,fields,selected}=vfState;
  if(!ctx||!canvas) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#f3f4f6';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  if(img) ctx.drawImage(img,0,0,canvas.width,canvas.height);

  fields.forEach((f,i)=>{
    const b=f.bounding_box||{};
    ctx.lineWidth=2;
    ctx.strokeStyle=i===selected?'#dc2626':'#2563eb';
    ctx.strokeRect(b.x||0,b.y||0,b.w||0,b.h||0);
    ctx.fillStyle='rgba(37,99,235,0.15)';
    ctx.fillRect(b.x||0,b.y||0,b.w||0,b.h||0);
    ctx.fillStyle='#111827';
    ctx.font='12px sans-serif';
    ctx.fillText(f.name||`field_${i+1}`,(b.x||0)+2,(b.y||0)-4);

    ctx.fillStyle='#111827';
    ctx.fillRect((b.x||0)+(b.w||0)-6,(b.y||0)+(b.h||0)-6,12,12);
  });

  const out = document.getElementById('bbox-output');
  if(out && selected >= 0){
    out.textContent = JSON.stringify(vfState.fields[selected], null, 2);
  }
}

function vfLoadTemplate(templatePath){
  if(!templatePath) return;
  const name = templatePath.split('/').pop();
  const img = new Image();
  img.onload = () => { vfState.img = img; vfRedraw(); };
  img.src = `/templates/${name}`;
}

function vfInit(){
  const canvas = document.getElementById('bbox-canvas');
  if(!canvas) return;
  vfState.canvas = canvas;
  vfState.ctx = canvas.getContext('2d');
  vfState.fields = JSON.parse(canvas.dataset.fields || '[]');
  vfLoadTemplate(canvas.dataset.template || '');
  vfRedraw();

  canvas.addEventListener('mousedown', (e)=>{
    const p=vfCanvasPos(e);
    const hit=vfHitRect(p.x,p.y);
    vfState.selected=hit.idx;
    vfState.dragMode=hit.mode;
    vfState.startX=p.x;vfState.startY=p.y;
    vfRedraw();
  });

  canvas.addEventListener('mousemove', (e)=>{
    if(vfState.selected<0||!vfState.dragMode) return;
    const p=vfCanvasPos(e);
    const f=vfState.fields[vfState.selected];
    const b=f.bounding_box||(f.bounding_box={x:0,y:0,w:0,h:0});
    const dx=Math.round(p.x-vfState.startX), dy=Math.round(p.y-vfState.startY);
    if(vfState.dragMode==='move'){
      b.x = Math.max(0,(b.x||0)+dx);
      b.y = Math.max(0,(b.y||0)+dy);
    } else {
      b.w = Math.max(1,(b.w||1)+dx);
      b.h = Math.max(1,(b.h||1)+dy);
    }
    vfState.startX=p.x;vfState.startY=p.y;
    vfRedraw();
  });

  canvas.addEventListener('mouseup', ()=>{vfState.dragMode=null;});
  canvas.addEventListener('mouseleave', ()=>{vfState.dragMode=null;});
}
window.addEventListener('DOMContentLoaded', vfInit);

function vfLoadFromJson(){
  const ta = document.getElementById('config-json');
  if(!ta) return;
  try{
    const cfg = JSON.parse(ta.value||'{}');
    vfState.fields = cfg.fields || [];
    vfLoadTemplate(cfg.template_path || '');
    vfState.selected = -1;
    vfRedraw();
  }catch(err){alert('Invalid JSON');}
}

function vfApplyBoxesToJson(){
  const ta = document.getElementById('config-json');
  if(!ta) return;
  try{
    const cfg = JSON.parse(ta.value||'{}');
    const boxByName = {};
    vfState.fields.forEach(f=>{boxByName[f.name]=f.bounding_box||{};});
    (cfg.fields||[]).forEach(f=>{if(boxByName[f.name]) f.bounding_box = boxByName[f.name];});
    ta.value = JSON.stringify(cfg, null, 2);
    alert('Bounding boxes applied to JSON editor');
  }catch(err){alert('Invalid JSON');}
}

const illegibleFields = {};
function markIllegible(field){
  illegibleFields[field]=true;
  const i=document.querySelector(`input[name="${field}"]`);
  if(i)i.value='';
}

async function submitReview(jobId){
  const corrections={};
  document.querySelectorAll('#review-form input[type="text"]').forEach((el)=>{
    corrections[el.name]=illegibleFields[el.name]?'__ILLEGIBLE__':el.value;
  });
  const res=await fetch(`/jobs/${jobId}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corrections})});
  if(!res.ok){alert('Failed to submit');return;}
  location.href=`/jobs/${jobId}`;
}
