let vfState = {
  canvas: null,
  ctx: null,
  img: null,
  fields: [],
  selected: -1,
  dragMode: null,
  startX: 0,
  startY: 0,
  fieldCounter: 1,
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
  ctx.fillStyle='#0f172a';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  if(img) ctx.drawImage(img,0,0,canvas.width,canvas.height);

  fields.forEach((f,i)=>{
    const b=f.bounding_box||{};
    ctx.lineWidth=2;
    ctx.strokeStyle=i===selected?'#f472b6':'#8b5cf6';
    ctx.strokeRect(b.x||0,b.y||0,b.w||0,b.h||0);
    ctx.fillStyle=i===selected?'rgba(244,114,182,0.1)':'rgba(139,92,246,0.1)';
    ctx.fillRect(b.x||0,b.y||0,b.w||0,b.h||0);
    
    // Label tag
    ctx.fillStyle=i===selected?'#f472b6':'#8b5cf6';
    ctx.font='600 12px Inter, sans-serif';
    const txt = f.name||`field_${i+1}`;
    const tw = ctx.measureText(txt).width;
    ctx.fillRect((b.x||0), (b.y||0)-20, tw+10, 20);
    ctx.fillStyle='#ffffff';
    ctx.fillText(txt,(b.x||0)+5,(b.y||0)-6);

    // Resize handle
    ctx.fillStyle='#ffffff';
    ctx.fillRect((b.x||0)+(b.w||0)-6,(b.y||0)+(b.h||0)-6,12,12);
    ctx.strokeStyle=i===selected?'#f472b6':'#8b5cf6';
    ctx.strokeRect((b.x||0)+(b.w||0)-6,(b.y||0)+(b.h||0)-6,12,12);
  });

  const out = document.getElementById('bbox-output');
  if(out && selected >= 0){
    out.textContent = JSON.stringify(vfState.fields[selected], null, 2);
  }
}

function vfLoadTemplate(templatePath){
  if(!templatePath) return;
  const img = new Image();
  img.onload = () => { vfState.img = img; vfRedraw(); };
  img.src = `/api/template-preview?template_path=${encodeURIComponent(templatePath)}&v=${Date.now()}`;
}

function vfRefreshCounter(){
  const nums = vfState.fields.map((f) => {
    const m = String(f.name || '').match(/^field_(\d+)$/);
    return m ? parseInt(m[1], 10) : 0;
  });
  vfState.fieldCounter = Math.max(1, ...(nums.length ? nums : [1])) + 1;
}

function vfSyncFieldsPanel(){
  const panel = document.getElementById('field-list');
  if(!panel) return;
  panel.innerHTML = '';
  vfState.fields.forEach((f, idx) => {
    const b = f.bounding_box || {};
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = idx === vfState.selected ? 'secondary' : 'ghost';
    btn.style.width = '100%';
    btn.style.justifyContent = 'space-between';
    btn.style.marginBottom = '0.5rem';
    btn.innerHTML = `<span>${f.name || `field_${idx+1}`}</span><span class="muted">${Math.round(b.x||0)},${Math.round(b.y||0)} ${Math.round(b.w||0)}×${Math.round(b.h||0)}</span>`;
    btn.onclick = () => { vfState.selected = idx; vfRedraw(); vfSyncFieldsPanel(); };
    panel.appendChild(btn);
  });
}

function vfAddField(){
  const name = `field_${vfState.fieldCounter++}`;
  vfState.fields.push({
    name,
    label_hint: name,
    expected_type: 'string',
    critical: false,
    bounding_box: {x: 50, y: 50, w: 220, h: 40}
  });
  vfState.selected = vfState.fields.length - 1;
  vfRedraw();
  vfSyncFieldsPanel();
}

function vfDeleteSelectedField(){
  if(vfState.selected < 0) return;
  vfState.fields.splice(vfState.selected, 1);
  vfState.selected = Math.min(vfState.selected, vfState.fields.length - 1);
  vfRedraw();
  vfSyncFieldsPanel();
}

function vfInit(){
  const canvas = document.getElementById('bbox-canvas');
  if(!canvas) return;
  vfState.canvas = canvas;
  vfState.ctx = canvas.getContext('2d');
  vfState.fields = JSON.parse(canvas.dataset.fields || '[]');
  vfRefreshCounter();
  vfLoadTemplate(canvas.dataset.template || '');
  vfRedraw();
  vfSyncFieldsPanel();

  canvas.addEventListener('mousedown', (e)=>{
    const p=vfCanvasPos(e);
    const hit=vfHitRect(p.x,p.y);
    vfState.selected=hit.idx;
    vfState.dragMode=hit.mode;
    vfState.startX=p.x;vfState.startY=p.y;
    vfRedraw();
    vfSyncFieldsPanel();
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
    vfRefreshCounter();
    vfLoadTemplate(cfg.template_path || '');
    vfState.selected = -1;
    vfRedraw();
    vfSyncFieldsPanel();
  }catch(err){alert('Invalid JSON');}
}

function vfApplyBoxesToJson(){
  const ta = document.getElementById('config-json');
  if(!ta) return;
  try{
    const cfg = JSON.parse(ta.value||'{}');
    cfg.fields = vfState.fields;
    if (vfState.canvas) {
      cfg.editor_canvas = { width: vfState.canvas.width, height: vfState.canvas.height };
    }
    ta.value = JSON.stringify(cfg, null, 2);
    alert('Fields synced to JSON editor');
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

async function vfDiscoverFromImage() {
  const fileInput = document.getElementById('ai-template-file');
  const status = document.getElementById('ai-status');
  if (!fileInput || !fileInput.files[0]) {
    alert('Please select a blank template image first.');
    return;
  }

  status.textContent = 'Processing with AI (Gemini)... please wait...';
  const formData = new FormData();
  formData.append('template_file', fileInput.files[0]);

  try {
    const res = await fetch('/api/config/discover', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('AI discovery failed');
    
    const data = await res.json();
    if (data.error) {
       status.textContent = 'Error: ' + data.error;
       return;
    }

    if (data.fields) {
      // Load the image into the canvas background
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => { vfState.img = img; vfRedraw(); };
        img.src = e.target.result;
      };
      reader.readAsDataURL(fileInput.files[0]);

      // NEW: Scale normalized AI coordinates (0-1000) to our 800x1100 canvas
      const cw = vfState.canvas ? vfState.canvas.width : 800;
      const ch = vfState.canvas ? vfState.canvas.height : 1100;
      const scaledFields = data.fields.map(f => {
        const b = f.bounding_box || {x: 0, y: 0, w: 100, h: 20};
        return {
          ...f,
          expected_type: f.type === 'checkbox' ? 'checkbox' : 'string',
          label_hint: f.label || f.name,
          bounding_box: {
            x: Math.round((b.x / 1000) * cw),
            y: Math.round((b.y / 1000) * ch),
            w: Math.round((b.w / 1000) * cw),
            h: Math.round((b.h / 1000) * ch)
          }
        };
      });

      const ta = document.getElementById('config-json');
      const current = JSON.parse(ta.value || '{}');
      current.fields = scaledFields;
      current.editor_canvas = { width: cw, height: ch };
      ta.value = JSON.stringify(current, null, 2);
      
      // Update canvas state
      vfState.fields = scaledFields;
      vfRefreshCounter();
      vfRedraw();
      vfSyncFieldsPanel();
      
      status.textContent = `Found ${data.fields.length} fields! Check the editor below.`;
    } else {
      status.textContent = 'No fields found or AI error.';
    }
  } catch (err) {
    console.error(err);
    status.textContent = 'Error: ' + err.message;
  }
}
