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
  zoom: 1.0,
  aspectRatio: 'auto',
  isPanning: false,
};

function vfCanvasPos(e){
  const r = vfState.canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / vfState.zoom,
    y: (e.clientY - r.top) / vfState.zoom
  };
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
  
  // Background
  ctx.fillStyle='#0f172a';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  
  if(img) {
    ctx.drawImage(img,0,0,canvas.width,canvas.height);
  } else {
    // Placeholder if no image
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.font = '24px Outfit';
    ctx.textAlign = 'center';
    ctx.fillText('No Template Image Overlaid', canvas.width/2, canvas.height/2);
    ctx.textAlign = 'start';
  }

  fields.forEach((f,i)=>{
    const b=f.bounding_box||{};
    ctx.lineWidth=2;
    ctx.strokeStyle=i===selected?'#f472b6':'#8b5cf6';
    
    // Draw box
    ctx.strokeRect(b.x||0,b.y||0,b.w||0,b.h||0);
    ctx.fillStyle=i===selected?'rgba(244,114,182,0.15)':'rgba(139,92,246,0.1)';
    ctx.fillRect(b.x||0,b.y||0,b.w||0,b.h||0);
    
    // Label tag
    ctx.fillStyle=i===selected?'#f472b6':'#8b5cf6';
    ctx.font='600 12px Inter, sans-serif';
    const txt = f.name||`field_${i+1}`;
    const tw = ctx.measureText(txt).width;
    ctx.fillRect((b.x||0), (b.y||0)-22, tw+14, 22);
    ctx.fillStyle='#ffffff';
    ctx.fillText(txt,(b.x||0)+7,(b.y||0)-7);

    // Resize handle (only for selected)
    if (i === selected) {
      ctx.fillStyle='#ffffff';
      ctx.fillRect((b.x||0)+(b.w||0)-6,(b.y||0)+(b.h||0)-6,12,12);
      ctx.strokeStyle='#f472b6';
      ctx.strokeRect((b.x||0)+(b.w||0)-6,(b.y||0)+(b.h||0)-6,12,12);
    }
  });

  const out = document.getElementById('bbox-output');
  if(out && selected >= 0){
    out.textContent = JSON.stringify(vfState.fields[selected], null, 2);
    vfPopulatePropertyEditor(vfState.fields[selected]);
  } else if (out) {
    out.textContent = "// Select a field to see details";
    const pe = document.getElementById('property-editor');
    if(pe) pe.style.display = 'none';
  }
}

function vfPopulatePropertyEditor(field) {
  const pe = document.getElementById('property-editor');
  if (!pe) return;
  pe.style.display = 'block';
  
  document.getElementById('prop-name').value = field.name || '';
  document.getElementById('prop-label').value = field.label_hint || '';
  document.getElementById('prop-type').value = field.expected_type || 'string';
  document.getElementById('prop-critical').checked = field.critical || false;
  
  const b = field.bounding_box || {};
  document.getElementById('prop-x').value = b.x || 0;
  document.getElementById('prop-y').value = b.y || 0;
  document.getElementById('prop-w').value = b.w || 0;
  document.getElementById('prop-h').value = b.h || 0;
}

function vfUpdateFieldProperty(path, value) {
  if (vfState.selected < 0) return;
  const field = vfState.fields[vfState.selected];
  
  if (path.startsWith('bounding_box.')) {
    const key = path.split('.')[1];
    if (!field.bounding_box) field.bounding_box = {x:0, y:0, w:0, h:0};
    field.bounding_box[key] = value;
  } else {
    field[path] = value;
  }
  
  vfRedraw();
  vfSyncFieldsPanel();
}

async function vfLoadLocalImage(file) {
  if (!file) return;
  console.log('Loading local image overlay:', file.name);
  
  const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');
  
  if (isTiff) {
    console.log('TIFF detected, using server-side conversion');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/utils/convert-to-png', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Server conversion failed');
      const blob = await res.blob();
      const img = new Image();
      img.onload = () => {
        vfState.img = img;
        vfRedraw();
      };
      img.src = URL.createObjectURL(blob);
      return;
    } catch (err) {
      console.error(err);
      alert('Failed to process TIFF file. Please try a PNG or JPG.');
      return;
    }
  }

  // Normal browser-supported images
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      console.log('Image object ready, redrawing');
      vfState.img = img;
      vfRedraw();
    };
    img.onerror = (err) => {
      console.error('Image load error:', err);
      alert('Failed to load image file. Try a different format (PNG/JPG).');
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function vfHandlePreviewUpload(e) {
  vfLoadLocalImage(e.target.files[0]);
}

function vfLoadTemplate(templatePath){
  if(!templatePath) return;
  const img = new Image();
  img.onload = () => { vfState.img = img; vfRedraw(); };
  img.onerror = () => {
    const out = document.getElementById('bbox-output');
    if (out) out.textContent = `Failed to load template preview: ${templatePath}`;
  };
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
    const left = document.createElement('span');
    left.textContent = f.name || `field_${idx+1}`;
    const right = document.createElement('span');
    right.className = 'muted';
    right.textContent = `${Math.round(b.x||0)},${Math.round(b.y||0)} ${Math.round(b.w||0)}×${Math.round(b.h||0)}`;
    btn.appendChild(left);
    btn.appendChild(right);
    btn.onclick = () => { 
      vfState.selected = idx; 
      vfRedraw(); 
      vfSyncFieldsPanel(); 
      vfPopulatePropertyEditor(f);
    };
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

  const viewport = document.getElementById('canvas-viewport');

  canvas.addEventListener('mousedown', (e)=>{
    const p=vfCanvasPos(e);
    const hit=vfHitRect(p.x,p.y);
    
    if (hit.idx === -1 && e.button === 0) {
      vfState.isPanning = true;
      vfState.startX = e.clientX;
      vfState.startY = e.clientY;
      canvas.style.cursor = 'grabbing';
    } else {
      vfState.selected=hit.idx;
      vfState.dragMode=hit.mode;
      vfState.startX=p.x;vfState.startY=p.y;
    }
    vfRedraw();
    vfSyncFieldsPanel();
  });

  window.addEventListener('mousemove', (e)=>{
    if (vfState.isPanning) {
      const dx = e.clientX - vfState.startX;
      const dy = e.clientY - vfState.startY;
      viewport.scrollLeft -= dx;
      viewport.scrollTop -= dy;
      vfState.startX = e.clientX;
      vfState.startY = e.clientY;
      return;
    }

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

  window.addEventListener('mouseup', ()=>{
    vfState.dragMode=null;
    vfState.isPanning = false;
    canvas.style.cursor = 'grab';
  });

  viewport.addEventListener('wheel', (e) => {
    if (e.ctrlKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      vfZoom(delta);
    }
  }, { passive: false });

  // Also hook into the AI template file input for instant preview
  const aiFile = document.getElementById('ai-template-file');
  if (aiFile) {
    aiFile.addEventListener('change', (e) => {
      if (e.target.files[0]) vfLoadLocalImage(e.target.files[0]);
    });
  }
}

function vfZoom(delta) {
  vfState.zoom = Math.min(Math.max(0.1, vfState.zoom + delta), 5.0);
  const canvas = document.getElementById('bbox-canvas');
  if (canvas) {
    canvas.style.transform = `scale(${vfState.zoom})`;
    document.getElementById('zoom-level').textContent = `${Math.round(vfState.zoom * 100)}%`;
  }
}

function vfResetZoom() {
  vfState.zoom = 1.0;
  vfZoom(0);
}

function vfUpdateAspectRatio(ratio) {
  vfState.aspectRatio = ratio;
  const canvas = vfState.canvas;
  if (!canvas) return;
  
  if (ratio === 'auto') {
    if (vfState.img) {
      canvas.height = (vfState.img.height / vfState.img.width) * canvas.width;
    }
  } else {
    const r = parseFloat(ratio.split(':')[1]) / parseFloat(ratio.split(':')[0]);
    canvas.height = canvas.width * r;
  }
  
  document.getElementById('viewport-h').value = Math.round(canvas.height);
  vfRedraw();
}

function vfUpdateDimensions(type, value) {
  const canvas = vfState.canvas;
  if (!canvas) return;
  
  if (type === 'w') canvas.width = parseInt(value) || 800;
  if (type === 'h') canvas.height = parseInt(value) || 1100;
  
  vfRedraw();
}

function vfToggleFullscreen() {
  const root = document.getElementById('workspace-root');
  if (!document.fullscreenElement) {
    root.requestFullscreen().catch(err => {
      alert(`Error attempting to enable full-screen mode: ${err.message}`);
    });
  } else {
    document.exitFullscreen();
  }
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
      // Coordinates are already handled, the image should already be loaded via the 'change' listener
      // but we call it again just in case or to ensure sync
      if (fileInput.files[0]) vfLoadLocalImage(fileInput.files[0]);

      // Scale normalized AI coordinates (0-1000) to our 800x1100 canvas
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
