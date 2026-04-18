let drawing=false,startX=0,startY=0,currentRect=null,rects=[];
const canvas=()=>document.getElementById('bbox-canvas');

window.addEventListener('DOMContentLoaded',()=>{
  const c=canvas();
  if(!c)return;
  const ctx=c.getContext('2d');
  c.addEventListener('mousedown',(e)=>{drawing=true;const r=c.getBoundingClientRect();startX=e.clientX-r.left;startY=e.clientY-r.top;});
  c.addEventListener('mousemove',(e)=>{
    if(!drawing)return;
    const r=c.getBoundingClientRect();
    const x=e.clientX-r.left,y=e.clientY-r.top;
    currentRect={x:Math.round(Math.min(startX,x)),y:Math.round(Math.min(startY,y)),w:Math.round(Math.abs(x-startX)),h:Math.round(Math.abs(y-startY))};
    redraw(ctx,c);
  });
  c.addEventListener('mouseup',()=>{if(currentRect){rects.push(currentRect);document.getElementById('bbox-output').textContent=JSON.stringify(currentRect);}drawing=false;currentRect=null;redraw(ctx,c);});
  redraw(ctx,c);
});

function redraw(ctx,c){
  ctx.clearRect(0,0,c.width,c.height);
  ctx.fillStyle='#f9fafb';ctx.fillRect(0,0,c.width,c.height);
  ctx.strokeStyle='#2563eb';ctx.lineWidth=2;
  for(const r of rects){ctx.strokeRect(r.x,r.y,r.w,r.h)}
  if(currentRect){ctx.strokeStyle='#dc2626';ctx.strokeRect(currentRect.x,currentRect.y,currentRect.w,currentRect.h)}
}

async function saveConfig(){
  const name=document.getElementById('config-name').value.trim();
  if(!name){alert('Config name required');return;}
  let payload={};
  try{payload=JSON.parse(document.getElementById('config-json').value||'{}');}
  catch(e){alert('Invalid JSON');return;}
  const res=await fetch(`/api/configs/${encodeURIComponent(name)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!res.ok){alert('Save failed');return;}
  location.href=`/configs?name=${encodeURIComponent(name)}`;
}

async function deleteConfig(){
  const name=document.getElementById('config-name').value.trim();
  if(!name){return;}
  if(!confirm(`Delete ${name}?`))return;
  await fetch(`/api/configs/${encodeURIComponent(name)}`,{method:'DELETE'});
  location.href='/configs';
}

const illegibleFields = {};
function markIllegible(field){illegibleFields[field]=true;const i=document.querySelector(`input[name="${field}"]`);if(i)i.value='';}

async function submitReview(jobId){
  const corrections={};
  document.querySelectorAll('#review-form input[type="text"]').forEach((el)=>{
    corrections[el.name]=illegibleFields[el.name]?'__ILLEGIBLE__':el.value;
  });
  const res=await fetch(`/api/jobs/${jobId}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corrections})});
  if(!res.ok){alert('Failed to submit');return;}
  location.href='/jobs';
}
