#!/usr/bin/env python3
"""Local diagnostic identity review; zero-based frames, independent annotation first."""
import argparse, csv, hashlib, json, math, os, re, secrets, sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / 'videos/5.mp4'
LOG = ROOT / 'logs/tracking_log_20261026_023427.csv'
OUT = ROOT / 'research_handoff/review/sessions'
SCHEMA = 'human-identity-review/1.0'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def box_ok(box, width, height):
    return (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)

def validate_label(r, width, height, frame_count, video_hash):
    if r.get('schema_version') != SCHEMA or r.get('record_type') != 'annotation': raise ValueError('unsupported annotation schema')
    if type(r.get('frame')) is not int or not 0 <= r['frame'] < frame_count: raise ValueError('invalid frame')
    if r.get('video_sha256') != video_hash: raise ValueError('video hash mismatch')
    for k in ('reviewer_id', 'evidence', 'recorded_at_utc', 'session_id'):
        if not isinstance(r.get(k), str) or not r[k].strip(): raise ValueError('missing ' + k)
    if r.get('label') not in ('TARGET', 'TARGET_NOT_VISIBLE', 'UNKNOWN'): raise ValueError('invalid label')
    if r['label'] == 'TARGET':
        if not box_ok(r.get('box'), width, height): raise ValueError('TARGET requires a valid box')
    elif r.get('box') is not None: raise ValueError('non-target label must not have a box')
    if not isinstance(r.get('notes'), str): raise ValueError('invalid notes')
    if r.get('annotation_mode') != 'blind': raise ValueError('annotation must be blind')
    return r

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def category(label, drawn, output_box):
    if label['label'] != 'TARGET': return 'visibility_unknown_output' if drawn else 'visibility_unknown_no_output'
    if not drawn: return 'no_output'
    if output_box is None: return 'output_box_unavailable'
    value = iou(label['box'], output_box)
    return 'low_overlap_review' if value < .1 else 'overlap_agreement' if value >= .5 else 'ambiguous_overlap'

def queue(rows):
    # Diagnostic selection: known candidate events, historical view boundaries, state transitions and crowd peaks.
    priority = [1061,1062,1066,1140,1141,1145]
    boundaries = [905,1751,2085,2175,2205,2265,2385,2445,2505,2565,2625,2655]
    result = priority + [n for x in boundaries for n in (x-1,x,x+1) if 0 <= n < len(rows)]
    result += [0, 60] if len(rows) > 60 else [0]
    result += [i for i in range(1,len(rows)) if rows[i]['state'] != rows[i-1]['state']]
    result += sorted(range(len(rows)), key=lambda i: (-int(rows[i]['num_people']), i))[:15]
    return list(dict.fromkeys(result))

def read_log(path):
    with open(path, newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
    if not rows: raise ValueError('empty log')
    for i, r in enumerate(rows):
        if int(r['frame']) != i or r['valid'] not in ('0','1') or r['state'] not in ('LOST','TRACKING','INITIALIZING','IDLE') or int(r['num_people']) < 0: raise ValueError('invalid log frame ' + str(i))
    return rows

def read_manifest(path, video_hash, log_hash, rows, width, height):
    if not path: return {}
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    if m.get('video_sha256') != video_hash or m.get('log_sha256') != log_hash or not m.get('config', {}).get('state_and_target_id_and_valid_checked_each_frame'): raise ValueError('manifest provenance mismatch')
    result = {}
    for r in m['frames']:
        n = r['frame']
        if type(n) is not int or not 0 <= n < len(rows) or n in result or r['state'] != rows[n]['state']: raise ValueError('invalid manifest frame')
        b = r['output_box']
        if (b is not None) != (rows[n]['valid'] == '1') or (b is not None and not box_ok(b,width,height)): raise ValueError('manifest output mismatch')
        for t in r['tracks']:
            if not box_ok(t['box'],width,height): raise ValueError('invalid track box')
        result[n] = r
    return result

def append(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', help='optional checked replay manifest from extract_candidates.py')
    p.add_argument('--port', type=int, default=8501)
    a = p.parse_args()
    rows = read_log(LOG); video_hash = digest(VIDEO); log_hash = digest(LOG)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened(): raise RuntimeError('cannot open video')
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0: raise ValueError('invalid video metadata')
    manifests = read_manifest(a.manifest,video_hash,log_hash,rows,width,height)
    manifest_hash = digest(a.manifest) if a.manifest else None
    OUT.mkdir(parents=True, exist_ok=True)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(6)
    path = OUT / (session + '.jsonl')
    with open(path, 'x', encoding='utf-8'): pass
    annotations = {}; comparisons = set(); frozen = {'value': False}; token = secrets.token_urlsafe(32)
    # Sequential decoding avoids unreliable seeks. Cache only last frame.
    cache = {'n': -1, 'jpg': None}
    def frame_jpeg(n):
        if cache['n'] == n: return cache['jpg']
        c = cv2.VideoCapture(str(VIDEO)); image = None
        try:
            for i in range(n+1):
                ok, image = c.read()
                if not ok: raise ValueError('decode failed at frame ' + str(i))
        finally: c.release()
        ok, data = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])
        if not ok: raise ValueError('JPEG encoding failed')
        cache.update(n=n,jpg=data.tobytes()); return cache['jpg']
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, kind='application/json'):
            if isinstance(body, (dict,list)): body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            elif isinstance(body,str): body=body.encode()
            self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            try:
                u=urlparse(self.path); q=parse_qs(u.query)
                if u.path == '/': return self.respond(200,HTML.replace('__TOKEN__',token), 'text/html; charset=utf-8')
                if u.path == '/api/info': return self.respond(200, {'width':width,'height':height,'fps':fps,'count':len(rows),'queue':queue(rows),'session':session,'video_hash':video_hash})
                if u.path == '/api/video':
                    size = VIDEO.stat().st_size
                    start, end = 0, size - 1
                    requested = self.headers.get('Range')
                    if requested:
                        match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
                        if not match: raise ValueError('unsupported byte range')
                        start = int(match.group(1)); end = min(int(match.group(2)) if match.group(2) else end, end)
                        if start > end: raise ValueError('invalid byte range')
                    self.send_response(206 if requested else 200)
                    self.send_header('Content-Type','video/mp4'); self.send_header('Accept-Ranges','bytes')
                    self.send_header('Content-Length',str(end-start+1))
                    if requested: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                    self.end_headers()
                    with open(VIDEO,'rb') as f:
                        f.seek(start); remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(1024*1024,remaining))
                            if not chunk: break
                            self.wfile.write(chunk); remaining-=len(chunk)
                    return
                if u.path == '/api/frame':
                    n=int(q['n'][0]);
                    if not 0 <= n < len(rows): raise ValueError('frame outside decoded/log range')
                    return self.respond(200,frame_jpeg(n),'image/jpeg')
                if u.path == '/api/compare':
                    n=int(q['n'][0])
                    if not frozen['value'] or n not in annotations: raise ValueError('finish and lock annotations first')
                    row=rows[n]; m=manifests.get(n); output=m['output_box'] if m else None
                    return self.respond(200,{'annotation':annotations[n], 'log':row,'replay':m,'category':category(annotations[n],row['valid']=='1',output),'iou':iou(annotations[n]['box'],output) if output and annotations[n]['label']=='TARGET' else None})
                self.respond(404,'not found','text/plain')
            except Exception as e: self.respond(400,{'error':str(e)})
        def do_POST(self):
            try:
                if self.headers.get('X-Review-Token') != token or self.headers.get('Origin') != 'http://localhost:'+str(a.port): raise ValueError('invalid request origin or token; use localhost')
                size=int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 65536: raise ValueError('invalid request size')
                d=json.loads(self.rfile.read(size)); n=d.get('frame'); now=datetime.now(timezone.utc).isoformat()
                if self.path == '/api/freeze':
                    if frozen['value'] or not annotations: raise ValueError('save at least one annotation before locking')
                    append(path, {'schema_version':SCHEMA,'record_type':'annotation_lock','session_id':session,'recorded_at_utc':now,'video_sha256':video_hash,'annotation_count':len(annotations)})
                    frozen['value'] = True
                    return self.respond(200,{'ok':True})
                if self.path == '/api/annotation':
                    if frozen['value']: raise ValueError('annotations locked; start a new session to annotate more frames')
                    r={'schema_version':SCHEMA,'record_type':'annotation','annotation_mode':'blind','session_id':session,'frame':n,'label':d.get('label'),'box':d.get('box'),'reviewer_id':d.get('reviewer_id'),'evidence':d.get('evidence'),'notes':d.get('notes',''),'recorded_at_utc':now,'video_sha256':video_hash,'video_path':'videos/5.mp4','selection':'diagnostic queue or manually selected','frame_index_basis':'zero-based sequential decode','tool':'identity_review.py'}
                    validate_label(r,width,height,len(rows),video_hash)
                    if n in annotations: raise ValueError('frame already annotated in this session; use a new session for independent correction')
                    append(path,r); annotations[n]=r; return self.respond(200,{'ok':True})
                if self.path == '/api/comparison':
                    if not frozen['value'] or n not in annotations or n in comparisons: raise ValueError('locked annotation required and comparison must be unique')
                    row=rows[n]; identity=d.get('identity')
                    if row['valid']=='0' and identity != 'none': raise ValueError('no output requires none')
                    if row['valid']=='1' and identity not in ('correct','wrong','uncertain'): raise ValueError('choose identity outcome')
                    if annotations[n]['label'] != 'TARGET' and identity in ('correct','wrong'): raise ValueError('uncertain visibility cannot establish identity outcome')
                    m=manifests.get(n)
                    if row['valid']=='1' and m is None and identity != 'uncertain': raise ValueError('output box unavailable; identity must remain uncertain')
                    r={'schema_version':SCHEMA,'record_type':'comparison','session_id':session,'frame':n,'reviewer_id':annotations[n]['reviewer_id'],'recorded_at_utc':now,'video_sha256':video_hash,'log_sha256':log_hash,'manifest_sha256':manifest_hash,'drawn':row['valid'],'state':row['state'],'identity':identity,'geometry_category':category(annotations[n],row['valid']=='1',m['output_box'] if m else None),'notes':str(d.get('notes',''))}
                    append(path,r); comparisons.add(n); return self.respond(200,{'ok':True})
                self.respond(404,{'error':'not found'})
            except Exception as e: self.respond(400,{'error':str(e)})
    print('Local URL: http://localhost:%d\nSession: %s' % (a.port,path),flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

HTML = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Human identity review</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:16px}canvas{max-width:100%;border:1px solid #777;cursor:crosshair}input,select,button,textarea{font:inherit;margin:4px}section{border:1px solid #bbb;padding:12px;margin:12px 0}.warn{color:#853b00}pre{white-space:pre-wrap;overflow-wrap:anywhere}#zoom{max-width:450px}</style>
<h1>Human identity review — video 5</h1><p class="warn">Review chẩn đoán, không phải held-out. Dùng bằng chứng mục tiêu độc lập; ID track không phải danh tính. Nếu không chắc, chọn UNKNOWN. TARGET_NOT_VISIBLE không chứng minh vắng mặt.</p>
<video controls width="600" src="/video" id="video" style="display:none"></video><button onclick="toggleVideo()">Xem / ẩn video gốc</button><div id="vid"></div>
<section><label>Reviewer ID <input id="reviewer"></label><label>Nguồn bằng chứng độc lập <input id="evidence" placeholder="Mô tả nguồn / tham chiếu"></label><p>Hàng đợi được chọn theo rủi ro, không hiển thị lý do hoặc output trong lúc gán nhãn.</p><select id="queue" onchange="go(+this.value)"></select><button onclick="go(n-1)">◀ Frame</button><input id="frame" type="number" min="0" value="0"><button onclick="go(+document.querySelector('#frame').value)">Đi tới ID</button><button onclick="go(n+1)">Frame ▶</button><span id="status"></span><p>Kéo chuột trên ảnh để vẽ hộp mục tiêu. Vùng phóng to xuất hiện bên dưới.</p><canvas id="canvas"></canvas><canvas id="zoom"></canvas><br><label>Nhãn <select id="label"><option value="UNKNOWN">UNKNOWN</option><option value="TARGET">TARGET (cần hộp)</option><option value="TARGET_NOT_VISIBLE">TARGET_NOT_VISIBLE</option></select></label><input id="notes" placeholder="Ghi chú"><button onclick="save()">Lưu nhãn độc lập</button></section>
<section><h2>Đối chiếu sau khi lưu</h2><button onclick="freeze()">Hoàn thành và khóa toàn bộ annotation</button> <button onclick="compare()">Mở tracking output cho frame đã lưu</button><p>Sau khi khóa, phiên này không nhận thêm annotation; hãy gán nhãn tất cả frame cần review trước khi đối chiếu.</p><pre id="comparison"></pre><div id="judgment" hidden><p>Chỉ xác nhận correct/wrong khi thấy hộp output và có bằng chứng độc lập; IoU không xác nhận danh tính.</p><select id="identity"><option value="uncertain">uncertain</option><option value="correct">correct</option><option value="wrong">wrong</option><option value="none">none (không output)</option></select><input id="compareNotes" placeholder="Ghi chú đối chiếu"><button onclick="saveComparison()">Lưu kết luận đối chiếu</button></div></section><p id="message"></p>
<script>
const token='__TOKEN__', $=s=>document.querySelector(s), c=$('#canvas'), ctx=c.getContext('2d'), z=$('#zoom'), zctx=z.getContext('2d'); let info,n=0,img=new Image(),box=null,start=null,comp=null;
async function api(url,opt){let r=await fetch(url,opt), x=await r.json();if(!r.ok)throw Error(x.error);return x}function msg(x){$('#message').textContent=x}function draw(){ctx.drawImage(img,0,0);if(box){ctx.strokeStyle='yellow';ctx.lineWidth=4;ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);z.width=450;z.height=Math.max(1,Math.round(450*(box[3]-box[1])/(box[2]-box[0])));zctx.drawImage(img,box[0],box[1],box[2]-box[0],box[3]-box[1],0,0,z.width,z.height)}}
function point(e){let r=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width,(e.clientX-r.left)*c.width/r.width)),Math.max(0,Math.min(c.height,(e.clientY-r.top)*c.height/r.height))]}c.onpointerdown=e=>{start=point(e);c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!start)return;let p=point(e);box=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];start=null;if(box[2]-box[0]<2||box[3]-box[1]<2)box=null;draw()};
function go(i){if(!Number.isInteger(i)||i<0||i>=info.count)return msg('Frame không hợp lệ'); n=i;box=null;comp=null;$('#frame').value=n;$('#comparison').textContent='';$('#judgment').hidden=true;z.width=0;img.onload=draw;img.src='/api/frame?n='+n;$('#status').textContent='Frame '+n+' / '+(info.count-1);msg('')}
async function post(path,data){return api(path,{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(data)})}async function save(){try{await post('/api/annotation',{frame:n,reviewer_id:$('#reviewer').value,evidence:$('#evidence').value,label:$('#label').value,box:$('#label').value==='TARGET'?box:null,notes:$('#notes').value});msg('Đã lưu nhãn độc lập. Có thể mở đối chiếu.')}catch(e){msg(e.message)}}
async function freeze(){try{await post('/api/freeze',{});msg('Đã khóa annotation. Có thể đối chiếu các frame đã lưu.')}catch(e){msg(e.message)}}async function compare(){try{comp=await api('/api/compare?n='+n);$('#comparison').textContent=JSON.stringify(comp,null,2);$('#judgment').hidden=false;draw();if(comp.replay){for(let t of comp.replay.tracks){ctx.strokeStyle='cyan';ctx.lineWidth=3;ctx.strokeRect(t.box[0],t.box[1],t.box[2]-t.box[0],t.box[3]-t.box[1]);ctx.fillStyle='cyan';ctx.font='24px sans-serif';ctx.fillText('track '+t.track_id,t.box[0],Math.max(24,t.box[1]))}if(comp.replay.output_box){let b=comp.replay.output_box;ctx.strokeStyle='red';ctx.lineWidth=5;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}}}catch(e){msg(e.message)}}async function saveComparison(){try{await post('/api/comparison',{frame:n,identity:$('#identity').value,notes:$('#compareNotes').value});msg('Đã lưu đối chiếu.')}catch(e){msg(e.message)}}
function toggleVideo(){let v=$('#vid');if(v.firstChild){v.replaceChildren();return}let x=document.createElement('video');x.controls=true;x.width=600;x.src='/api/video';v.append(x)}
api('/api/info').then(x=>{info=x;c.width=x.width;c.height=x.height;for(let f of x.queue){let o=document.createElement('option');o.value=f;o.textContent='Frame '+f;$('#queue').append(o)}go(x.queue[0])}).catch(e=>msg(e.message));
</script></html>'''

if __name__ == '__main__': main()
