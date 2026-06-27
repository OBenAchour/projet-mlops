from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import subprocess
import sys
import httpx
import threading
import collections
import time

from data_loader import prepare_data
from model_io import save_model

MODEL_PATH = "classifier.joblib"
app = FastAPI(title="Customer Churn MLOps Dashboard", version="3.0.0")

# Store logs in memory (last 200 lines)
log_buffer = collections.deque(maxlen=200)

def add_log(source, message, level="info"):
    log_buffer.append({
        "time": time.strftime("%H:%M:%S"),
        "source": source,
        "message": message,
        "level": level,
    })

try:
    model = joblib.load(MODEL_PATH)
    add_log("system", f"Modele charge depuis {MODEL_PATH}", "success")
except Exception as e:
    model = None
    add_log("system", f"Impossible de charger le modele: {e}", "error")

class CustomerInput(BaseModel):
    CreditScore: float
    Gender: int
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    class Config:
        json_schema_extra = {"example": {"CreditScore":619,"Gender":0,"Age":42,"Tenure":2,"Balance":0.0,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":101348.88}}

class RetrainInput(BaseModel):
    n_estimators: int = 100
    random_state: int = 42

class FlowInput(BaseModel):
    flow_name: str = "all"

class ServiceInput(BaseModel):
    service_name: str

@app.get("/")
def root():
    return {"message": "Customer Churn MLOps Dashboard", "status": "running"}

@app.post("/predict")
def predict(customer: CustomerInput):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    features = np.array([[customer.CreditScore, customer.Gender, customer.Age, customer.Tenure, customer.Balance, customer.NumOfProducts, customer.HasCrCard, customer.IsActiveMember, customer.EstimatedSalary]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    result = {"prediction": int(prediction[0]), "churn": bool(prediction[0]==1), "probability": {"no_churn": round(float(probability[0][0]),4), "churn": round(float(probability[0][1]),4)}}
    add_log("predict", f"Prediction: {'CHURN' if result['churn'] else 'NO CHURN'} (churn={result['probability']['churn']})", "success")
    return result

@app.post("/retrain")
def retrain(params: RetrainInput):
    global model
    add_log("retrain", f"Reentrainement n_estimators={params.n_estimators}", "info")
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=params.n_estimators, random_state=params.random_state)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    save_model(model, MODEL_PATH)
    add_log("retrain", f"Accuracy: {round(accuracy*100,2)}%", "success")
    return {"message": "Modele reentrainer avec succes", "accuracy": round(accuracy*100,2), "params": {"n_estimators": params.n_estimators, "random_state": params.random_state}}

def run_flow_background(flow_name):
    try:
        add_log("prefect", f"Demarrage du flow '{flow_name}'...", "info")
        result = subprocess.run(
            [sys.executable, "pipeline_prefect.py", "--flow", flow_name],
            capture_output=True, text=True, timeout=600
        )
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                add_log("prefect", line.strip(), "info")
        if result.returncode == 0:
            add_log("prefect", f"Flow '{flow_name}' termine avec succes", "success")
        else:
            for line in result.stderr.strip().split("\n")[-5:]:
                if line.strip():
                    add_log("prefect", line.strip(), "error")
            add_log("prefect", f"Flow '{flow_name}' echoue", "error")
    except subprocess.TimeoutExpired:
        add_log("prefect", f"Flow '{flow_name}' timeout (10min)", "error")
    except Exception as e:
        add_log("prefect", f"Erreur: {str(e)}", "error")

@app.post("/api/run-flow")
def run_flow(flow_input: FlowInput):
    valid = ["all","install","code","test","prepare","train","evaluate","predict","clean"]
    if flow_input.flow_name not in valid:
        raise HTTPException(status_code=400, detail=f"Flow inconnu. Valides: {valid}")
    t = threading.Thread(target=run_flow_background, args=(flow_input.flow_name,), daemon=True)
    t.start()
    return {"status": "started", "flow": flow_input.flow_name}

@app.post("/api/start-service")
def start_service(svc: ServiceInput):
    commands = {
        "prefect": ["prefect", "server", "start"],
        "mlflow": ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlartifacts", "--host", "0.0.0.0", "--port", "5000"],
        "elasticsearch": ["/usr/bin/docker-compose", "up", "-d", "elasticsearch"],
        "kibana": ["/usr/bin/docker-compose", "up", "-d", "kibana"],
    }
    if svc.service_name not in commands:
        raise HTTPException(status_code=400, detail=f"Service inconnu: {svc.service_name}")
    try:
        add_log("service", f"Demarrage de {svc.service_name}...", "info")
        subprocess.Popen(commands[svc.service_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        add_log("service", f"{svc.service_name} demarre", "success")
        return {"status": "starting", "service": svc.service_name}
    except Exception as e:
        add_log("service", f"Erreur demarrage {svc.service_name}: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-service")
def stop_service(svc: ServiceInput):
    commands = {
        "elasticsearch": ["/usr/bin/docker-compose", "stop", "elasticsearch"],
        "kibana": ["/usr/bin/docker-compose", "stop", "kibana"],
    }
    if svc.service_name not in commands:
        return {"status": "manual", "message": f"Arretez {svc.service_name} manuellement (Ctrl+C)"}
    try:
        add_log("service", f"Arret de {svc.service_name}...", "info")
        subprocess.Popen(commands[svc.service_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        add_log("service", f"{svc.service_name} arrete", "success")
        return {"status": "stopping", "service": svc.service_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services-status")
async def services_status():
    services = {}
    checks = [("prefect","http://127.0.0.1:4200/api/health","http://localhost:4200"),("mlflow","http://127.0.0.1:5000/","http://localhost:5000"),("elasticsearch","http://127.0.0.1:9200","http://localhost:9200"),("kibana","http://127.0.0.1:5601/api/status","http://localhost:5601")]
    for name, url, display in checks:
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                await c.get(url)
                services[name] = {"status":"running","url":display}
        except Exception:
            services[name] = {"status":"stopped","url":display}
    services["fastapi"] = {"status":"running","url":"http://localhost:8000/docs"}
    return services

@app.get("/api/metrics")
def get_metrics():
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(["http://localhost:9200"])
        r = es.search(index="mlflow-metrics", body={"query":{"match_all":{}},"sort":[{"timestamp":"desc"}],"size":10})
        return {"count":len(r["hits"]["hits"]), "metrics":[h["_source"] for h in r["hits"]["hits"]]}
    except Exception as e:
        return {"count":0,"metrics":[],"error":str(e)}

@app.get("/api/logs")
def get_logs():
    return {"logs": list(log_buffer)}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLOps Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:linear-gradient(135deg,#1e293b,#334155);padding:20px 30px;border-bottom:1px solid #475569;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:22px;color:#60a5fa}
.header p{color:#94a3b8;font-size:13px;margin-top:4px}
.header-right{display:flex;gap:8px}
.container{max-width:1400px;margin:0 auto;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-bottom:16px}
.card{background:#1e293b;border-radius:12px;padding:18px;border:1px solid #334155}
.card h2{font-size:15px;color:#60a5fa;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.service-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #334155}
.service-row:last-child{border-bottom:none}
.status{padding:3px 8px;border-radius:20px;font-size:11px;font-weight:600}
.status.running{background:#065f46;color:#34d399}
.status.stopped{background:#7f1d1d;color:#fca5a5}
.svc-actions{display:flex;align-items:center;gap:6px}
.svc-btn{padding:3px 8px;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:600}
.svc-start{background:#2563eb;color:white}
.svc-start:hover{background:#1d4ed8}
.svc-stop{background:#dc2626;color:white}
.svc-stop:hover{background:#b91c1c}
.svc-link{color:#60a5fa;text-decoration:none;font-size:12px}
.btn{padding:9px 14px;border:none;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600;transition:all 0.2s;width:100%;margin-bottom:6px}
.btn-sm{width:auto;padding:6px 12px;margin:0}
.btn-blue{background:#2563eb;color:white}.btn-blue:hover{background:#1d4ed8}
.btn-green{background:#059669;color:white}.btn-green:hover{background:#047857}
.btn-orange{background:#d97706;color:white}.btn-orange:hover{background:#b45309}
.btn-red{background:#dc2626;color:white}.btn-red:hover{background:#b91c1c}
.btn-purple{background:#7c3aed;color:white}.btn-purple:hover{background:#6d28d9}
.btn-gray{background:#475569;color:white}.btn-gray:hover{background:#374151}
.btn-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.form-group{display:flex;flex-direction:column}
.form-group label{font-size:11px;color:#94a3b8;margin-bottom:2px}
.form-group input,.form-group select{background:#0f172a;border:1px solid #475569;color:#e2e8f0;padding:7px;border-radius:6px;font-size:12px}
.log-box{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px;font-family:'Fira Code',monospace;font-size:11px;max-height:300px;overflow-y:auto;color:#94a3b8}
.log-line{padding:2px 0;border-bottom:1px solid #1e293b;display:flex;gap:8px}
.log-time{color:#475569;min-width:55px}
.log-src{font-weight:600;min-width:65px}
.log-src.prefect{color:#818cf8}.log-src.service{color:#f59e0b}.log-src.predict{color:#34d399}.log-src.retrain{color:#fb923c}.log-src.system{color:#94a3b8}
.log-msg{flex:1}
.log-msg.success{color:#34d399}.log-msg.error{color:#fca5a5}.log-msg.info{color:#e2e8f0}
.metrics-table{width:100%;border-collapse:collapse;font-size:12px}
.metrics-table th{text-align:left;padding:6px;color:#94a3b8;border-bottom:1px solid #475569}
.metrics-table td{padding:6px;border-bottom:1px solid #334155}
.result-box{background:#0f172a;border-radius:8px;padding:12px;margin-top:8px;border:1px solid #334155}
.result-churn{color:#fca5a5;font-weight:bold;font-size:16px}
.result-no-churn{color:#34d399;font-weight:bold;font-size:16px}
.links-bar{display:flex;gap:8px;flex-wrap:wrap}
.link-btn{display:inline-flex;align-items:center;gap:5px;padding:6px 12px;background:#334155;color:#e2e8f0;text-decoration:none;border-radius:6px;font-size:12px;transition:background 0.2s}
.link-btn:hover{background:#475569}
.wide{grid-column:span 2}
.tabs{display:flex;gap:4px;margin-bottom:10px}
.tab{padding:6px 12px;background:#334155;border:none;color:#94a3b8;border-radius:6px 6px 0 0;cursor:pointer;font-size:12px}
.tab.active{background:#0f172a;color:#60a5fa}
@media(max-width:700px){.wide{grid-column:span 1}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
<div>
<h1>&#x1F680; MLOps Dashboard</h1>
<p>Prefect &#x2022; MLflow &#x2022; FastAPI &#x2022; Docker &#x2022; Elasticsearch &#x2022; Kibana</p>
</div>
<div class="header-right">
<button class="btn btn-sm btn-gray" onclick="checkServices();loadLogs();loadMetrics()">&#x1F504; Tout rafraichir</button>
</div>
</div>
<div class="container">

<div class="card" style="margin-bottom:16px">
<h2>&#x1F517; Acces rapide</h2>
<div class="links-bar">
<a href="http://localhost:4200" target="_blank" class="link-btn">&#x1F4CB; Prefect</a>
<a href="http://localhost:5000" target="_blank" class="link-btn">&#x1F4CA; MLflow</a>
<a href="http://localhost:8000/docs" target="_blank" class="link-btn">&#x26A1; Swagger</a>
<a href="http://localhost:5601" target="_blank" class="link-btn">&#x1F4C8; Kibana</a>
<a href="http://localhost:9200" target="_blank" class="link-btn">&#x1F50D; Elastic</a>
<a href="http://localhost:5601/app/discover" target="_blank" class="link-btn">&#x1F50E; Discover</a>
<a href="http://localhost:5601/app/management/kibana/indexPatterns" target="_blank" class="link-btn">&#x2699; Index</a>
</div>
</div>

<div class="grid">
<div class="card">
<h2>&#x1F7E2; Services</h2>
<div id="services">Chargement...</div>
<button class="btn btn-gray" style="margin-top:10px" onclick="checkServices()">&#x1F504; Rafraichir</button>
</div>

<div class="card">
<h2>&#x2699; Pipeline Prefect</h2>
<div class="btn-row">
<button class="btn btn-blue" onclick="runFlow('all')">&#x25B6; Complet</button>
<button class="btn btn-green" onclick="runFlow('train')">&#x1F680; Train</button>
<button class="btn btn-green" onclick="runFlow('evaluate')">&#x1F4C8; Evaluate</button>
<button class="btn btn-green" onclick="runFlow('predict')">&#x1F52E; Predict</button>
<button class="btn btn-orange" onclick="runFlow('code')">&#x1F3A8; Code</button>
<button class="btn btn-orange" onclick="runFlow('test')">&#x1F9EA; Tests</button>
<button class="btn btn-purple" onclick="runFlow('install')">&#x1F4E6; Install</button>
<button class="btn btn-red" onclick="runFlow('clean')">&#x1F9F9; Clean</button>
</div>
</div>

<div class="card">
<h2>&#x1F52E; Prediction</h2>
<div class="form-grid">
<div class="form-group"><label>CreditScore</label><input type="number" id="CreditScore" value="619"></div>
<div class="form-group"><label>Gender</label><select id="Gender"><option value="0">Female</option><option value="1">Male</option></select></div>
<div class="form-group"><label>Age</label><input type="number" id="Age" value="42"></div>
<div class="form-group"><label>Tenure</label><input type="number" id="Tenure" value="2"></div>
<div class="form-group"><label>Balance</label><input type="number" id="Balance" value="0" step="0.01"></div>
<div class="form-group"><label>NumOfProducts</label><input type="number" id="NumOfProducts" value="1"></div>
<div class="form-group"><label>HasCrCard</label><select id="HasCrCard"><option value="1">Oui</option><option value="0">Non</option></select></div>
<div class="form-group"><label>IsActiveMember</label><select id="IsActiveMember"><option value="1">Oui</option><option value="0">Non</option></select></div>
<div class="form-group" style="grid-column:span 2"><label>EstimatedSalary</label><input type="number" id="EstimatedSalary" value="101348.88" step="0.01"></div>
</div>
<button class="btn btn-blue" style="margin-top:8px" onclick="predict()">&#x1F52E; Predire</button>
<div id="predict-result"></div>
</div>

<div class="card">
<h2>&#x1F504; Reentrainement</h2>
<div class="form-grid">
<div class="form-group"><label>n_estimators</label><input type="number" id="n_estimators" value="100"></div>
<div class="form-group"><label>random_state</label><input type="number" id="random_state" value="42"></div>
</div>
<button class="btn btn-orange" style="margin-top:8px" onclick="retrainModel()">&#x1F504; Reentrainer</button>
<div id="retrain-result"></div>
</div>

<div class="card wide">
<h2>&#x1F4CB; Logs en temps reel</h2>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
<div class="tabs">
<button class="tab active" onclick="logFilter='all';renderLogs();setActiveTab(this)">Tout</button>
<button class="tab" onclick="logFilter='prefect';renderLogs();setActiveTab(this)">Prefect</button>
<button class="tab" onclick="logFilter='service';renderLogs();setActiveTab(this)">Services</button>
<button class="tab" onclick="logFilter='predict';renderLogs();setActiveTab(this)">Predict</button>
<button class="tab" onclick="logFilter='retrain';renderLogs();setActiveTab(this)">Retrain</button>
</div>
<div>
<button class="btn btn-sm btn-gray" onclick="loadLogs()">&#x1F504;</button>
<button class="btn btn-sm btn-red" onclick="clearLogs()">&#x1F5D1; Vider</button>
</div>
</div>
<div id="log-container" class="log-box" style="min-height:120px"></div>
</div>

<div class="card wide">
<h2>&#x1F4CA; Metriques (Elasticsearch)</h2>
<button class="btn btn-sm btn-gray" onclick="loadMetrics()" style="margin-bottom:8px">&#x1F504; Rafraichir</button>
<div id="metrics-container">Chargement...</div>
</div>
</div>
</div>

<script>
let logFilter = 'all';
let logsData = [];

function setActiveTab(el) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
}

async function checkServices() {
    try {
        const r = await fetch('/api/services-status');
        const d = await r.json();
        const icons = {prefect:'&#x1F4CB;',mlflow:'&#x1F4CA;',elasticsearch:'&#x1F50D;',kibana:'&#x1F4C8;',fastapi:'&#x26A1;'};
        let h = '';
        for (const [n, i] of Object.entries(d)) {
            const canStart = n !== 'fastapi' && i.status === 'stopped';
            const canStop = (n === 'elasticsearch' || n === 'kibana') && i.status === 'running';
            h += '<div class="service-row"><span>' + icons[n] + ' <b>' + n + '</b></span>';
            h += '<div class="svc-actions">';
            h += '<span class="status ' + i.status + '">' + i.status + '</span>';
            if (canStart) h += '<button class="svc-btn svc-start" onclick="startSvc(\\'' + n + '\\')">Demarrer</button>';
            if (canStop) h += '<button class="svc-btn svc-stop" onclick="stopSvc(\\'' + n + '\\')">Stop</button>';
            h += '<a href="' + i.url + '" target="_blank" class="svc-link">Ouvrir</a>';
            h += '</div></div>';
        }
        document.getElementById('services').innerHTML = h;
    } catch(e) {
        document.getElementById('services').innerHTML = '<span class="log-msg error">Erreur connexion</span>';
    }
}

async function startSvc(name) {
    try {
        await fetch('/api/start-service', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({service_name:name})});
        setTimeout(checkServices, 3000);
        setTimeout(loadLogs, 1000);
    } catch(e) {}
}

async function stopSvc(name) {
    try {
        await fetch('/api/stop-service', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({service_name:name})});
        setTimeout(checkServices, 2000);
        setTimeout(loadLogs, 1000);
    } catch(e) {}
}

async function runFlow(name) {
    try {
        await fetch('/api/run-flow', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({flow_name:name})});
        setTimeout(loadLogs, 1000);
        // Auto-refresh logs during flow execution
        let count = 0;
        const iv = setInterval(() => { loadLogs(); count++; if(count>60)clearInterval(iv); }, 5000);
    } catch(e) {}
}

async function predict() {
    const f = ['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'];
    const b = {};
    f.forEach(x => { b[x] = parseFloat(document.getElementById(x).value); });
    try {
        const r = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
        const d = await r.json();
        const c = d.churn ? 'result-churn' : 'result-no-churn';
        const l = d.churn ? 'CHURN' : 'PAS DE CHURN';
        document.getElementById('predict-result').innerHTML = '<div class="result-box"><div class="'+c+'">'+l+'</div><div style="margin-top:6px">Churn: <b>'+(d.probability.churn*100).toFixed(1)+'%</b> | No churn: <b>'+(d.probability.no_churn*100).toFixed(1)+'%</b></div></div>';
        setTimeout(loadLogs, 500);
    } catch(e) {
        document.getElementById('predict-result').innerHTML = '<div class="result-box log-msg error">'+e.message+'</div>';
    }
}

async function retrainModel() {
    document.getElementById('retrain-result').innerHTML = '<div class="result-box"><span class="log-msg info">En cours...</span></div>';
    try {
        const r = await fetch('/retrain', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({n_estimators:parseInt(document.getElementById('n_estimators').value),random_state:parseInt(document.getElementById('random_state').value)})});
        const d = await r.json();
        document.getElementById('retrain-result').innerHTML = '<div class="result-box"><div class="log-msg success" style="font-size:14px">'+d.message+'</div><div style="margin-top:6px">Accuracy: <b>'+d.accuracy+'%</b></div></div>';
        setTimeout(loadLogs, 500);
    } catch(e) {
        document.getElementById('retrain-result').innerHTML = '<div class="result-box log-msg error">'+e.message+'</div>';
    }
}

async function loadLogs() {
    try {
        const r = await fetch('/api/logs');
        const d = await r.json();
        logsData = d.logs;
        renderLogs();
    } catch(e) {}
}

function renderLogs() {
    const filtered = logFilter === 'all' ? logsData : logsData.filter(l => l.source === logFilter);
    if (filtered.length === 0) {
        document.getElementById('log-container').innerHTML = '<div style="color:#475569;padding:20px;text-align:center">Aucun log</div>';
        return;
    }
    let h = '';
    filtered.forEach(l => {
        h += '<div class="log-line"><span class="log-time">'+l.time+'</span><span class="log-src '+l.source+'">'+l.source+'</span><span class="log-msg '+l.level+'">'+l.message+'</span></div>';
    });
    const el = document.getElementById('log-container');
    el.innerHTML = h;
    el.scrollTop = el.scrollHeight;
}

async function clearLogs() {
    logsData = [];
    renderLogs();
}

async function loadMetrics() {
    try {
        const r = await fetch('/api/metrics');
        const d = await r.json();
        if (d.count === 0) {
            document.getElementById('metrics-container').innerHTML = '<span style="color:#475569">Aucune metrique dans Elasticsearch.</span>';
            return;
        }
        let h = '<table class="metrics-table"><tr><th>Run</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Timestamp</th></tr>';
        d.metrics.forEach(m => {
            h += '<tr><td>'+m.run_name+'</td>';
            h += '<td>'+(m.metrics&&m.metrics.accuracy?(m.metrics.accuracy*100).toFixed(1)+'%':'-')+'</td>';
            h += '<td>'+(m.metrics&&m.metrics.precision?(m.metrics.precision*100).toFixed(1)+'%':'-')+'</td>';
            h += '<td>'+(m.metrics&&m.metrics.recall?(m.metrics.recall*100).toFixed(1)+'%':'-')+'</td>';
            h += '<td>'+(m.metrics&&m.metrics.f1_score?(m.metrics.f1_score*100).toFixed(1)+'%':'-')+'</td>';
            h += '<td style="color:#94a3b8;font-size:11px">'+(m.timestamp||'-')+'</td></tr>';
        });
        h += '</table>';
        document.getElementById('metrics-container').innerHTML = h;
    } catch(e) {
        document.getElementById('metrics-container').innerHTML = '<span class="log-msg error">'+e.message+'</span>';
    }
}

checkServices(); loadLogs(); loadMetrics();
setInterval(checkServices, 30000);
setInterval(loadLogs, 10000);
</script>
</body>
</html>"""
