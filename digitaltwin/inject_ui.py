import re

with open('digitaltwin/temple_simulation (3).html', 'r', encoding='utf-8') as f:
    content = f.read()

heatmap_class = """
// ==================== TempleHeatmap ====================
class TempleHeatmap {
    constructor(canvasId, statusId) {
        this.canvas = document.getElementById(canvasId);
        this.status = document.getElementById(statusId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.bounds = { minX: -120, maxX: 40, minZ: -60, maxZ: 110 };
    }

    render(people) {
        if (!this.ctx || !this.canvas) return;

        const ctx = this.ctx;
        const { width, height } = this.canvas;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, width, height);

        this._drawGroundPlan();
        this._drawPeople(people);

        if (this.status) {
            this.status.textContent = `Live density view · ${people.length} people`;
        }
    }

    _layout() {
        const pad = 14;
        const canvasW = this.canvas.width - pad * 2;
        const canvasH = this.canvas.height - pad * 2;
        const worldW = this.bounds.maxX - this.bounds.minX;
        const worldD = this.bounds.maxZ - this.bounds.minZ;
        const scale = Math.min(canvasW / worldW, canvasH / worldD) * 1.5;
        const width = worldW * scale;
        const height = worldD * scale;
        return {
            left: (this.canvas.width - width) / 2 + 30, // shift right
            top: (this.canvas.height - height) / 2 + 10,
            width, height, worldW, worldD, scale
        };
    }

    _project(x, z) {
        const layout = this._layout();
        return {
            x: layout.left + ((x - this.bounds.minX) / layout.worldW) * layout.width,
            y: layout.top + layout.height - (((z - this.bounds.minZ) / layout.worldD) * layout.height),
        };
    }

    _drawGroundPlan() {
        const ctx = this.ctx;
        const layout = this._layout();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.strokeRect(layout.left, layout.top, layout.width, layout.height);
        
        // Temple Path Drawing
        const zones = [
            { id: 'Entry', x: -10, z: 80, w: 20, d: 25 },
            { id: 'Queue', x: -25, z: 15, w: 50, d: 65 },
            { id: 'Courtyard', x: -40, z: -10, w: 50, d: 25 },
            { id: 'Temple', x: -15, z: -45, w: 30, d: 35 },
            { id: 'Exit', x: -120, z: -55, w: 80, d: 20 },
        ];
        
        zones.forEach(z => {
            const tl = this._project(z.x, z.z + z.d);
            const br = this._project(z.x + z.w, z.z);
            ctx.fillStyle = 'rgba(200, 160, 80, 0.05)';
            ctx.strokeStyle = 'rgba(200, 160, 80, 0.2)';
            ctx.fillRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
            ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
            
            ctx.fillStyle = 'rgba(200, 160, 80, 0.6)';
            ctx.font = '10px Arial';
            ctx.fillText(z.id, tl.x + 4, tl.y + 12);
        });
    }

    _drawPeople(people) {
        const ctx = this.ctx;
        const radius = 2;

        people.forEach((person) => {
            const pos = this._project(person.mesh.position.x, person.mesh.position.z);
            let color = 'rgba(255, 255, 255, 0.7)';
            if (person.gender === 'male') color = 'rgba(96, 165, 250, 0.8)';
            else if (person.gender === 'female') color = 'rgba(244, 114, 182, 0.8)';

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        });
    }
}
"""

app_update_logic = """
  _updatePredictionPanel() {
    // 1. STATS PANEL (Control Panel equivalent)
    const people = this.crowdManager.agents.length;
    const statTotal = document.getElementById('stat-total');
    if (statTotal) statTotal.textContent = people;
    
    // FPS Calculation
    if (!this._frameCount) this._frameCount = 0;
    if (!this._lastFpsTime) this._lastFpsTime = Date.now();
    this._frameCount++;
    const now = Date.now();
    if (now - this._lastFpsTime >= 1000) {
        const fpsEl = document.getElementById('stat-fps');
        if (fpsEl) fpsEl.textContent = this._frameCount;
        this._frameCount = 0;
        this._lastFpsTime = now;
    }

    // Update Heatmap
    if (!this.heatmap) {
        this.heatmap = new TempleHeatmap('heatmap-canvas', 'heatmap-status');
    }
    this.heatmap.render(this.crowdManager.agents);

    // 2. ZONE STATUS PANEL (Stadium equivalent)
    const zoneList = document.getElementById('zone-list');
    if (zoneList) {
        let html = '';
        
        let d = {male:0, female:0};
        let stampede = {active:false};
        
        if (this._predictionState) {
            d = this._predictionState.demographics || d;
            stampede = this._predictionState.stampede || stampede;
            
            // Inject AI real world alerts
            html += `<div style="background: rgba(255,40,40,0.1); padding: 8px; border: 1px solid rgba(255,40,40,0.3); border-radius: 4px; margin-bottom: 15px;">
                <h3 style="color:#ff6666; font-size:12px; margin-bottom:5px;">🚨 REAL WORLD AI ALERTS</h3>
                <p style="color:#ddd; font-size:11px; margin-bottom:3px;">Male: ${d.male || 0} | Female: ${d.female || 0}</p>
                <p style="color:#ddd; font-size:11px;">Active Stampede: <b style="color:${stampede.active ? '#ff3333' : '#66cc66'}">${stampede.active ? 'YES' : 'NO'}</b> (${stampede.latestFallenCount || 0} fallen)</p>
            </div>`;
            
            // Auto trigger panic
            if (stampede.active && !window._stampedeActive) {
                this.crowdManager.triggerStampede();
                window._stampedeActive = true;
            } else if (!stampede.active && window._stampedeActive) {
                this.crowdManager._rushMode = false;
                window._stampedeActive = false;
            }
        }

        // Aggregate densities
        const counts = { Entry:0, Queue:0, Courtyard:0, Temple:0, Exit:0 };
        this.crowdManager.agents.forEach(a => {
           let z = a.zoneTag;
           if(z==='entry') counts.Entry++;
           if(z==='queue') counts.Queue++;
           if(z==='courtyard') counts.Courtyard++;
           if(z==='temple') counts.Temple++;
           if(z==='exit') counts.Exit++;
        });
        
        let avgDens = 0; let activeZ = 0;
        
        for (const [zoneId, c] of Object.entries(counts)) {
            if (c > 0) {
                const denPercent = Math.min(100, Math.round((c / 200) * 100)); // normalized to 200 max per zone
                avgDens += denPercent; activeZ++;
                const dc = denPercent > 70 ? '#ef4444' : denPercent > 40 ? '#f59e0b' : '#22c55e';
                html += `<div class="zone-item" style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:11px; background:rgba(0,0,0,0.3); padding:4px; border-radius:3px;">
                    <span class="zone-label" style="color:#ccc;">${zoneId}</span>
                    <span class="zone-count" style="color:#fff; font-weight:bold;">${c}</span>
                    <span class="zone-density" style="background:${dc}22;color:${dc}; padding: 1px 4px; border-radius: 2px;">${denPercent}%</span>
                </div>`;
            }
        }
        
        zoneList.innerHTML = html || '<div style="color:#6b8a7a;padding:6px;">No active zones</div>';
        
        const avg = activeZ > 0 ? Math.round(avgDens / activeZ) : 0;
        const statDen = document.getElementById('stat-density');
        if (statDen) statDen.textContent = avg + '%';
        const statZone = document.getElementById('stat-zones');
        if (statZone) statZone.textContent = activeZ;
        
        const dFill = document.getElementById('density-fill');
        if (dFill) {
            dFill.style.width = avg + '%';
        }
    }
  }
"""

content = re.sub(r'  _updatePredictionPanel\(\) \{.*?\n  \}', app_update_logic, content, flags=re.DOTALL)

content = content.replace('// ── Bootstrap ────────────────────────────────────────────────────────', heatmap_class + '\n\n// ── Bootstrap ────────────────────────────────────────────────────────')

with open('digitaltwin/temple_simulation (3).html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Injected Heatmap and UI logic')
