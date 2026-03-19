// ============================================================
//  FaceAttend — All-in-One Backend
//  Express + SQLite + face-api.js (FaceNet 128-D)
//
//  Deploy free on Render.com:
//    1. Push this folder to GitHub
//    2. New Web Service on render.com → connect repo
//    3. Build command: npm install
//    4. Start command: node server.js
//    Done — get your public URL
//
//  All data stored in SQLite (file on disk).
//  On Render free tier, disk resets on redeploy —
//  use Render's persistent disk ($1/mo) or Railway for persistence.
// ============================================================

const express   = require('express');
const cors      = require('cors');
const multer    = require('multer');
const canvas    = require('canvas');
const Database  = require('better-sqlite3');
const { v4: uuidv4 } = require('uuid');
const path      = require('path');
const fs        = require('fs');

// ── face-api.js setup ─────────────────────────────────────
const faceapi = require('@vladmandic/face-api/dist/face-api.node.js');
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app    = express();
const PORT   = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

app.use(cors());
app.use(express.json());

// ── SQLite database setup ─────────────────────────────────
const DB_PATH = process.env.DB_PATH || path.join(__dirname, 'data', 'faceattend.db');
fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
const db = new Database(DB_PATH);

// Enable WAL mode for better performance
db.pragma('journal_mode = WAL');

// Create tables
db.exec(`
  CREATE TABLE IF NOT EXISTS instances (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    secret_code TEXT UNIQUE NOT NULL,
    is_active   INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS tokens (
    id          TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL,
    device_name TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (instance_id) REFERENCES instances(id)
  );

  CREATE TABLE IF NOT EXISTS employees (
    id              TEXT PRIMARY KEY,
    instance_id     TEXT NOT NULL,
    employee_code   TEXT,
    name            TEXT NOT NULL,
    department      TEXT,
    photo_url       TEXT,
    face_embedding  TEXT,
    face_enrolled_at TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (instance_id) REFERENCES instances(id)
  );

  CREATE TABLE IF NOT EXISTS attendance_logs (
    id            TEXT PRIMARY KEY,
    instance_id   TEXT NOT NULL,
    employee_id   TEXT NOT NULL,
    employee_name TEXT,
    department    TEXT,
    punched_at    TEXT NOT NULL,
    date          TEXT NOT NULL,
    confidence    REAL,
    face_distance REAL,
    source        TEXT DEFAULT 'mobile',
    created_at    TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (instance_id) REFERENCES instances(id),
    FOREIGN KEY (employee_id) REFERENCES employees(id)
  );
`);

console.log('SQLite database ready at:', DB_PATH);

// ── Seed a default instance if none exists ────────────────
const instanceCount = db.prepare('SELECT COUNT(*) as c FROM instances').get();
if (instanceCount.c === 0) {
  db.prepare(`INSERT INTO instances (id, name, secret_code) VALUES (?, ?, ?)`)
    .run(uuidv4(), 'Default Office', 'DEMO1234');
  console.log('Created default instance — secret code: DEMO1234');
}

// ── Auth middleware ───────────────────────────────────────
function requireAuth(req, res, next) {
  const auth = req.headers.authorization || '';
  const token = auth.replace('Bearer ', '').trim();
  if (!token) return res.status(401).json({ success: false, message: 'No token provided' });

  const row = db.prepare('SELECT * FROM tokens WHERE id = ?').get(token);
  if (!row) return res.status(401).json({ success: false, message: 'Invalid or expired token' });

  const instance = db.prepare('SELECT * FROM instances WHERE id = ? AND is_active = 1').get(row.instance_id);
  if (!instance) return res.status(401).json({ success: false, message: 'Instance not found or inactive' });

  req.instance = instance;
  next();
}

// ── Face recognition helpers ──────────────────────────────
let modelsLoaded = false;
const MODEL_PATH = path.join(__dirname, 'models');

async function loadModels() {
  // On Render.com, models are downloaded at build time via render-build.sh
  // Locally, run: node download-models.js
  if (!fs.existsSync(MODEL_PATH)) {
    console.warn('⚠ Models folder not found. Face recognition will not work.');
    console.warn('  Run: node download-models.js');
    return;
  }
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
    modelsLoaded = true;
    console.log('✓ Face recognition models loaded');
  } catch (err) {
    console.error('Failed to load models:', err.message);
  }
}

async function extractDescriptor(buffer) {
  if (!modelsLoaded) throw new Error('Models not loaded');
  const img       = await canvas.loadImage(buffer);
  const detection = await faceapi
    .detectSingleFace(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!detection) return null;
  return {
    descriptor: Array.from(detection.descriptor),
    confidence: detection.detection.score,
  };
}

function euclideanDistance(a, b) {
  if (!a || !b || a.length !== b.length) return 1.0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; sum += d * d; }
  return Math.sqrt(sum);
}

function findBestMatch(descriptor, employees) {
  const THRESHOLD = 0.50;
  const MARGIN    = 0.08;
  let best = Infinity, second = Infinity, bestEmp = null;

  for (const emp of employees) {
    if (!emp.face_embedding) continue;
    let stored;
    try { stored = JSON.parse(emp.face_embedding); } catch { continue; }
    if (!Array.isArray(stored)) continue;
    const dist = euclideanDistance(descriptor, stored);
    if (dist < best) { second = best; best = dist; bestEmp = emp; }
    else if (dist < second) { second = dist; }
  }

  if (!bestEmp || best > THRESHOLD) return null;
  if (employees.length > 1 && (second - best) < MARGIN) return null;

  return {
    employee:   bestEmp,
    distance:   +best.toFixed(4),
    confidence: Math.round(Math.max(0, 1 - best / THRESHOLD) * 100),
  };
}

// ════════════════════════════════════════════════════════════
//  ROUTES
// ════════════════════════════════════════════════════════════

// GET / — health check
app.get('/', (req, res) => {
  res.json({
    service:     'FaceAttend Backend',
    status:      'running',
    modelsLoaded,
    timestamp:   new Date().toISOString(),
  });
});

// GET /health
app.get('/health', (req, res) => {
  res.json({ status: 'ok', modelsLoaded });
});

// ── AUTH ──────────────────────────────────────────────────

// POST /api/login
// Body: { secret_code, device_name }
app.post('/api/login', (req, res) => {
  const { secret_code, device_name } = req.body;
  if (!secret_code) return res.status(400).json({ success: false, message: 'secret_code required' });

  const instance = db.prepare('SELECT * FROM instances WHERE secret_code = ? AND is_active = 1').get(secret_code);
  if (!instance) return res.status(401).json({ success: false, message: 'Invalid secret code' });

  const tokenId = uuidv4();
  db.prepare('INSERT INTO tokens (id, instance_id, device_name) VALUES (?, ?, ?)').run(tokenId, instance.id, device_name || 'unknown');

  res.json({
    success:       true,
    token:         tokenId,
    instance_id:   instance.id,
    instance_name: instance.name,
  });
});

// POST /api/logout
app.post('/api/logout', requireAuth, (req, res) => {
  const token = req.headers.authorization.replace('Bearer ', '').trim();
  db.prepare('DELETE FROM tokens WHERE id = ?').run(token);
  res.json({ success: true });
});

// ── EMPLOYEES ─────────────────────────────────────────────

// GET /api/employees
app.get('/api/employees', requireAuth, (req, res) => {
  const employees = db.prepare(
    'SELECT id, employee_code, name, department, photo_url, face_embedding, face_enrolled_at FROM employees WHERE instance_id = ? ORDER BY name'
  ).all(req.instance.id);
  res.json({ success: true, employees });
});

// POST /api/employees — create employee
app.post('/api/employees', requireAuth, (req, res) => {
  const { name, employee_code, department } = req.body;
  if (!name) return res.status(400).json({ success: false, message: 'name is required' });
  const id = uuidv4();
  db.prepare('INSERT INTO employees (id, instance_id, name, employee_code, department) VALUES (?, ?, ?, ?, ?)')
    .run(id, req.instance.id, name, employee_code || null, department || null);
  const emp = db.prepare('SELECT * FROM employees WHERE id = ?').get(id);
  res.json({ success: true, employee: emp });
});

// POST /api/employees/:id/enroll — enroll face
app.post('/api/employees/:id/enroll', requireAuth, upload.single('photo'), async (req, res) => {
  const emp = db.prepare('SELECT * FROM employees WHERE id = ? AND instance_id = ?').get(req.params.id, req.instance.id);
  if (!emp) return res.status(404).json({ success: false, message: 'Employee not found' });
  if (!req.file) return res.status(400).json({ success: false, message: 'No photo uploaded' });
  if (!modelsLoaded) return res.status(503).json({ success: false, message: 'Face models not loaded yet. Try again in 30 seconds.' });

  try {
    const result = await extractDescriptor(req.file.buffer);
    if (!result) return res.status(422).json({ success: false, message: 'No face detected. Use a clear, well-lit, front-facing photo.' });

    db.prepare('UPDATE employees SET face_embedding = ?, face_enrolled_at = ? WHERE id = ?')
      .run(JSON.stringify(result.descriptor), new Date().toISOString(), emp.id);

    res.json({ success: true, message: 'Face enrolled successfully', confidence: result.confidence });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// DELETE /api/employees/:id/face — remove face
app.delete('/api/employees/:id/face', requireAuth, (req, res) => {
  const emp = db.prepare('SELECT * FROM employees WHERE id = ? AND instance_id = ?').get(req.params.id, req.instance.id);
  if (!emp) return res.status(404).json({ success: false, message: 'Employee not found' });
  db.prepare('UPDATE employees SET face_embedding = NULL, face_enrolled_at = NULL WHERE id = ?').run(emp.id);
  res.json({ success: true, message: 'Face enrollment removed' });
});

// ── ATTENDANCE ────────────────────────────────────────────

// POST /api/punch — mark attendance via face recognition
app.post('/api/punch', requireAuth, upload.single('photo'), async (req, res) => {
  if (!req.file) return res.status(400).json({ success: false, message: 'No photo uploaded' });
  if (!modelsLoaded) return res.status(503).json({ success: false, message: 'Face models not loaded yet. Try again in 30 seconds.' });

  try {
    // Get all enrolled employees for this instance
    const employees = db.prepare(
      'SELECT id, name, department, employee_code, face_embedding FROM employees WHERE instance_id = ? AND face_embedding IS NOT NULL'
    ).all(req.instance.id);

    if (employees.length === 0) {
      return res.json({ success: false, matched: false, message: 'No enrolled employees found. Enroll faces first.' });
    }

    // Extract descriptor from uploaded photo
    const result = await extractDescriptor(req.file.buffer);
    if (!result) {
      return res.json({ success: true, matched: false, message: 'No face detected in photo. Try again with better lighting.' });
    }

    // Find best match
    const match = findBestMatch(result.descriptor, employees);
    if (!match) {
      return res.json({ success: true, matched: false, message: 'Face not recognised. Not in employee database.', distance: null });
    }

    // Check duplicate punch (5 minute window)
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const recent = db.prepare(
      'SELECT * FROM attendance_logs WHERE employee_id = ? AND punched_at > ? ORDER BY punched_at DESC LIMIT 1'
    ).get(match.employee.id, fiveMinAgo);

    if (recent) {
      const minsAgo = Math.round((Date.now() - new Date(recent.punched_at).getTime()) / 60000);
      const minsLeft = 5 - minsAgo;
      return res.json({
        success: true, matched: true, punched: false,
        message: `Already marked ${minsAgo} min ago. Wait ${minsLeft} more min.`,
        employee: { id: match.employee.id, name: match.employee.name },
      });
    }

    // Save attendance log
    const now    = new Date();
    const logId  = uuidv4();
    db.prepare(`
      INSERT INTO attendance_logs (id, instance_id, employee_id, employee_name, department, punched_at, date, confidence, face_distance, source)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      logId, req.instance.id, match.employee.id,
      match.employee.name, match.employee.department,
      now.toISOString(), now.toISOString().split('T')[0],
      match.confidence, match.distance, 'mobile'
    );

    res.json({
      success:    true,
      matched:    true,
      punched:    true,
      log_id:     logId,
      employee: {
        id:         match.employee.id,
        name:       match.employee.name,
        department: match.employee.department,
        code:       match.employee.employee_code,
      },
      punched_at:  now.toISOString(),
      confidence:  match.confidence,
      distance:    match.distance,
    });

  } catch (err) {
    console.error('Punch error:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// POST /api/punch/batch — sync offline queue
app.post('/api/punch/batch', requireAuth, express.json(), async (req, res) => {
  const { punches } = req.body;
  if (!Array.isArray(punches) || punches.length === 0) {
    return res.status(400).json({ success: false, message: 'punches array required' });
  }

  const employees = db.prepare(
    'SELECT id, name, department, employee_code, face_embedding FROM employees WHERE instance_id = ? AND face_embedding IS NOT NULL'
  ).all(req.instance.id);

  const results = [];

  for (const punch of punches) {
    try {
      if (!punch.photo_b64) { results.push({ local_id: punch.local_id, status: 'error', reason: 'No photo' }); continue; }

      const buffer = Buffer.from(punch.photo_b64, 'base64');
      const result = await extractDescriptor(buffer);
      if (!result) { results.push({ local_id: punch.local_id, status: 'not_matched', reason: 'No face detected' }); continue; }

      const match = findBestMatch(result.descriptor, employees);
      if (!match) { results.push({ local_id: punch.local_id, status: 'not_matched', reason: 'Not recognised' }); continue; }

      const punchedAt = new Date(punch.punched_at);
      const window    = 5 * 60 * 1000;
      const dupCheck  = db.prepare(
        'SELECT id FROM attendance_logs WHERE employee_id = ? AND punched_at BETWEEN ? AND ?'
      ).get(match.employee.id,
        new Date(punchedAt.getTime() - window).toISOString(),
        new Date(punchedAt.getTime() + window).toISOString()
      );

      if (dupCheck) { results.push({ local_id: punch.local_id, status: 'duplicate' }); continue; }

      const logId = uuidv4();
      db.prepare(`
        INSERT INTO attendance_logs (id, instance_id, employee_id, employee_name, department, punched_at, date, confidence, face_distance, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'offline_sync')
      `).run(logId, req.instance.id, match.employee.id, match.employee.name, match.employee.department,
             punchedAt.toISOString(), punchedAt.toISOString().split('T')[0], match.confidence, match.distance);

      results.push({ local_id: punch.local_id, status: 'synced', log_id: logId });
    } catch (err) {
      results.push({ local_id: punch.local_id, status: 'error', reason: err.message });
    }
  }

  const synced = results.filter(r => r.status === 'synced').length;
  res.json({ success: true, synced, total: results.length, results });
});

// GET /api/attendance — list logs
app.get('/api/attendance', requireAuth, (req, res) => {
  let query = 'SELECT * FROM attendance_logs WHERE instance_id = ?';
  const params = [req.instance.id];

  if (req.query.date) {
    query += ' AND date = ?';
    params.push(req.query.date);
  }
  if (req.query.employee_id) {
    query += ' AND employee_id = ?';
    params.push(req.query.employee_id);
  }

  query += ' ORDER BY punched_at DESC LIMIT 200';
  const logs = db.prepare(query).all(...params);
  res.json({ success: true, data: { data: logs } });
});

// ── ADMIN — create instance (no auth needed, for setup) ───
// POST /api/admin/instance
// Body: { name, secret_code, admin_key }
// admin_key must match ADMIN_KEY env var (default: "changeme")
app.post('/api/admin/instance', (req, res) => {
  const adminKey = process.env.ADMIN_KEY || 'changeme';
  if (req.body.admin_key !== adminKey) return res.status(403).json({ success: false, message: 'Wrong admin key' });
  const { name, secret_code } = req.body;
  if (!name || !secret_code) return res.status(400).json({ success: false, message: 'name and secret_code required' });

  const existing = db.prepare('SELECT id FROM instances WHERE secret_code = ?').get(secret_code);
  if (existing) return res.status(409).json({ success: false, message: 'Secret code already exists' });

  const id = uuidv4();
  db.prepare('INSERT INTO instances (id, name, secret_code) VALUES (?, ?, ?)').run(id, name, secret_code);
  res.json({ success: true, instance: { id, name, secret_code } });
});

// GET /api/admin/instances — list all instances
app.get('/api/admin/instances', (req, res) => {
  const adminKey = process.env.ADMIN_KEY || 'changeme';
  if (req.query.admin_key !== adminKey) return res.status(403).json({ success: false, message: 'Wrong admin key' });
  const instances = db.prepare('SELECT id, name, secret_code, is_active, created_at FROM instances').all();
  res.json({ success: true, instances });
});

// ════════════════════════════════════════════════════════════
//  START SERVER
// ════════════════════════════════════════════════════════════
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`\nFaceAttend backend running on port ${PORT}`);
    console.log(`Health: http://localhost:${PORT}/health`);
    console.log(`Default secret code: DEMO1234\n`);
  });
});
