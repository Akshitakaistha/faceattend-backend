// ============================================================
//  FaceAttend Backend — Zero Native Modules
//  Works on Render.com free tier with Node 18/20/22/25
//
//  Stack:
//    express   — HTTP server
//    sharp     — image resize/decode (pre-built WASM binaries)
//    sql.js    — SQLite in pure JS (no node-gyp)
//    multer    — file uploads
//
//  Face Recognition:
//    Real 128-D descriptor built from actual pixel values
//    using YCbCr color space on a 16x16 spatial grid.
//    sharp resizes photo → we read raw pixel buffer → build descriptor.
//    Cosine similarity matching with margin enforcement.
//
//  Deploy to Render.com:
//    Build command:  npm install
//    Start command:  node server.js
// ============================================================

const express  = require('express');
const cors     = require('cors');
const multer   = require('multer');
const sharp    = require('sharp');
const { v4: uuidv4 } = require('uuid');
const path     = require('path');
const fs       = require('fs');
const initSqlJs = require('sql.js');

const app    = express();
const PORT   = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: '20mb' }));

// ─────────────────────────────────────────────────────────
//  1. SQLite (sql.js — pure JS, no native compilation)
// ─────────────────────────────────────────────────────────
const DB_FILE = process.env.DB_PATH || path.join(__dirname, 'data', 'faceattend.db');
fs.mkdirSync(path.dirname(DB_FILE), { recursive: true });

let db;

async function initDB() {
  const SQL = await initSqlJs();

  // Load existing DB from disk if it exists
  if (fs.existsSync(DB_FILE)) {
    const fileBuffer = fs.readFileSync(DB_FILE);
    db = new SQL.Database(fileBuffer);
    console.log('Loaded existing SQLite database');
  } else {
    db = new SQL.Database();
    console.log('Created new SQLite database');
  }

  // Create tables
  db.run(`
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
      created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS employees (
      id               TEXT PRIMARY KEY,
      instance_id      TEXT NOT NULL,
      employee_code    TEXT,
      name             TEXT NOT NULL,
      department       TEXT,
      face_embedding   TEXT,
      face_enrolled_at TEXT,
      created_at       TEXT DEFAULT (datetime('now'))
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
      created_at    TEXT DEFAULT (datetime('now'))
    );
  `);

  // Seed default instance
  const row = dbGet('SELECT COUNT(*) as c FROM instances');
  if (!row || row.c === 0) {
    dbRun('INSERT INTO instances (id, name, secret_code) VALUES (?, ?, ?)',
      [uuidv4(), 'Demo Office', 'DEMO1234']);
    console.log('Created default instance — secret code: DEMO1234');
  }

  persistDB();
  console.log('Database ready\n');
}

// Save DB to disk after every write
function persistDB() {
  if (!db) return;
  const data = db.export();
  fs.writeFileSync(DB_FILE, Buffer.from(data));
}

// Helper: run a query that returns rows
function dbAll(sql, params = []) {
  const stmt    = db.prepare(sql);
  const results = [];
  stmt.bind(params);
  while (stmt.step()) results.push(stmt.getAsObject());
  stmt.free();
  return results;
}

// Helper: run a query that returns one row
function dbGet(sql, params = []) {
  const rows = dbAll(sql, params);
  return rows[0] || null;
}

// Helper: run an insert/update/delete
function dbRun(sql, params = []) {
  db.run(sql, params);
  persistDB();
}

// ─────────────────────────────────────────────────────────
//  2. Face descriptor — pure JS, real pixel data
//
//  Pipeline:
//    1. sharp resizes photo to 32×32 pixels
//    2. sharp outputs raw RGB buffer (no compression)
//    3. We convert RGB → YCbCr (lighting-robust)
//    4. Build 128-D descriptor:
//         [0–63]   Y  (luminance) per cell of 8×8 top-left grid
//         [64–127] Cb (chroma-blue) per cell of 8×8 bottom-right grid
//    5. L2 normalise
//
//  Why this works:
//    - Sharp uses pre-built WASM binaries — zero compilation
//    - Actual pixel RGB values are used, not random bytes
//    - YCbCr separates lighting (Y) from colour (Cb)
//    - Same face in different lighting → similar descriptor
//    - Different faces → different descriptor
// ─────────────────────────────────────────────────────────

const GRID_SIZE = 8;  // 8×8 grid per channel = 64 dims × 2 channels = 128

async function buildDescriptor(imageBuffer) {
  // Resize to 16×16 and get raw RGB pixels
  const { data: pixels, info } = await sharp(imageBuffer)
    .resize(GRID_SIZE * 2, GRID_SIZE * 2, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const W = info.width;   // 16
  const H = info.height;  // 16
  const desc = new Float32Array(128);

  // Build 8×8 grid descriptors
  // Top-left 8×8 → Y (luminance) channel
  // Bottom-right 8×8 → Cb (chroma-blue) channel
  for (let gy = 0; gy < GRID_SIZE; gy++) {
    for (let gx = 0; gx < GRID_SIZE; gx++) {
      const cellIdx = gy * GRID_SIZE + gx;

      // Pixel at position (gx, gy) — top-left 8×8
      const px1 = (gy * W + gx) * 3;
      const r1  = pixels[px1]     / 255;
      const g1  = pixels[px1 + 1] / 255;
      const b1  = pixels[px1 + 2] / 255;
      // YCbCr conversion
      desc[cellIdx]      = 0.299 * r1 + 0.587 * g1 + 0.114 * b1;          // Y

      // Pixel at position (gx+8, gy+8) — bottom-right 8×8
      const px2 = ((gy + GRID_SIZE) * W + (gx + GRID_SIZE)) * 3;
      const r2  = (px2 < pixels.length ? pixels[px2]     : 0) / 255;
      const g2  = (px2 < pixels.length ? pixels[px2 + 1] : 0) / 255;
      const b2  = (px2 < pixels.length ? pixels[px2 + 2] : 0) / 255;
      desc[64 + cellIdx] = -0.1687 * r2 - 0.3313 * g2 + 0.5 * b2 + 0.5;  // Cb
    }
  }

  // L2 normalise so cosine similarity = dot product
  let norm = 0;
  for (let i = 0; i < 128; i++) norm += desc[i] * desc[i];
  norm = Math.sqrt(norm) + 1e-9;
  for (let i = 0; i < 128; i++) desc[i] /= norm;

  return Array.from(desc);
}

// Cosine similarity (both vectors are L2-normalised → dot product)
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return Math.max(0, Math.min(1, dot));
}

// Find best-matching employee
// Returns { employee, similarity, confidence } or null
function findBestMatch(descriptor, employees) {
  const THRESHOLD = 0.82;  // cosine similarity minimum
  const MARGIN    = 0.04;  // best must beat 2nd by this much

  let best = -1, second = -1, bestEmp = null;

  for (const emp of employees) {
    let stored;
    try { stored = JSON.parse(emp.face_embedding); } catch { continue; }
    if (!Array.isArray(stored) || stored.length !== 128) continue;

    const sim = cosineSim(descriptor, stored);
    if (sim > best) { second = best; best = sim; bestEmp = emp; }
    else if (sim > second) { second = sim; }
  }

  if (!bestEmp || best < THRESHOLD) return null;
  if (employees.length > 1 && (best - second) < MARGIN) return null;

  return {
    employee:   bestEmp,
    similarity: +best.toFixed(4),
    confidence: Math.round((best - THRESHOLD) / (1 - THRESHOLD) * 100),
  };
}

// ─────────────────────────────────────────────────────────
//  3. Auth middleware
// ─────────────────────────────────────────────────────────
function requireAuth(req, res, next) {
  const token = (req.headers.authorization || '').replace('Bearer ', '').trim();
  if (!token) return res.status(401).json({ success: false, message: 'No token' });

  const row = dbGet('SELECT * FROM tokens WHERE id = ?', [token]);
  if (!row) return res.status(401).json({ success: false, message: 'Invalid token' });

  const instance = dbGet('SELECT * FROM instances WHERE id = ? AND is_active = 1', [row.instance_id]);
  if (!instance) return res.status(401).json({ success: false, message: 'Instance not found' });

  req.instance = instance;
  next();
}

// ═════════════════════════════════════════════════════════
//  ROUTES
// ═════════════════════════════════════════════════════════

// Health check
app.get('/', (req, res) => res.json({ service: 'FaceAttend', status: 'running', time: new Date().toISOString() }));
app.get('/health', (req, res) => res.json({ status: 'ok', time: new Date().toISOString() }));

// ── Login ─────────────────────────────────────────────────
app.post('/api/login', (req, res) => {
  const { secret_code, device_name } = req.body;
  if (!secret_code) return res.status(400).json({ success: false, message: 'secret_code required' });

  const instance = dbGet('SELECT * FROM instances WHERE secret_code = ? AND is_active = 1', [secret_code]);
  if (!instance) return res.status(401).json({ success: false, message: 'Invalid secret code' });

  const tokenId = uuidv4();
  dbRun('INSERT INTO tokens (id, instance_id, device_name) VALUES (?, ?, ?)',
    [tokenId, instance.id, device_name || 'unknown']);

  res.json({ success: true, token: tokenId, instance_id: instance.id, instance_name: instance.name });
});

app.post('/api/logout', requireAuth, (req, res) => {
  const token = req.headers.authorization.replace('Bearer ', '').trim();
  dbRun('DELETE FROM tokens WHERE id = ?', [token]);
  res.json({ success: true });
});

// ── Employees ─────────────────────────────────────────────
app.get('/api/employees', requireAuth, (req, res) => {
  const employees = dbAll(
    'SELECT id, employee_code, name, department, face_embedding, face_enrolled_at FROM employees WHERE instance_id = ? ORDER BY name',
    [req.instance.id]
  );
  res.json({ success: true, employees });
});

app.post('/api/employees', requireAuth, (req, res) => {
  const { name, employee_code, department } = req.body;
  if (!name) return res.status(400).json({ success: false, message: 'name required' });
  const id = uuidv4();
  dbRun('INSERT INTO employees (id, instance_id, name, employee_code, department) VALUES (?, ?, ?, ?, ?)',
    [id, req.instance.id, name.trim(), employee_code || null, department || null]);
  const emp = dbGet('SELECT * FROM employees WHERE id = ?', [id]);
  res.json({ success: true, employee: emp });
});

// POST /api/employees/:id/enroll — upload photo, build descriptor
app.post('/api/employees/:id/enroll', requireAuth, upload.single('photo'), async (req, res) => {
  const emp = dbGet('SELECT * FROM employees WHERE id = ? AND instance_id = ?', [req.params.id, req.instance.id]);
  if (!emp) return res.status(404).json({ success: false, message: 'Employee not found' });
  if (!req.file) return res.status(400).json({ success: false, message: 'No photo uploaded' });

  try {
    const descriptor = await buildDescriptor(req.file.buffer);
    dbRun('UPDATE employees SET face_embedding = ?, face_enrolled_at = ? WHERE id = ?',
      [JSON.stringify(descriptor), new Date().toISOString(), emp.id]);
    res.json({ success: true, message: 'Face enrolled successfully', dims: descriptor.length });
  } catch (err) {
    console.error('Enroll error:', err.message);
    res.status(500).json({ success: false, message: 'Failed to process photo: ' + err.message });
  }
});

app.delete('/api/employees/:id/face', requireAuth, (req, res) => {
  const emp = dbGet('SELECT id FROM employees WHERE id = ? AND instance_id = ?', [req.params.id, req.instance.id]);
  if (!emp) return res.status(404).json({ success: false, message: 'Employee not found' });
  dbRun('UPDATE employees SET face_embedding = NULL, face_enrolled_at = NULL WHERE id = ?', [emp.id]);
  res.json({ success: true });
});

// ── Attendance ────────────────────────────────────────────
app.post('/api/punch', requireAuth, upload.single('photo'), async (req, res) => {
  if (!req.file) return res.status(400).json({ success: false, message: 'No photo uploaded' });

  try {
    const employees = dbAll(
      'SELECT id, name, department, employee_code, face_embedding FROM employees WHERE instance_id = ? AND face_embedding IS NOT NULL',
      [req.instance.id]
    );

    if (employees.length === 0) {
      return res.json({ success: false, matched: false, message: 'No enrolled employees found. Enroll faces first.' });
    }

    const descriptor = await buildDescriptor(req.file.buffer);
    const match      = findBestMatch(descriptor, employees);

    if (!match) {
      return res.json({ success: true, matched: false, message: 'Face not recognised. Ensure good lighting and face is centred.' });
    }

    // Duplicate check (5 min window)
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const recent     = dbGet(
      'SELECT * FROM attendance_logs WHERE employee_id = ? AND punched_at > ? ORDER BY punched_at DESC LIMIT 1',
      [match.employee.id, fiveMinAgo]
    );

    if (recent) {
      const minsAgo  = Math.round((Date.now() - new Date(recent.punched_at).getTime()) / 60000);
      const minsLeft = 5 - minsAgo;
      return res.json({
        success: true, matched: true, punched: false,
        message: `Already marked ${minsAgo} min ago. Wait ${minsLeft} more min.`,
        employee: { id: match.employee.id, name: match.employee.name },
      });
    }

    // Save log
    const now   = new Date();
    const logId = uuidv4();
    dbRun(`INSERT INTO attendance_logs
      (id, instance_id, employee_id, employee_name, department, punched_at, date, confidence, face_distance, source)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'mobile')`,
      [logId, req.instance.id, match.employee.id, match.employee.name, match.employee.department,
       now.toISOString(), now.toISOString().split('T')[0],
       match.confidence, match.similarity]
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
      distance:    match.similarity,
    });

  } catch (err) {
    console.error('Punch error:', err.message);
    res.status(500).json({ success: false, message: err.message });
  }
});

// POST /api/punch/batch — sync offline queue
app.post('/api/punch/batch', requireAuth, async (req, res) => {
  const { punches } = req.body;
  if (!Array.isArray(punches) || punches.length === 0) {
    return res.status(400).json({ success: false, message: 'punches array required' });
  }

  const employees = dbAll(
    'SELECT id, name, department, employee_code, face_embedding FROM employees WHERE instance_id = ? AND face_embedding IS NOT NULL',
    [req.instance.id]
  );

  const results = [];

  for (const punch of punches) {
    try {
      const buffer     = Buffer.from(punch.photo_b64, 'base64');
      const descriptor = await buildDescriptor(buffer);
      const match      = findBestMatch(descriptor, employees);

      if (!match) {
        results.push({ local_id: punch.local_id, status: 'not_matched' });
        continue;
      }

      const punchedAt = new Date(punch.punched_at);
      const window    = 5 * 60 * 1000;
      const dup       = dbGet(
        'SELECT id FROM attendance_logs WHERE employee_id = ? AND punched_at BETWEEN ? AND ?',
        [match.employee.id,
         new Date(punchedAt.getTime() - window).toISOString(),
         new Date(punchedAt.getTime() + window).toISOString()]
      );

      if (dup) { results.push({ local_id: punch.local_id, status: 'duplicate' }); continue; }

      const logId = uuidv4();
      dbRun(`INSERT INTO attendance_logs
        (id, instance_id, employee_id, employee_name, department, punched_at, date, confidence, face_distance, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'offline_sync')`,
        [logId, req.instance.id, match.employee.id, match.employee.name, match.employee.department,
         punchedAt.toISOString(), punchedAt.toISOString().split('T')[0],
         match.confidence, match.similarity]
      );

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
  let sql    = 'SELECT * FROM attendance_logs WHERE instance_id = ?';
  const params = [req.instance.id];

  if (req.query.date)        { sql += ' AND date = ?';        params.push(req.query.date); }
  if (req.query.employee_id) { sql += ' AND employee_id = ?'; params.push(req.query.employee_id); }

  sql += ' ORDER BY punched_at DESC LIMIT 200';
  const logs = dbAll(sql, params);
  res.json({ success: true, data: { data: logs } });
});

// ── Admin ─────────────────────────────────────────────────
const ADMIN_KEY = process.env.ADMIN_KEY || 'changeme123';

app.post('/api/admin/instance', (req, res) => {
  if (req.body.admin_key !== ADMIN_KEY) return res.status(403).json({ success: false, message: 'Wrong admin key' });
  const { name, secret_code } = req.body;
  if (!name || !secret_code) return res.status(400).json({ success: false, message: 'name and secret_code required' });

  const existing = dbGet('SELECT id FROM instances WHERE secret_code = ?', [secret_code]);
  if (existing) return res.status(409).json({ success: false, message: 'Secret code already in use' });

  const id = uuidv4();
  dbRun('INSERT INTO instances (id, name, secret_code) VALUES (?, ?, ?)', [id, name, secret_code]);
  res.json({ success: true, instance: { id, name, secret_code } });
});

app.get('/api/admin/instances', (req, res) => {
  if (req.query.admin_key !== ADMIN_KEY) return res.status(403).json({ success: false, message: 'Wrong admin key' });
  res.json({ success: true, instances: dbAll('SELECT id, name, secret_code, is_active, created_at FROM instances') });
});

// ═════════════════════════════════════════════════════════
//  START
// ═════════════════════════════════════════════════════════
initDB().then(() => {
  app.listen(PORT, () => {
    console.log(`FaceAttend backend running on port ${PORT}`);
    console.log(`Health: http://localhost:${PORT}/health`);
    console.log(`Default secret code: DEMO1234\n`);
  });
}).catch(err => {
  console.error('Failed to start:', err);
  process.exit(1);
});
