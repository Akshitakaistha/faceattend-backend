// ============================================================
//  FaceAttend Backend — REAL Face Recognition
//
//  Uses face-api.js with TensorFlow.js (pure JS backend).
//  No native modules — works on Render.com Node 25.
//
//  Face recognition pipeline:
//    1. jimp loads the image (pure JS, no canvas needed)
//    2. face-api.js detects face with SSD MobileNet v1
//    3. 68-point landmark detection
//    4. FaceNet extracts real 128-D neural descriptor
//    5. Euclidean distance matching (threshold 0.50)
//
//  Deploy on Render.com:
//    Build command:  npm install && node download-models.js
//    Start command:  node server.js
// ============================================================

const express    = require('express');
const cors       = require('cors');
const multer     = require('multer');
const { v4: uuidv4 } = require('uuid');
const path       = require('path');
const fs         = require('fs');
const initSqlJs  = require('sql.js');
const Jimp       = require('jimp');

// ── TensorFlow.js pure JS backend (no native compilation) ─
const tf = require('@tensorflow/tfjs');

// ── face-api.js browser/node-agnostic build ───────────────
// We use the ES module compatible dist that works with tfjs
const faceapi = require('@vladmandic/face-api/dist/face-api.node-wasm.js');

const app    = express();
const PORT   = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: '20mb' }));

// ─────────────────────────────────────────────────────────
//  face-api.js model loading
// ─────────────────────────────────────────────────────────
const MODEL_PATH  = path.join(__dirname, 'models');
let   modelsReady = false;

async function loadFaceModels() {
  if (!fs.existsSync(MODEL_PATH)) {
    console.warn('WARNING: models/ folder not found. Run: node download-models.js');
    return false;
  }
  try {
    console.log('Loading face recognition models...');
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
    console.log('  ✓ SSD MobileNet (face detection)');
    await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
    console.log('  ✓ Face Landmark 68');
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
    console.log('  ✓ FaceNet 128-D recognition');
    modelsReady = true;
    console.log('All models loaded — face recognition ready!\n');
    return true;
  } catch (err) {
    console.error('Model load failed:', err.message);
    return false;
  }
}

// ─────────────────────────────────────────────────────────
//  Image → TF tensor using jimp (pure JS, no canvas)
// ─────────────────────────────────────────────────────────
async function imageBufferToTensor(buffer) {
  // Load image with jimp (handles JPEG, PNG, etc.)
  const image = await Jimp.read(buffer);
  const { width, height } = image.bitmap;

  // Extract RGBA pixel data
  const rgba = new Uint8Array(width * height * 4);
  let idx = 0;
  image.scan(0, 0, width, height, function(x, y, offset) {
    rgba[idx++] = this.bitmap.data[offset];     // R
    rgba[idx++] = this.bitmap.data[offset + 1]; // G
    rgba[idx++] = this.bitmap.data[offset + 2]; // B
    rgba[idx++] = this.bitmap.data[offset + 3]; // A
  });

  // Convert to RGB tensor (face-api expects [H, W, 3])
  const tensor = tf.tidy(() => {
    const t = tf.tensor3d(rgba, [height, width, 4], 'int32');
    return t.slice([0, 0, 0], [-1, -1, 3]).toFloat(); // drop alpha channel
  });

  return { tensor, width, height };
}

// ─────────────────────────────────────────────────────────
//  Extract real 128-D FaceNet descriptor
// ─────────────────────────────────────────────────────────
async function extractFaceDescriptor(imageBuffer) {
  if (!modelsReady) {
    throw new Error('Face models not loaded yet');
  }

  const { tensor } = await imageBufferToTensor(imageBuffer);

  try {
    const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 });

    // Detect face + landmarks + descriptor in one pass
    const detection = await faceapi
      .detectSingleFace(tensor, options)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      return { descriptor: null, confidence: 0, error: 'No face detected' };
    }

    return {
      descriptor: Array.from(detection.descriptor), // real 128-D FaceNet vector
      confidence: detection.detection.score,
      box: {
        x:      Math.round(detection.detection.box.x),
        y:      Math.round(detection.detection.box.y),
        width:  Math.round(detection.detection.box.width),
        height: Math.round(detection.detection.box.height),
      },
      error: null,
    };
  } finally {
    tensor.dispose(); // free GPU/memory
  }
}

// ─────────────────────────────────────────────────────────
//  Matching — Euclidean distance (face-api.js standard)
//
//  face-api.js uses L2 (Euclidean) distance, NOT cosine.
//  Standard threshold: 0.6
//    distance < 0.4  = very confident same person
//    distance < 0.6  = likely same person
//    distance > 0.6  = different person
// ─────────────────────────────────────────────────────────
function euclideanDist(a, b) {
  if (!a || !b || a.length !== b.length) return 1.0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function distToConfidence(dist) {
  // distance 0 = 100%, distance 0.6 = 0%
  return Math.max(0, Math.round((1 - dist / 0.6) * 100));
}

function findBestMatch(capturedDescriptor, employees) {
  const THRESHOLD = 0.55; // Euclidean distance — lower = stricter
  const MARGIN    = 0.06; // best must beat 2nd-best by this much

  let bestDist   = Infinity;
  let secondDist = Infinity;
  let bestEmp    = null;

  for (const emp of employees) {
    let stored;
    try { stored = JSON.parse(emp.face_embedding); } catch { continue; }
    if (!Array.isArray(stored) || stored.length < 64) continue;

    const dist = euclideanDist(capturedDescriptor, stored);

    if (dist < bestDist) {
      secondDist = bestDist;
      bestDist   = dist;
      bestEmp    = emp;
    } else if (dist < secondDist) {
      secondDist = dist;
    }
  }

  if (!bestEmp || bestDist > THRESHOLD) return null;
  if (employees.length > 1 && (secondDist - bestDist) < MARGIN) return null;

  return {
    employee:   bestEmp,
    distance:   +bestDist.toFixed(4),
    confidence: distToConfidence(bestDist),
  };
}

// ─────────────────────────────────────────────────────────
//  SQLite (sql.js — pure JS)
// ─────────────────────────────────────────────────────────
const DB_FILE = process.env.DB_PATH || path.join(__dirname, 'data', 'faceattend.db');
fs.mkdirSync(path.dirname(DB_FILE), { recursive: true });
let db;

async function initDB() {
  const SQL = await initSqlJs();
  db = fs.existsSync(DB_FILE)
    ? new SQL.Database(fs.readFileSync(DB_FILE))
    : new SQL.Database();

  db.run(`
    CREATE TABLE IF NOT EXISTS instances (
      id TEXT PRIMARY KEY, name TEXT NOT NULL,
      secret_code TEXT UNIQUE NOT NULL, is_active INTEGER DEFAULT 1,
      created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS tokens (
      id TEXT PRIMARY KEY, instance_id TEXT NOT NULL,
      device_name TEXT, created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS employees (
      id TEXT PRIMARY KEY, instance_id TEXT NOT NULL,
      employee_code TEXT, name TEXT NOT NULL, department TEXT,
      face_embedding TEXT, face_enrolled_at TEXT,
      created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS attendance_logs (
      id TEXT PRIMARY KEY, instance_id TEXT NOT NULL,
      employee_id TEXT NOT NULL, employee_name TEXT, department TEXT,
      punched_at TEXT NOT NULL, date TEXT NOT NULL,
      confidence REAL, face_distance REAL, source TEXT DEFAULT 'mobile',
      created_at TEXT DEFAULT (datetime('now'))
    );
  `);

  if (dbGet('SELECT COUNT(*) as c FROM instances').c === 0) {
    dbRun('INSERT INTO instances (id,name,secret_code) VALUES (?,?,?)',
      [uuidv4(), 'Demo Office', 'DEMO1234']);
    console.log('Created default instance — secret code: DEMO1234');
  }

  persistDB();
  console.log('SQLite database ready\n');
}

function persistDB() {
  if (db) fs.writeFileSync(DB_FILE, Buffer.from(db.export()));
}

function dbAll(sql, p = []) {
  const s = db.prepare(sql), rows = [];
  s.bind(p);
  while (s.step()) rows.push(s.getAsObject());
  s.free();
  return rows;
}
function dbGet(sql, p = []) { return dbAll(sql, p)[0] || null; }
function dbRun(sql, p = []) { db.run(sql, p); persistDB(); }

// ─────────────────────────────────────────────────────────
//  Auth middleware
// ─────────────────────────────────────────────────────────
function requireAuth(req, res, next) {
  const token = (req.headers.authorization || '').replace('Bearer ', '').trim();
  if (!token) return res.status(401).json({ success: false, message: 'No token' });
  const row = dbGet('SELECT * FROM tokens WHERE id=?', [token]);
  if (!row) return res.status(401).json({ success: false, message: 'Invalid token' });
  const inst = dbGet('SELECT * FROM instances WHERE id=? AND is_active=1', [row.instance_id]);
  if (!inst) return res.status(401).json({ success: false, message: 'Instance not found' });
  req.instance = inst;
  next();
}

// ═════════════════════════════════════════════════════════
//  ROUTES
// ═════════════════════════════════════════════════════════

app.get('/', (req, res) => res.json({
  service: 'FaceAttend', status: 'running',
  modelsReady, time: new Date().toISOString(),
}));

app.get('/health', (req, res) => res.json({
  status: 'ok', modelsReady, time: new Date().toISOString(),
}));

// ── Auth ──────────────────────────────────────────────────
app.post('/api/login', (req, res) => {
  const { secret_code, device_name } = req.body;
  if (!secret_code) return res.status(400).json({ success: false, message: 'secret_code required' });
  const inst = dbGet('SELECT * FROM instances WHERE secret_code=? AND is_active=1', [secret_code]);
  if (!inst) return res.status(401).json({ success: false, message: 'Invalid secret code' });
  const tid = uuidv4();
  dbRun('INSERT INTO tokens (id,instance_id,device_name) VALUES (?,?,?)',
    [tid, inst.id, device_name || 'unknown']);
  res.json({ success: true, token: tid, instance_id: inst.id, instance_name: inst.name });
});

app.post('/api/logout', requireAuth, (req, res) => {
  dbRun('DELETE FROM tokens WHERE id=?',
    [(req.headers.authorization || '').replace('Bearer ', '').trim()]);
  res.json({ success: true });
});

// ── Employees ─────────────────────────────────────────────
app.get('/api/employees', requireAuth, (req, res) => {
  res.json({ success: true, employees: dbAll(
    'SELECT id,employee_code,name,department,face_embedding,face_enrolled_at FROM employees WHERE instance_id=? ORDER BY name',
    [req.instance.id]
  )});
});

app.post('/api/employees', requireAuth, (req, res) => {
  const { name, employee_code, department } = req.body;
  if (!name) return res.status(400).json({ success: false, message: 'name required' });
  const id = uuidv4();
  dbRun('INSERT INTO employees (id,instance_id,name,employee_code,department) VALUES (?,?,?,?,?)',
    [id, req.instance.id, name.trim(), employee_code || null, department || null]);
  res.json({ success: true, employee: dbGet('SELECT * FROM employees WHERE id=?', [id]) });
});

// POST /api/employees/:id/enroll
app.post('/api/employees/:id/enroll', requireAuth, upload.single('photo'), async (req, res) => {
  if (!modelsReady) {
    return res.status(503).json({
      success: false,
      message: 'Face models are still loading. Wait 30 seconds and try again.',
    });
  }

  const emp = dbGet('SELECT * FROM employees WHERE id=? AND instance_id=?',
    [req.params.id, req.instance.id]);
  if (!emp) return res.status(404).json({ success: false, message: 'Employee not found' });
  if (!req.file) return res.status(400).json({ success: false, message: 'No photo uploaded' });

  try {
    const result = await extractFaceDescriptor(req.file.buffer);

    if (!result.descriptor) {
      return res.status(422).json({
        success: false,
        message: result.error || 'No face detected. Use a clear, well-lit, front-facing photo.',
      });
    }

    dbRun('UPDATE employees SET face_embedding=?,face_enrolled_at=? WHERE id=?',
      [JSON.stringify(result.descriptor), new Date().toISOString(), emp.id]);

    res.json({
      success:    true,
      message:    'Face enrolled successfully with FaceNet 128-D descriptor',
      confidence: +(result.confidence * 100).toFixed(1) + '%',
      box:        result.box,
    });
  } catch (err) {
    console.error('Enroll error:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

app.delete('/api/employees/:id/face', requireAuth, (req, res) => {
  const emp = dbGet('SELECT id FROM employees WHERE id=? AND instance_id=?',
    [req.params.id, req.instance.id]);
  if (!emp) return res.status(404).json({ success: false, message: 'Employee not found' });
  dbRun('UPDATE employees SET face_embedding=NULL,face_enrolled_at=NULL WHERE id=?', [emp.id]);
  res.json({ success: true });
});

// ── Attendance punch ──────────────────────────────────────
app.post('/api/punch', requireAuth, upload.single('photo'), async (req, res) => {
  if (!modelsReady) {
    return res.status(503).json({
      success: false,
      message: 'Face models are still loading. Wait 30 seconds and try again.',
    });
  }
  if (!req.file) return res.status(400).json({ success: false, message: 'No photo uploaded' });

  try {
    const employees = dbAll(
      'SELECT id,name,department,employee_code,face_embedding FROM employees WHERE instance_id=? AND face_embedding IS NOT NULL',
      [req.instance.id]
    );

    if (employees.length === 0) {
      return res.json({ success: false, matched: false, message: 'No enrolled employees found. Enroll faces first.' });
    }

    const result = await extractFaceDescriptor(req.file.buffer);

    if (!result.descriptor) {
      return res.json({
        success: true, matched: false,
        message: 'No face detected in photo. ' + (result.error || 'Try better lighting.'),
      });
    }

    const match = findBestMatch(result.descriptor, employees);

    // Log all scores for debugging
    const allScores = employees.map(emp => {
      try {
        const stored = JSON.parse(emp.face_embedding);
        return { name: emp.name, distance: +euclideanDist(result.descriptor, stored).toFixed(4) };
      } catch { return { name: emp.name, distance: 1.0 }; }
    }).sort((a, b) => a.distance - b.distance);

    console.log('Face match scores (lower = better):', JSON.stringify(allScores));

    if (!match) {
      return res.json({
        success:      true,
        matched:      false,
        message:      `Face not recognised. Best distance: ${allScores[0]?.distance} (need < 0.55)`,
        debug_scores: allScores.slice(0, 3),
      });
    }

    // Duplicate check (5 min)
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const recent = dbGet(
      'SELECT * FROM attendance_logs WHERE employee_id=? AND punched_at>? ORDER BY punched_at DESC LIMIT 1',
      [match.employee.id, fiveMinAgo]
    );
    if (recent) {
      const ago  = Math.round((Date.now() - new Date(recent.punched_at)) / 60000);
      const left = 5 - ago;
      return res.json({
        success: true, matched: true, punched: false,
        message: `Already marked ${ago} min ago. Wait ${left} more min.`,
        employee: { id: match.employee.id, name: match.employee.name },
      });
    }

    const now = new Date(), logId = uuidv4();
    dbRun(`INSERT INTO attendance_logs (id,instance_id,employee_id,employee_name,department,punched_at,date,confidence,face_distance,source)
           VALUES (?,?,?,?,?,?,?,?,?,'mobile')`,
      [logId, req.instance.id, match.employee.id, match.employee.name,
       match.employee.department, now.toISOString(), now.toISOString().split('T')[0],
       match.confidence, match.distance]);

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

// ── Batch sync ────────────────────────────────────────────
app.post('/api/punch/batch', requireAuth, async (req, res) => {
  const { punches } = req.body;
  if (!Array.isArray(punches) || !punches.length)
    return res.status(400).json({ success: false, message: 'punches array required' });

  const employees = dbAll(
    'SELECT id,name,department,employee_code,face_embedding FROM employees WHERE instance_id=? AND face_embedding IS NOT NULL',
    [req.instance.id]
  );
  const results = [];

  for (const punch of punches) {
    try {
      const buf    = Buffer.from(punch.photo_b64, 'base64');
      const result = await extractFaceDescriptor(buf);
      if (!result.descriptor) { results.push({ local_id: punch.local_id, status: 'no_face' }); continue; }

      const match = findBestMatch(result.descriptor, employees);
      if (!match) { results.push({ local_id: punch.local_id, status: 'not_matched' }); continue; }

      const pAt = new Date(punch.punched_at), win = 5 * 60 * 1000;
      const dup = dbGet(
        'SELECT id FROM attendance_logs WHERE employee_id=? AND punched_at BETWEEN ? AND ?',
        [match.employee.id, new Date(+pAt - win).toISOString(), new Date(+pAt + win).toISOString()]
      );
      if (dup) { results.push({ local_id: punch.local_id, status: 'duplicate' }); continue; }

      const lid = uuidv4();
      dbRun(`INSERT INTO attendance_logs (id,instance_id,employee_id,employee_name,department,punched_at,date,confidence,face_distance,source)
             VALUES (?,?,?,?,?,?,?,?,?,'offline_sync')`,
        [lid, req.instance.id, match.employee.id, match.employee.name, match.employee.department,
         pAt.toISOString(), pAt.toISOString().split('T')[0], match.confidence, match.distance]);
      results.push({ local_id: punch.local_id, status: 'synced', log_id: lid });
    } catch (err) {
      results.push({ local_id: punch.local_id, status: 'error', reason: err.message });
    }
  }
  res.json({ success: true, synced: results.filter(r => r.status === 'synced').length, total: results.length, results });
});

// ── Logs ─────────────────────────────────────────────────
app.get('/api/attendance', requireAuth, (req, res) => {
  let sql = 'SELECT * FROM attendance_logs WHERE instance_id=?';
  const p = [req.instance.id];
  if (req.query.date)        { sql += ' AND date=?';        p.push(req.query.date); }
  if (req.query.employee_id) { sql += ' AND employee_id=?'; p.push(req.query.employee_id); }
  sql += ' ORDER BY punched_at DESC LIMIT 200';
  res.json({ success: true, data: { data: dbAll(sql, p) } });
});

// ── Admin ─────────────────────────────────────────────────
const ADMIN_KEY = process.env.ADMIN_KEY || 'changeme123';

app.post('/api/admin/instance', (req, res) => {
  if (req.body.admin_key !== ADMIN_KEY)
    return res.status(403).json({ success: false, message: 'Wrong admin key' });
  const { name, secret_code } = req.body;
  if (!name || !secret_code)
    return res.status(400).json({ success: false, message: 'name and secret_code required' });
  if (dbGet('SELECT id FROM instances WHERE secret_code=?', [secret_code]))
    return res.status(409).json({ success: false, message: 'Secret code already in use' });
  const id = uuidv4();
  dbRun('INSERT INTO instances (id,name,secret_code) VALUES (?,?,?)', [id, name, secret_code]);
  res.json({ success: true, instance: { id, name, secret_code } });
});

app.get('/api/admin/instances', (req, res) => {
  if (req.query.admin_key !== ADMIN_KEY)
    return res.status(403).json({ success: false, message: 'Wrong admin key' });
  res.json({ success: true, instances: dbAll('SELECT * FROM instances') });
});

// ═════════════════════════════════════════════════════════
//  START
// ═════════════════════════════════════════════════════════
async function start() {
  await initDB();
  await loadFaceModels();  // loads ~100MB of neural network weights

  app.listen(PORT, () => {
    console.log('FaceAttend backend running on port ' + PORT);
    console.log('Health: http://localhost:' + PORT + '/health');
    console.log('Default secret code: DEMO1234');
    if (!modelsReady) {
      console.log('\nWARNING: Models not loaded. Run: node download-models.js');
    }
  });
}

start().catch(err => { console.error('Fatal:', err); process.exit(1); });
