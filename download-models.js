// Run once locally OR used as Render.com build step
// node download-models.js

const https = require('https');
const fs    = require('fs');
const path  = require('path');

const MODELS_DIR = path.join(__dirname, 'models');
if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, { recursive: true });

const BASE  = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model';
const FILES = [
  'ssd_mobilenetv1_model-weights_manifest.json',
  'ssd_mobilenetv1_model-shard1',
  'ssd_mobilenetv1_model-shard2',
  'face_landmark_68_model-weights_manifest.json',
  'face_landmark_68_model-shard1',
  'face_recognition_model-weights_manifest.json',
  'face_recognition_model-shard1',
  'face_recognition_model-shard2',
];

const download = (url, dest) => new Promise((resolve, reject) => {
  if (fs.existsSync(dest) && fs.statSync(dest).size > 100) {
    console.log(`  ✓ ${path.basename(dest)} already exists`);
    return resolve();
  }
  const file = fs.createWriteStream(dest);
  https.get(url, res => {
    if (res.statusCode === 301 || res.statusCode === 302) {
      file.close(); fs.unlinkSync(dest);
      return download(res.headers.location, dest).then(resolve).catch(reject);
    }
    if (res.statusCode !== 200) { return reject(new Error(`HTTP ${res.statusCode}`)); }
    let bytes = 0;
    res.on('data', c => { bytes += c.length; process.stdout.write(`\r  ↓ ${path.basename(dest)} — ${(bytes/1024).toFixed(0)} KB`); });
    res.pipe(file);
    file.on('finish', () => { file.close(); process.stdout.write('\n'); resolve(); });
  }).on('error', err => { fs.unlinkSync(dest); reject(err); });
});

(async () => {
  console.log('Downloading FaceNet model weights (~100 MB)...\n');
  for (const f of FILES) {
    try { await download(`${BASE}/${f}`, path.join(MODELS_DIR, f)); }
    catch (e) { console.error(`  ✗ ${f}: ${e.message}`); }
  }
  console.log('\nDone! Run: node server.js');
})();
