// node download-models.js
const https = require('https');
const fs    = require('fs');
const path  = require('path');

const DIR  = path.join(__dirname, 'models');
const BASE = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model';
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

if (!fs.existsSync(DIR)) fs.mkdirSync(DIR, { recursive: true });

const dl = (url, dest) => new Promise((resolve, reject) => {
  if (fs.existsSync(dest) && fs.statSync(dest).size > 100) {
    console.log('  already exists: ' + path.basename(dest));
    return resolve();
  }
  const file = fs.createWriteStream(dest);
  const get  = (u) => https.get(u, res => {
    if (res.statusCode === 301 || res.statusCode === 302) {
      file.close();
      return get(res.headers.location);
    }
    if (res.statusCode !== 200) return reject(new Error('HTTP ' + res.statusCode));
    let n = 0;
    res.on('data', c => { n += c.length; process.stdout.write('\r  downloading ' + path.basename(dest) + ' ' + (n/1024).toFixed(0) + 'KB'); });
    res.pipe(file);
    file.on('finish', () => { file.close(); process.stdout.write('\n'); resolve(); });
  }).on('error', reject);
  get(url);
});

(async () => {
  console.log('Downloading face recognition models (~100MB)...\n');
  let failed = 0;
  for (const f of FILES) {
    try { await dl(BASE + '/' + f, path.join(DIR, f)); }
    catch (e) { console.error('  FAILED ' + f + ': ' + e.message); failed++; }
  }
  console.log(failed === 0 ? '\nAll models ready.' : '\n' + failed + ' files failed, re-run to retry.');
})();
