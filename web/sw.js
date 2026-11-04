/* Service Worker：只快取靜態資源。
 * 規格 §8.2：不要攔截 WebSocket 或 API 請求。
 *
 * 策略是 network-first（先連線、失敗才用快取），不是 cache-first。
 * 這點很重要：程式碼檔案用 cache-first 的話，改版永遠送不到使用者手上——
 * 頁面會一直跑舊的 app.js，而且外觀上完全看不出來。
 * 快取的角色只是「伺服器連不上時還能開啟頁面」的備援。
 *
 * 快取只在安裝時整批寫入，頁面和程式才會是同一版。
 */
const VERSION = 'v29';
const CACHE = 'nobook-' + VERSION;
// 手機剛打開時 Tailscale 常常還沒連上，請求不會失敗、只會一直卡著。
// 等太久畫面就是一片空白，所以先用快取把畫面開起來。
const NET_TIMEOUT_MS = 2500;
// 剛逾時過就先不等了，不然頁面和程式檔要各等一次，封面要 5 秒才出來
const SLOW_NET_MS = 5000;
let slowUntil = 0;
const ASSETS = [
  './', 'index.html', 'app.js', 'recorder-worklet.js',
  'manifest.json', 'icon.svg', 'icon-180.png', 'icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(ASSETS))
      .then(() => self.skipWaiting())        // 新版立刻接手，不等舊分頁關閉
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
      // 告訴開著的頁面現在是哪一版，頁面自己判斷要不要重新載入
      .then(() => self.clients.matchAll({ type: 'window' }))
      .then((list) => list.forEach((c) => c.postMessage({ type: 'version', version: VERSION })))
  );
});

function fromCache(request) {
  return caches.match(request, { ignoreSearch: true }).then((hit) => {
    if (hit || request.mode !== 'navigate') return hit;
    return caches.match('index.html');
  });
}

// 網路優先；失敗、伺服器沒開（Tailscale 回 502）或等太久才用快取。
// 快取也沒有的話繼續等網路，不要把 HTML 當成程式檔回傳。
function networkFirst(request) {
  return new Promise((resolve, reject) => {
    const abort = new AbortController();
    const net = fetch(request, { cache: 'no-cache', signal: abort.signal });
    const useCache = (fallback) => fromCache(request).then((hit) => {
      if (hit) resolve(hit);
      else if (fallback) fallback();
      return hit;
    });
    // 用了快取就取消請求，卡住的連線不取消會佔著位子，連上之後的請求也出不去
    const giveUp = () => useCache().then((hit) => { if (hit) abort.abort(); });
    const timer = setTimeout(() => {
      slowUntil = Date.now() + SLOW_NET_MS;
      giveUp();
    }, Date.now() < slowUntil ? 0 : NET_TIMEOUT_MS);
    net.then((res) => {
      slowUntil = 0;
      clearTimeout(timer);
      if (res.ok || res.status === 304) resolve(res);
      else useCache(() => resolve(res));
    }).catch((err) => {
      clearTimeout(timer);
      useCache(() => reject(err));
    });
  });
}

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // API 與 WebSocket 一律直通，不進快取
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) return;
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;

  e.respondWith(networkFirst(e.request));
});
