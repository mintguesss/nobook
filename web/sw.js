/* Service Worker：只快取靜態資源。
 * 規格 §8.2：不要攔截 WebSocket 或 API 請求。
 *
 * 策略是 network-first（先連線、失敗才用快取），不是 cache-first。
 * 這點很重要：程式碼檔案用 cache-first 的話，改版永遠送不到使用者手上——
 * 頁面會一直跑舊的 app.js，而且外觀上完全看不出來。
 * 快取的角色只是「伺服器連不上時還能開啟頁面」的備援。
 */
const VERSION = 'v21';
const CACHE = 'nobook-' + VERSION;
const ASSETS = [
  './', 'index.html', 'app.js', 'recorder-worklet.js',
  'manifest.json', 'icon.svg',
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
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // API 與 WebSocket 一律直通，不進快取
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) return;
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;

  e.respondWith(
    fetch(e.request)
      .then((res) => {
        // 連得上就用最新的，順便更新快取
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(e.request).then(
        (hit) => hit || caches.match('index.html')))
  );
});
