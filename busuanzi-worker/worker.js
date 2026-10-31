/**
 * 不蒜子(busuanzi)兼容计数后端 — Cloudflare Workers + KV
 * 部署:CF Dashboard → Workers → 创建 Worker → 粘贴本文件 → Settings → Bindings →
 *       添加 KV Namespace,变量名必须是 COUNTS → 保存部署
 * 用法:https://<worker名>.<子域>.workers.dev/busuanzi?jsonpCallback=BusuanziCallback_xxx
 * 协议:与 busuanzi.ibruce.info 完全兼容,博客端只需替换脚本地址
 */

const SITE_PV = 'site_pv';
const SITE_UV = 'site_uv';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // CORS:博客页面跨域加载
    const cors = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Credentials': 'true',
      'Cache-Control': 'max-age=0, no-store',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: { ...cors, 'Access-Control-Allow-Headers': '*' } });
    }

    if (url.pathname !== '/busuanzi') {
      return new Response('not found', { status: 404, headers: cors });
    }

    // 并发取三个计数(站点 PV/UV + 本页 PV)
    const referer = request.headers.get('referer') || 'unknown';
    const pageKey = 'page:' + referer;

    const [sitePv, siteUv, pagePv] = await Promise.all([
      env.COUNTS.get(SITE_PV).then(v => parseInt(v || '0', 10)),
      env.COUNTS.get(SITE_UV).then(v => parseInt(v || '0', 10)),
      env.COUNTS.get(pageKey).then(v => parseInt(v || '0', 10)),
    ]);

    // UV 判定:cookie 里的访客 ID,没有则 +1 并种 cookie(两年有效)
    const cookies = parseCookie(request.headers.get('cookie') || '');
    let visitor = cookies['bsz_uuid'];
    const isNewVisitor = !visitor;
    if (isNewVisitor) {
      visitor = crypto.randomUUID();
    }

    // 写回(先算新值)
    await Promise.all([
      env.COUNTS.put(SITE_PV, String(sitePv + 1)),
      env.COUNTS.put(pageKey, String(pagePv + 1)),
      isNewVisitor ? env.COUNTS.put(SITE_UV, String(siteUv + 1)) : Promise.resolve(),
    ]);

    const data = {
      site_pv: sitePv + 1,
      site_uv: isNewVisitor ? siteUv + 1 : siteUv,
      page_pv: pagePv + 1,
    };

    const body = JSON.stringify(data);
    const cb = url.searchParams.get('jsonpCallback') || '';
    const resBody = cb ? `${cb}(${body});` : body;

    const headers = {
      ...cors,
      'Content-Type': 'application/javascript; charset=utf-8',
    };
    if (isNewVisitor) {
      headers['Set-Cookie'] = `bsz_uuid=${visitor}; Path=/; Max-Age=63072000; SameSite=None; Secure`;
    }

    return new Response(resBody, { headers });
  },
};

function parseCookie(str) {
  const out = {};
  for (const pair of str.split(';')) {
    const idx = pair.indexOf('=');
    if (idx > 0) out[pair.slice(0, idx).trim()] = pair.slice(idx + 1).trim();
  }
  return out;
}
