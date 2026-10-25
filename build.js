#!/usr/bin/env node
/* hexo-cli 在本机上下文检测异常,直接走 Hexo API 构建/部署 */
const Hexo = require('hexo');
const cmd = process.argv[2] || 'generate';
const hexo = new Hexo(__dirname, {});
hexo.init()
  .then(() => hexo.call(cmd, cmd === 'deploy' ? { generate: true } : {}))
  .then(() => hexo.exit())
  .catch(e => { console.error(e); hexo.exit(1); });
