# Miracle Techlink 博客源码(Hexo)

线上 `main` 分支是 `hexo deploy` 的产物;**本分支(source)是唯一真源**。

## 目录

- `source/_posts/*.md` — 文章(带 `abbrlink` front-matter,决定 URL,勿删改)
- `_config.yml` — 站点配置(permalink: posts/:abbrlink.html)
- `_config.butterfly.yml` — 主题覆盖(数学 CSS 注入、特效、live2d)
- `vendor/katex-mdit-shim` — markdown-it KaTeX 插件的 CJS shim
- `build.js` — 构建/部署入口(绕过 hexo-cli 的上下文检测问题)

## 两个必须知道的坑

1. **package.json 必须保留 `"hexo": {"version": "..."}` 段**:hexo 7 的 `update_package`
   只在存在该段时才置 `env.init=true`,否则配置/主题/插件全部静默不加载(症状:文章数为 0、无报错)。
2. **公式是服务端渲染**:`hexo-renderer-markdown-it` + `@vscode/markdown-it-katex`(经 shim 引入)。
   不要换回 `hexo-renderer-marked`——它会把 `$$` 块拆成多个文本节点、并把公式内的 `_` 吃成斜体。

## 常用命令

```bash
node build.js generate   # 构建 → public/
node build.js deploy     # 构建 + 推送 main(线上立即更新)
node build.js server     # 本地预览(hexo server 语义)
```

## 部署流程

改 md → `node build.js generate` → 本地 public/ 抽查 → `node build.js deploy`。
部署用的 SSH key 与仓库权限沿用 `hexo-deployer-git` 配置(_config.yml deploy 段)。
