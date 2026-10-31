/* 动态 3D 粒子波场背景(three.js)
 * 特性:粒子波场 + 缓慢漂移 + 鼠标视差;移动端/低分屏自动降粒子;WebGL 失败静默退出(静态背景图兜底)
 */
(function () {
  if (document.documentElement.dataset.theme === 'dark') {
    // 深色模式走暗色配色
  }
  var isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  try {
    var canvas = document.createElement('canvas');
    if (!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))) return;
  } catch (e) { return; }

  // 动态加载 three.js
  var s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
  s.onload = init;
  document.head.appendChild(s);

  function init() {
    var THREE = window.THREE;
    if (!THREE) return;

    var container = document.createElement('div');
    container.id = 'bg3d';
    container.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:-998;pointer-events:none;opacity:0;transition:opacity 1.2s ease;';
    document.body.appendChild(container);

    var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 3000);
    camera.position.set(0, 90, 320);
    camera.lookAt(0, 0, 0);

    // ---- 粒子波场 ----
    var COLS = isMobile ? 70 : 120;
    var ROWS = isMobile ? 50 : 90;
    var SPACING = 8;
    var count = COLS * ROWS;
    var positions = new Float32Array(count * 3);
    var colors = new Float32Array(count * 3);

    // 主题色:青蓝 → 紫(与 physical ai 的科技感一致)
    var c1 = new THREE.Color(0x39c5bb); // 青
    var c2 = new THREE.Color(0x5b8cff); // 蓝
    var c3 = new THREE.Color(0xa06bff); // 紫

    var idx = 0;
    for (var i = 0; i < COLS; i++) {
      for (var j = 0; j < ROWS; j++) {
        var x = (i - COLS / 2) * SPACING;
        var z = (j - ROWS / 2) * SPACING;
        positions[idx * 3] = x;
        positions[idx * 3 + 1] = 0;
        positions[idx * 3 + 2] = z;

        // 颜色按距离中心渐变
        var t = Math.sqrt((i - COLS / 2) ** 2 + (j - ROWS / 2) ** 2) / Math.sqrt((COLS / 2) ** 2 + (ROWS / 2) ** 2);
        var c = t < 0.5 ? c1.clone().lerp(c2, t * 2) : c2.clone().lerp(c3, (t - 0.5) * 2);
        c.multiplyScalar(0.55 + 0.45 * Math.sin(t * Math.PI)); // 边缘稍暗
        colors[idx * 3] = c.r;
        colors[idx * 3 + 1] = c.g;
        colors[idx * 3 + 2] = c.b;
        idx++;
      }
    }

    var geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    var material = new THREE.PointsMaterial({
      size: 2.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    var points = new THREE.Points(geometry, material);
    scene.add(points);

    // ---- 漂浮的细线连接层(增加深度感) ----
    var lineGroup = new THREE.Group();
    var lineMat = new THREE.LineBasicMaterial({ color: 0x5b8cff, transparent: true, opacity: 0.12 });
    for (var k = 0; k < (isMobile ? 8 : 16); k++) {
      var lg = new THREE.BufferGeometry();
      var lpts = new Float32Array(6);
      lpts[0] = (Math.random() - 0.5) * 800;
      lpts[1] = 20 + Math.random() * 120;
      lpts[2] = (Math.random() - 0.5) * 800;
      lpts[3] = lpts[0] + (Math.random() - 0.5) * 200;
      lpts[4] = lpts[1] + 30 + Math.random() * 80;
      lpts[5] = lpts[2] + (Math.random() - 0.5) * 200;
      lg.setAttribute('position', new THREE.BufferAttribute(lpts, 3));
      lineGroup.add(new THREE.Line(lg, lineMat));
    }
    scene.add(lineGroup);

    // ---- 动画 ----
    var mouse = { x: 0, y: 0, tx: 0, ty: 0 };
    window.addEventListener('mousemove', function (e) {
      mouse.tx = (e.clientX / window.innerWidth - 0.5) * 2;
      mouse.ty = (e.clientY / window.innerHeight - 0.5) * 2;
    }, { passive: true });

    var clock = new THREE.Clock();
    var time = 0;
    var pos = geometry.attributes.position.array;

    function animate() {
      requestAnimationFrame(animate);
      time += clock.getDelta();
      var t = time * 0.8;

      // 多组正弦叠加波场
      for (var n = 0; n < count; n++) {
        var x = pos[n * 3];
        var z = pos[n * 3 + 2];
        pos[n * 3 + 1] =
          Math.sin((x + t * 40) * 0.02) * 14 +
          Math.sin((z * 0.7 - t * 26) * 0.03) * 10 +
          Math.sin((x + z) * 0.012 + t * 0.9) * 8;
      }
      geometry.attributes.position.needsUpdate = true;

      // 鼠标视差(相机缓动)
      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;
      camera.position.x = mouse.x * 60;
      camera.position.y = 90 - mouse.y * 36;
      camera.lookAt(0, 0, 0);

      // 波场缓慢整体旋转 + 线层反向漂移
      points.rotation.y = t * 0.05;
      lineGroup.rotation.y = -t * 0.03;

      renderer.render(scene, camera);
    }
    animate();

    // 淡入
    requestAnimationFrame(function () { container.style.opacity = '1'; });

    // 自适应
    var rt;
    window.addEventListener('resize', function () {
      clearTimeout(rt);
      rt = setTimeout(function () {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }, 200);
    });

    // 页面不可见时暂停(省电)
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) clock.stop(); else clock.start();
    });
  }
})();
