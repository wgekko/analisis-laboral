// script.js - adaptado para auto-start y cargar imagen embebida
// usa %%BACKGROUND_DATA_URL%% como placeholder que reemplaza app.py

(function () {
  "use strict";

  // tiny compatible shim para reqAnimFrame si hiciera falta
  if (!window.requestAnimationFrame) window.requestAnimationFrame = function (cb) { return setTimeout(cb, 16); };

  // APP
  const App = {
    PARAMS: {
      distortion: {
        strength: 0.15,
        radius: 0.2,
        size: 1,
        edgeWidth: 0.05,
        edgeOpacity: 0.2,
        rimLightIntensity: 0.3,
        rimLightWidth: 0.08,
        chromaticAberration: 0.03,
        reflectionIntensity: 0.3,
        waveDistortion: 0.08,
        waveSpeed: 1.2,
        lensBlur: 0.15,
        clearCenterSize: 0.3,
        followMouse: true,
        animationSpeed: 1,
        overallIntensity: 1
      }
    },

    // core
    backgroundTexture: null,
    backgroundMesh: null,
    backgroundScene: null,
    backgroundCamera: null,
    composer: null,
    renderer: null,
    aspect: 1,
    mousePosition: { x: 0.5, y: 0.5 },
    targetMousePosition: { x: 0.5, y: 0.5 },
    customPass: null,
    isTextureLoaded: false,
    webglSupported: (function () {
      try {
        const c = document.createElement("canvas");
        return !!(window.WebGLRenderingContext && (c.getContext("webgl") || c.getContext("experimental-webgl")));
      } catch (e) { return false; }
    })(),

    init() {
      // arrancamos ya mismo
      this.bindEvents();
      if (!this.webglSupported) {
        document.getElementById("fallbackBg")?.classList.add("active");
        return;
      }
      this.initScene();
      this.startPreloaderAndFinish();
    },

    bindEvents() {
      window.addEventListener("resize", () => this.onWindowResize());
      document.addEventListener("mousemove", (e) => {
        this.targetMousePosition.x = e.clientX / window.innerWidth;
        this.targetMousePosition.y = 1 - e.clientY / window.innerHeight;
      });
    },

    initScene() {
      const canvas = document.getElementById("canvas");
      this.renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.aspect = window.innerWidth / window.innerHeight;

      // background scene & camera
      this.backgroundScene = new THREE.Scene();
      this.backgroundCamera = new THREE.OrthographicCamera(-this.aspect, this.aspect, 1, -1, 0.1, 10);
      this.backgroundCamera.position.z = 1;

      // cargar textura desde la data URL inyectada por Python
      const backgroundDataUrl = "%%BACKGROUND_DATA_URL%%";
      new THREE.TextureLoader().load(
        backgroundDataUrl,
        (tex) => {
          this.backgroundTexture = tex;
          this.createBackgroundMesh();
          this.isTextureLoaded = true;
        },
        undefined,
        () => {
          // error -> mostrar fallback
          document.getElementById("fallbackBg")?.classList.add("active");
          this.isTextureLoaded = true;
        }
      );

      // composer básico para renderizar la escena de fondo
      this.composer = new THREE.EffectComposer(this.renderer);
      const renderPass = new THREE.RenderPass(this.backgroundScene, this.backgroundCamera);
      this.composer.addPass(renderPass);

      // shader pass simple (efecto basado en mouse para dar sensación)
      const vsh = 'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}';
      const fsh = `
        uniform sampler2D tDiffuse;
        uniform vec2 uMouse;
        uniform float uTime;
        varying vec2 vUv;
        void main(){
          vec2 uv = vUv;
          vec2 m = uMouse;
          float d = distance(uv, m);
          float r = 0.18;
          float strength = 0.035;
          float mask = smoothstep(r, r - 0.02, d);
          vec2 offset = normalize(uv - m) * (1.0 - mask) * strength;
          vec4 col = texture2D(tDiffuse, uv + offset);
          gl_FragColor = col;
        }
      `;
      const shaderPass = new THREE.ShaderPass({
        uniforms: {
          tDiffuse: { value: null },
          uMouse: { value: new THREE.Vector2(0.5, 0.5) },
          uTime: { value: 0.0 }
        },
        vertexShader: vsh,
        fragmentShader: fsh
      });
      shaderPass.renderToScreen = true;
      this.customPass = shaderPass;
      this.composer.addPass(this.customPass);

      // arrancar loop
      this.lastTime = performance.now();
      this.animate();
    },

    createBackgroundMesh() {
      if (this.backgroundMesh) this.backgroundScene.remove(this.backgroundMesh);

      const imgAspect = this.backgroundTexture.image.width / this.backgroundTexture.image.height;
      const scAspect = window.innerWidth / window.innerHeight;
      let sx, sy;
      if (scAspect > imgAspect) {
        sx = scAspect * 2;
        sy = sx / imgAspect;
      } else {
        sy = 2;
        sx = sy * imgAspect;
      }
      const g = new THREE.PlaneGeometry(sx, sy);
      const m = new THREE.MeshBasicMaterial({ map: this.backgroundTexture });
      this.backgroundMesh = new THREE.Mesh(g, m);
      this.backgroundScene.add(this.backgroundMesh);
    },

    onWindowResize() {
      this.aspect = window.innerWidth / window.innerHeight;
      if (this.backgroundCamera) {
        this.backgroundCamera.left = -this.aspect;
        this.backgroundCamera.right = this.aspect;
        this.backgroundCamera.updateProjectionMatrix();
      }
      if (this.renderer) {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
      }
      if (this.composer) {
        this.composer.setSize(window.innerWidth, window.innerHeight);
      }
      if (this.backgroundTexture) this.createBackgroundMesh();
    },

    startPreloaderAndFinish() {
      // contador sencillo (simula carga)
      let c = 0;
      const el = document.getElementById("counter");
      const t = setInterval(() => {
        c++;
        if (el) el.textContent = "[" + (c < 10 ? "00" : c < 100 ? "0" : "") + c + "]";
        if (c >= 100) {
          clearInterval(t);
          const pre = document.getElementById("preloader");
          if (pre) pre.style.display = "none";
          // mostrar fondo si fallback
          document.getElementById("fallbackBg")?.classList.add("active");
        }
      }, 18);
    },

    animate(now) {
      requestAnimationFrame((t) => this.animate(t));
      if (!this.composer) return;
      const time = now * 0.001;
      // suavizado del mouse
      this.mousePosition.x += (this.targetMousePosition.x - this.mousePosition.x) * 0.08;
      this.mousePosition.y += (this.targetMousePosition.y - this.mousePosition.y) * 0.08;
      if (this.customPass && this.customPass.uniforms) {
        this.customPass.uniforms.uMouse.value.set(this.mousePosition.x, this.mousePosition.y);
        this.customPass.uniforms.uTime.value = time;
      }
      this.composer.render();
    }
  };

  // arrancar automáticamente al cargar el script
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => App.init());
  } else {
    App.init();
  }

  // Exponer App por si quieres debug
  window.App = App;

})();


  // script del boton de acceso 


