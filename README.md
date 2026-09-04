<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TI EdgeAI Model Hub</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #1a1a1a; line-height: 1.6; }
    .page { max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }

    /* Header */
    .hero { background: #c41230; color: white; border-radius: 12px; padding: 2.5rem 2rem; text-align: center; margin-bottom: 2rem; }
    .hero-logo { margin-bottom: 1rem; }
    .hero h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
    .hero p { margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.9; }

    /* Cards */
    .card { background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 1.5rem 1.75rem; margin-bottom: 1.5rem; }
    .card h2 { font-size: 1.15rem; font-weight: 700; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #c41230; display: inline-block; }
    .card p { margin-bottom: 0.75rem; font-size: 0.95rem; }
    .card p:last-child { margin-bottom: 0; }

    .underline { color: #c41230; text-decoration: underline; }
    .underline:hover { color: #9a0e24; }

    /* Feature list */
    .feature-list { list-style: none; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.75rem; }
    .feature-list li { font-size: 0.9rem; padding: 0.4rem 0.75rem; background: #fafafa; border: 1px solid #e8e8e8; border-radius: 6px; }
    .feature-list li::before { content: "✓ "; color: #c41230; font-weight: 700; }

    /* Use case grid */
    .uc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem; }
    .uc-card { background: #c41230; border: 1px solid #c41230; border-radius: 8px; padding: 0.75rem 1.25rem; }
    .uc-card h3 { font-size: 0.9rem; font-weight: 700; color: white; }

    /* Tables */
    table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; font-size: 0.85rem; }
    th { background: #f0f0f0; text-align: left; padding: 0.5rem 0.75rem; border-bottom: 2px solid #ddd; font-weight: 600; }
    td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #eee; vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #fafafa; }

    /* License tags */
    .lic { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 4px; font-size: 0.82rem; font-weight: 600; white-space: nowrap; text-decoration: none; }
    .lic-apache { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
    .lic-agpl   { background: #fce7f3; color: #9d174d; border: 1px solid #f9a8d4; }
    .lic-bsd    { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
    .lic-apache:hover { background: #bfdbfe; }
    .lic-agpl:hover   { background: #fbcfe8; }
    .lic-bsd:hover    { background: #a7f3d0; }

    /* License blocks */
    .lic-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.75rem; justify-content: center; }

    /* Disclaimer */
    .disclaimer { margin-top: 1.25rem; padding: 0.75rem 1rem; background: #fef9ec; border: 1px solid #f59e0b; border-radius: 6px; font-size: 0.82rem; color: #555; }
    .disclaimer strong { color: #1a1a1a; }

    /* Matrix table */
    .matrix-yes { color: #065f46; font-weight: 600; }
    .matrix-no  { color: #9d174d; font-weight: 600; }

    /* Quick start */
    pre { background: #1e1e1e; color: #d4d4d4; padding: 1rem 1.25rem; border-radius: 6px; font-size: 0.82rem; overflow-x: auto; margin-top: 0.5rem; margin-bottom: 1rem; line-height: 1.5; }
    code { font-family: "Fira Code", "Courier New", monospace; }

    /* Compilation side-by-side */
    .tool-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.75rem; }
    .tool-card { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem 1.25rem; }
    .tool-card h3 { font-size: 0.95rem; font-weight: 700; margin-bottom: 0.35rem; }
    .tool-card h3 a { color: #c41230; text-decoration: underline; }
    .tool-card p { font-size: 0.85rem; color: #555; margin: 0; }
    .tool-badge { display: inline-block; font-size: 0.72rem; font-weight: 600; background: #c41230; color: white; border-radius: 3px; padding: 0.1rem 0.4rem; margin-left: 0.4rem; vertical-align: middle; }

    /* Supported hardware chips */
    .hw-list { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.75rem; }
    .hw-item { display: flex; align-items: center; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.88rem; }
    .hw-item strong { color: #c41230; min-width: 70px; margin-right: 0.5rem; }
    .hw-item .hw-aliases { color: #666; font-size: 0.82rem; }
    .hw-item .hw-sep { margin: 0 0.4rem; color: #ccc; }

    /* Resources */
    .res-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 0.75rem; }
    .res-card { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; }
    .res-card h3 { font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem; color: #555; text-transform: uppercase; letter-spacing: 0.04em; }
    .res-card a { display: block; font-size: 0.85rem; color: #c41230; text-decoration: underline; margin-bottom: 0.25rem; }

    /* Footer */
    .footer { text-align: center; font-size: 0.8rem; color: #888; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; }
    .footer a { color: #c41230; text-decoration: underline; margin: 0 0.5rem; }
  </style>
</head>
<body>
<div class="page">

  <!-- Hero -->
  <div class="hero" style="background:#c41230; color:white; border-radius:12px; padding:2.5rem 2rem; text-align:center; margin-bottom:2rem;">
    <div style="margin-bottom:1rem;">
      <img src="docs/assets/TXN-Logo.png" alt="Texas Instruments" width="56" height="56" style="width:56px; height:56px; object-fit:contain; background:white; border-radius:8px; padding:6px;" />
    </div>
    <h1 style="font-size:2rem; font-weight:700; letter-spacing:-0.5px; color:white;">EdgeAI Model Hub</h1>
    <p style="margin-top:0.5rem; font-size:1.05rem; opacity:0.9; color:white;">Pre-trained, hardware-optimized AI models for TI edge devices</p>
  </div>

  <!-- Overview -->
  <div class="card">
    <h2>Overview</h2>
    <p>
      The TI EdgeAI Model Hub is a curated repository of open-source computer vision models
      optimized for deployment on Texas Instruments MPU devices. Models are compiled for TI hardware
      using <a href="https://github.com/TexasInstruments/edgeai-tidlrunner" class="underline">TIDL Runner</a>
      or <a href="https://github.com/TexasInstruments/edgeai-tidl-tools" class="underline">TIDL Tools</a>,
      enabling production-ready inference without cloud dependency.
    </p>
    <ul class="feature-list">
      <li>Hardware-optimized for TI MPU processors</li>
      <li>Benchmarked on real TI hardware</li>
      <li>Format is portable and vendor-neutral</li>
      <li>Automated scripts for model preparation</li>
    </ul>
  </div>

  <!-- Use Cases -->
  <div class="card">
    <h2>Use Cases</h2>
    <div class="uc-grid">
      <div class="uc-card"><h3>Automotive</h3></div>
      <div class="uc-card"><h3>Aerospace &amp; Defense</h3></div>
      <div class="uc-card"><h3>Industrial</h3></div>
      <div class="uc-card"><h3>Surveillance</h3></div>
      <div class="uc-card"><h3>Robotics</h3></div>
      <div class="uc-card"><h3>Edge IoT</h3></div>
    </div>
  </div>

  <!-- License Summary -->
  <div class="card" id="licenses">
    <h2>License Summary</h2>
    <p>Models in this hub are distributed under various open-source licenses - each model's license is indicated in it's own documentation page.</p>
    <div class="disclaimer">
      <strong>Disclaimer:</strong> Certain licenses in this repository impose distribution restrictions that
      may affect commercial, proprietary, or regulated-industry use. It is the sole responsibility of the
      user to review the applicable license terms, assess compatibility with their intended use, and obtain
      any necessary legal clearances prior to use or distribution. Texas Instruments makes no representation
      regarding the suitability of these licenses for any particular purpose and accepts no legal
      responsibility for the user's compliance obligations.
    </div>
  </div>

  <!-- Compilation & Deployment -->
  <div class="card">
    <h2>Compilation &amp; Deployment</h2>
    <div class="tool-grid">
      <div class="tool-card">
        <h3><a href="https://github.com/TexasInstruments/edgeai-tidlrunner" class="underline">TIDL Runner</a> <span class="tool-badge">Recommended</span></h3>
        <p>High-level compilation interface suitable for most users and standard deployment workflows.</p>
      </div>
      <div class="tool-card">
        <h3><a href="https://github.com/TexasInstruments/edgeai-tidl-tools" class="underline">TIDL Tools</a></h3>
        <p>Low-level compilation framework for advanced users and custom workflows.</p>
      </div>
    </div>
  </div>

  <!-- Quick Start -->
  <div class="card">
    <h2>Quick Start</h2>
    <p><strong>1. Clone the repository</strong></p><pre><code>git clone https://github.com/TexasInstruments/edgeai-modelhub.git
cd edgeai-modelhub</code></pre>
    <p><strong>2. Navigate to a model directory and prepare the model</strong></p><pre><code>cd models/vision/&lt;task&gt;/&lt;model&gt;/
python prepare_model.py --model &lt;variant&gt;</code></pre>
    <p><strong>3. Compile for TI hardware (run from inside the egdeai-tidlrunner directory)</strong></p><pre><code>cd /path/to/egdeai-tidlrunner
tidlrunner-cli compile --target_device &lt;device&gt; \
  --config_path /path/to/egdeai-modelhub/&lt;model&gt;_config.yaml</code></pre>
  <p><strong>4. Infer on TI hardware (run from inside the egdeai-tidlrunner directory)</strong></p><pre><code>cd /path/to/egdeai-tidlrunner
tidlrunner-cli infer --target_device &lt;device&gt; \
  --config_path /path/to/egdeai-modelhub/&lt;model&gt;_config.yaml</code></pre>
  </div>

  <!-- Supported Hardware -->
  <div class="card">
    <h2>Supported Hardware</h2>
    <p>
      Compatible TI MPU device families compiled and validated via TIDL. See the supported devices, SDKs and version compatibility at 
      <a href="https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md" class="underline"> EdgeAI developer landing space </a> and <a href="https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/sdk_version_compatibility_table.md" class="underline">edgeai-tidl-tools sdk version compatibility matrix </a>
    </p>
    <div class="hw-list">
      <div class="hw-item"><strong>AM62A</strong></div>
      <div class="hw-item">
        <strong>J722S</strong>
        <span class="hw-aliases">TDA4AEN <span class="hw-sep">·</span> AM67A</span>
      </div>
      <div class="hw-item">
        <strong>J721E</strong>
        <span class="hw-aliases">TDA4VM</span>
      </div>
      <div class="hw-item">
        <strong>J721S2</strong>
        <span class="hw-aliases">TDA4VL <span class="hw-sep"> </span> TDA4AL <span class="hw-sep">·</span> AM68A</span>
      </div>
      <div class="hw-item">
        <strong>J784S4</strong>
        <span class="hw-aliases">TDA4VH <span class="hw-sep"> </span> TDA4AL <span class="hw-sep"> </span> AM69A </span>
      </div>
    </div>
  </div>

  <!-- Model Catalog -->
  <div class="card" id="catalog">
    <h2>Model Catalog</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Capability</th>
          <th>Variants</th>
          <th>Input</th>
          <th>Performance</th>
          <th>License</th>
          <th>Docs</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>MobileNetV3</strong></td>
          <td>Image Classification</td>
          <td>large</td>
          <td>224&times;224</td>
          <td>75.3% Top-1</td>
          <td><a href="https://opensource.org/licenses/BSD-3-Clause" class="lic lic-bsd" target="_blank">BSD-3-Clause</a></td>
          <td><a href="models/vision/classification/MobileNetV3/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>ResNet-50</strong></td>
          <td>Image Classification</td>
          <td>v1.5, v1</td>
          <td>224&times;224</td>
          <td>74.93&ndash;76.15% Top-1</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/classification/ResNet/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>DINO</strong></td>
          <td>Image Classification</td>
          <td>ViT-S/16, ViT-S/8, ViT-B/16, ViT-B/8, ResNet-50</td>
          <td>224&times;224</td>
          <td>75.3&ndash;80.1% Top-1</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/classification/DINO/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>DINOv2</strong></td>
          <td>Image Classification</td>
          <td>ViT-S/14, ViT-B/14 (w/ &amp; w/o registers)</td>
          <td>224&times;224</td>
          <td>80.9&ndash;84.6% Top-1</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/classification/DINOv2/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>ViT</strong></td>
          <td>Image Classification</td>
          <td>vit_b_16, vit_b_32, vit_l_16, vit_l_32</td>
          <td>224&times;224</td>
          <td>75.9&ndash;81.1% Top-1</td>
          <td><a href="https://opensource.org/licenses/BSD-3-Clause" class="lic lic-bsd" target="_blank">BSD-3-Clause</a></td>
          <td><a href="models/vision/classification/ViT/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>ConvNeXt</strong></td>
          <td>Image Classification</td>
          <td>convnext_tiny, convnext_small, convnext_base, convnext_large</td>
          <td>224&times;224</td>
          <td>82.5&ndash;84.4% Top-1</td>
          <td><a href="https://opensource.org/licenses/BSD-3-Clause" class="lic lic-bsd" target="_blank">BSD-3-Clause</a></td>
          <td><a href="models/vision/classification/ConvNeXt/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>DEIMv2</strong></td>
          <td>Object Detection</td>
          <td>s, m</td>
          <td>640&times;640</td>
          <td>50.9&ndash;53.0% mAP</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/detection/DEIMv2/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>DETR</strong></td>
          <td>Object Detection</td>
          <td>detr_resnet50, detr_resnet50_dc5, detr_resnet101, detr_resnet101_dc5</td>
          <td>800&times;800 (flexible)</td>
          <td>AP50:95 42.0&ndash;44.9, AP50 62.4&ndash;64.7</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/detection/DETR/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>Deformable-DETR</strong></td>
          <td>Object Detection</td>
          <td>single-scale</td>
          <td>800&times;800</td>
          <td>AP50:95 39.4%</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/detection/Deformable-DETR/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>RF-DETR</strong></td>
          <td>Object Detection</td>
          <td>nano, s, m, l</td>
          <td>384&ndash;704px</td>
          <td>48.4&ndash;56.5% mAP</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/detection/RF-DETR/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>RT-DETRv2</strong></td>
          <td>Object Detection</td>
          <td>s, ms, m, l, x</td>
          <td>640&times;640</td>
          <td>48.1&ndash;54.3% mAP</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/detection/RT-DETRv2/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>RTMDet</strong></td>
          <td>Object Detection</td>
          <td>tiny, s, m, l, x</td>
          <td>640&times;640</td>
          <td>40.9&ndash;52.8% mAP</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/detection/RTMDet/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>YOLO11</strong></td>
          <td>Object Detection</td>
          <td>n, s, m, l, x</td>
          <td>640&times;640</td>
          <td>39.5&ndash;54.7% mAP</td>
          <td><a href="https://www.gnu.org/licenses/agpl-3.0.html" class="lic lic-agpl" target="_blank">AGPL 3.0</a></td>
          <td><a href="models/vision/detection/YOLO11/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>YOLO26</strong></td>
          <td>Object Detection</td>
          <td>n, s, m, l, x</td>
          <td>640&times;640</td>
          <td>40.9&ndash;57.5% mAP</td>
          <td><a href="https://www.gnu.org/licenses/agpl-3.0.html" class="lic lic-agpl" target="_blank">AGPL 3.0</a></td>
          <td><a href="models/vision/detection/YOLO26/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>YOLOv8</strong></td>
          <td>Object Detection</td>
          <td>n, m</td>
          <td>640&times;640</td>
          <td>37.3&ndash;50.2% mAP</td>
          <td><a href="https://www.gnu.org/licenses/agpl-3.0.html" class="lic lic-agpl" target="_blank">AGPL 3.0</a></td>
          <td><a href="models/vision/detection/YOLOv8/" class="underline">View</a></td>
        </tr>
        <tr>
          <td><strong>YOLOX</strong></td>
          <td>Object Detection</td>
          <td>nano, tiny, m, l, x, darknet53</td>
          <td>416&times;416 / 640&times;640</td>
          <td>24.8&ndash;51.2% mAP</td>
          <td><a href="https://www.apache.org/licenses/LICENSE-2.0" class="lic lic-apache" target="_blank">Apache 2.0</a></td>
          <td><a href="models/vision/detection/YOLOX/" class="underline">View</a></td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Resources -->
  <div class="card">
    <h2>Resources &amp; Links</h2>
    <div class="res-grid">
      <div class="res-card">
        <h3>Ecosystem</h3>
        <a href="https://github.com/TexasInstruments/edgeai" class="underline">TI EdgeAI</a>
        <a href="https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md" class="underline">EdgeAI SDK</a>
      </div>
      <div class="res-card">
        <h3>Tools</h3>
        <a href="https://github.com/TexasInstruments/edgeai-tidlrunner" class="underline">TIDL Runner</a>
        <a href="https://github.com/TexasInstruments/edgeai-tidl-tools" class="underline">TIDL Tools</a>
      </div>
      <div class="res-card">
        <h3>Community</h3>
        <a href="https://e2e.ti.com/support/processors-group/processors/f/processors-forum" class="underline">E2E Support</a>
        <a href="https://github.com/TexasInstruments/edgeai/issues" class="underline">Issues</a>
        <a href="https://github.com/TexasInstruments/edgeai/discussions" class="underline">Discussions</a>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    <p>Maintained by Texas Instruments EdgeAI Team &nbsp;|&nbsp; Last Updated August 2026</p>
    <p style="margin-top:0.4rem;">
      <a href="https://github.com/TexasInstruments/edgeai/issues">Issues</a>
      <a href="https://github.com/TexasInstruments/edgeai/discussions">Discussions</a>
      <a href="mailto:edgeai-dev@list.ti.com">Contact</a>
    </p>
  </div>

</div>
</body>
</html>
