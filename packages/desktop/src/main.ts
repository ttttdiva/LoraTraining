import { open } from "@tauri-apps/api/dialog";
import { convertFileSrc, invoke } from "@tauri-apps/api/tauri";
import "./styles.css";

type BridgeEnvelope<T> = {
  ok: boolean;
  bridge: { status: "ok" | "error"; job: string; data?: T; warnings?: string[]; errors?: string[] };
  stderr: string;
};

type ToolStatus = { name: string; available: boolean; version?: string; path?: string; error?: string };
type GpuInfo = { index: number; name: string; memoryTotalMiB?: number; driverVersion?: string; temperatureC?: number; utilizationPct?: number };
type EngineCandidate = { id: string; name: string; type: string; root: string; available: boolean; notes: string[] };
type HealthData = {
  app: { projectRoot: string; bridgeRoot: string };
  system: { platform: string; pythonVersion: string; executable: string; cwd: string };
  tools: ToolStatus[];
  gpu: { available: boolean; gpus: GpuInfo[]; error?: string };
  engines: EngineCandidate[];
};

type Settings = {
  jobsRoot: string;
  defaultEngineId: string;
  taggerModelDir: string;
  engines: Array<{ id: string; name: string; type: string; root: string; venv: string }>;
  defaults: Record<string, unknown>;
};

type DatasetItem = {
  imagePath: string;
  captionPath: string;
  relativePath: string;
  captionExists: boolean;
  captionText: string;
  tagCount: number;
  width?: number | null;
  height?: number | null;
  issues: string[];
};
type OrphanCaption = { path: string; relativePath: string; text: string };
type DatasetScan = {
  root: string;
  captionExtension: string;
  imageCount: number;
  captionCount: number;
  shownItemCount: number;
  truncated: boolean;
  missingCaptionCount: number;
  emptyCaptionCount: number;
  orphanCaptionCount: number;
  issueCount: number;
  items: DatasetItem[];
  orphanCaptions: OrphanCaption[];
};
type SaveCaptionResult = { captionPath: string; captionText: string; tags: string[]; tagCount: number };
type TagOperationResult = { root: string; changedCount: number; skippedCount: number; changed: SaveCaptionResult[] };

type JobSummary = {
  name: string;
  displayName: string;
  engineId: string;
  architecture: string;
  datasetImageDir: string;
  updatedAt: string;
  path: string;
};

type Job = Record<string, any>;
type LaunchPlan = {
  id: string;
  kind: string;
  cwd: string;
  command: string;
  args: string[];
  env: Record<string, string>;
  displayCommand: string;
  files?: Record<string, string>;
  url?: string;
};
type ProcessStatus = {
  kind?: string;
  command?: string;
  running: boolean;
  exitCode?: number | null;
  logs: string[];
};
type ProcessEnvelope = { id: string; status: ProcessStatus };

type ViewId = "dashboard" | "dataset" | "jobs" | "tagger";

type DatasetState = {
  root: string;
  captionExtension: string;
  minPixels: number;
  loading: boolean;
  scan?: DatasetScan;
  selectedImagePath?: string;
  draftCaption: string;
  message?: string;
  error?: string;
  operation: "add" | "remove" | "replace" | "move_front" | "shuffle";
  tag: string;
  replacement: string;
  keepTokens: number;
  includeMissing: boolean;
  caseSensitive: boolean;
};

const state: {
  view: ViewId;
  health?: HealthData;
  settings?: Settings;
  healthLoading: boolean;
  healthError?: string;
  stderr?: string;
  dataset: DatasetState;
  jobs: JobSummary[];
  selectedJobName?: string;
  jobDraft?: Job;
  jobMessage?: string;
  jobError?: string;
  plans: Record<string, LaunchPlan | undefined>;
  processes: Record<string, ProcessStatus | undefined>;
  tagger: {
    modelDir: string;
    datasetRoot: string;
    threshold: number;
    characterThreshold: number;
    mode: "merge" | "overwrite";
    message?: string;
    error?: string;
    modelStatus?: Record<string, unknown>;
  };
} = {
  view: "dashboard",
  healthLoading: true,
  dataset: {
    root: "",
    captionExtension: ".txt",
    minPixels: 0,
    loading: false,
    draftCaption: "",
    operation: "remove",
    tag: "",
    replacement: "",
    keepTokens: 1,
    includeMissing: false,
    caseSensitive: false,
  },
  jobs: [],
  plans: {},
  processes: {},
  tagger: {
    modelDir: "",
    datasetRoot: "",
    threshold: 0.35,
    characterThreshold: 0.35,
    mode: "merge",
  },
};

const appRoot = document.querySelector<HTMLDivElement>("#app");
if (!appRoot) throw new Error("App root not found");
const rootElement = appRoot;

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function statusPill(ok: boolean, okLabel = "OK", badLabel = "Check"): string {
  return `<span class="pill ${ok ? "pill-ok" : "pill-warn"}">${ok ? okLabel : badLabel}</span>`;
}

function formatMiB(value?: number): string {
  if (!value) return "-";
  return value >= 1024 ? `${(value / 1024).toFixed(1)} GB` : `${value} MB`;
}

function getPath(obj: any, path: string, fallback: unknown = ""): any {
  return path.split(".").reduce((acc, key) => (acc && acc[key] !== undefined ? acc[key] : undefined), obj) ?? fallback;
}

function setPath(obj: any, path: string, value: unknown): void {
  const parts = path.split(".");
  let target = obj;
  for (const part of parts.slice(0, -1)) {
    target[part] = target[part] || {};
    target = target[part];
  }
  target[parts[parts.length - 1]] = value;
}

async function runBridge<T>(job: string, payload: Record<string, unknown> = {}): Promise<BridgeEnvelope<T>> {
  return invoke<BridgeEnvelope<T>>("run_bridge", { job, payload });
}

async function bridgeData<T>(job: string, payload: Record<string, unknown> = {}): Promise<T> {
  const result = await runBridge<T>(job, payload);
  state.stderr = result.stderr;
  if (!result.ok || result.bridge.status !== "ok" || result.bridge.data === undefined) {
    throw new Error(result.bridge.errors?.join("\n") || `Bridge job failed: ${job}`);
  }
  return result.bridge.data;
}

async function startProcess(plan: LaunchPlan): Promise<void> {
  const result = await invoke<ProcessEnvelope>("start_process", { plan });
  state.processes[plan.id] = result.status;
  render();
}

async function stopProcess(id: string): Promise<void> {
  await invoke("stop_process", { id });
  await refreshProcess(id);
}

async function refreshProcess(id: string): Promise<void> {
  const result = await invoke<ProcessEnvelope>("process_status", { id });
  state.processes[id] = result.status;
}

function renderTool(tool: ToolStatus): string {
  const detail = tool.available ? [tool.version, tool.path].filter(Boolean).join(" / ") : tool.error || "not found";
  return `<tr><td>${escapeHtml(tool.name)}</td><td>${statusPill(tool.available, "Found", "Missing")}</td><td class="muted mono">${escapeHtml(detail)}</td></tr>`;
}

function renderDashboard(): string {
  if (state.healthLoading) return `<section class="panel"><p class="muted">Checking local environment...</p></section>`;
  if (state.healthError || !state.health) {
    return `<section class="panel"><h2>Bridge Error</h2><p class="error-text">${escapeHtml(state.healthError)}</p></section>`;
  }
  const health = state.health;
  const gpuCards = health.gpu.gpus.length
    ? health.gpu.gpus.map((gpu) => `
      <article class="card compact">
        <div class="card-header"><div><p class="eyebrow">GPU ${gpu.index}</p><h3>${escapeHtml(gpu.name)}</h3></div>${statusPill(true, "Ready")}</div>
        <dl class="metrics">
          <div><dt>VRAM</dt><dd>${formatMiB(gpu.memoryTotalMiB)}</dd></div>
          <div><dt>Driver</dt><dd>${escapeHtml(gpu.driverVersion || "-")}</dd></div>
          <div><dt>Temp</dt><dd>${gpu.temperatureC ?? "-"} C</dd></div>
          <div><dt>Util</dt><dd>${gpu.utilizationPct ?? "-"}%</dd></div>
        </dl>
      </article>`).join("")
    : `<article class="card compact"><h3>No GPU detected</h3><p>${escapeHtml(health.gpu.error || "")}</p></article>`;

  return `
    <section class="hero-band">
      <div>
        <p class="eyebrow">Desktop LoRA Training GUI</p>
        <h1>LoraTraining</h1>
        <p class="lead">Dataset editing, job TOML generation, training launch, samples, TensorBoard, and WD14 tagging are wired through local processes.</p>
      </div>
      <div class="action-row">
        <button class="secondary-button" id="refresh-health" type="button">Refresh</button>
        <button class="primary-button" data-view="jobs" type="button">Jobs</button>
      </div>
    </section>
    <section class="grid two">
      <article class="panel">
        <div class="section-title"><h2>System</h2>${statusPill(true, "Bridge OK")}</div>
        <dl class="details">
          <div><dt>Python</dt><dd class="mono">${escapeHtml(health.system.pythonVersion)}</dd></div>
          <div><dt>Project</dt><dd class="mono">${escapeHtml(health.app.projectRoot)}</dd></div>
          <div><dt>Jobs</dt><dd class="mono">${escapeHtml(state.settings?.jobsRoot || "-")}</dd></div>
          <div><dt>Tagger</dt><dd class="mono">${escapeHtml(state.settings?.taggerModelDir || "-")}</dd></div>
        </dl>
      </article>
      <article class="panel">
        <div class="section-title"><h2>Implemented Workflows</h2><span class="pill">active</span></div>
        <ol class="steps">
          <li>Dataset Studio: caption edit and bulk tag operations.</li>
          <li>Jobs: create, edit, save, clone, delete, TOML generation.</li>
          <li>Run: training, sample generation, TensorBoard, process stop/logs.</li>
          <li>Tagger: dependency install, model download, WD14 tagging process.</li>
        </ol>
      </article>
    </section>
    <section class="panel"><div class="section-title"><h2>GPU</h2>${statusPill(health.gpu.available, "Found", "Missing")}</div><div class="grid cards">${gpuCards}</div></section>
    <section class="panel">
      <div class="section-title"><h2>Tools</h2><span class="pill">preflight</span></div>
      <table><thead><tr><th>Tool</th><th>Status</th><th>Detail</th></tr></thead><tbody>${health.tools.map(renderTool).join("")}</tbody></table>
    </section>`;
}

function selectedDatasetItem(): DatasetItem | undefined {
  return state.dataset.scan?.items.find((item) => item.imagePath === state.dataset.selectedImagePath);
}

function imageSizeLabel(item: DatasetItem): string {
  return item.width && item.height ? `${item.width} x ${item.height}` : "-";
}

function renderDatasetStudio(): string {
  const d = state.dataset;
  const scan = d.scan;
  const selected = selectedDatasetItem();
  return `
    <section class="hero-band">
      <div><p class="eyebrow">Dataset Studio</p><h1>Caption and Tag Editor</h1><p class="lead">Scan folders, edit captions, and apply batch tag operations.</p></div>
      <div class="action-row"><button class="secondary-button" id="choose-dataset-root" type="button">Choose Folder</button><button class="primary-button" id="scan-dataset" type="button" ${d.root ? "" : "disabled"}>Scan</button></div>
    </section>
    <section class="panel">
      <div class="form-grid dataset-controls">
        <label>Dataset root<input id="dataset-root" value="${escapeHtml(d.root)}" placeholder="D:\\datasets\\my_lora"></label>
        <label>Caption extension<input id="caption-extension" value="${escapeHtml(d.captionExtension)}"></label>
        <label>Low-res threshold pixels<input id="min-pixels" type="number" min="0" step="1" value="${d.minPixels}"></label>
      </div>
      ${d.loading ? `<p class="muted">Working...</p>` : ""}${d.message ? `<p class="success-text">${escapeHtml(d.message)}</p>` : ""}${d.error ? `<p class="error-text">${escapeHtml(d.error)}</p>` : ""}
    </section>
    ${scan ? `
      <section class="grid cards summary-grid">
        <article class="card compact"><p class="eyebrow">Images</p><h3>${scan.imageCount}</h3></article>
        <article class="card compact"><p class="eyebrow">Captions</p><h3>${scan.captionCount}</h3></article>
        <article class="card compact"><p class="eyebrow">Missing</p><h3>${scan.missingCaptionCount}</h3></article>
        <article class="card compact"><p class="eyebrow">Orphans</p><h3>${scan.orphanCaptionCount}</h3></article>
      </section>
      <section class="panel">
        <div class="section-title"><h2>Bulk Tag Tools</h2><span class="pill">all captions</span></div>
        <div class="form-grid">
          <label>Operation<select id="bulk-operation">
            ${["add", "remove", "replace", "move_front", "shuffle"].map((op) => `<option value="${op}" ${d.operation === op ? "selected" : ""}>${op}</option>`).join("")}
          </select></label>
          <label>Tag<input id="bulk-tag" value="${escapeHtml(d.tag)}"></label>
          <label>Replacement<input id="bulk-replacement" value="${escapeHtml(d.replacement)}"></label>
          <label>Keep tokens<input id="bulk-keep-tokens" type="number" min="0" value="${d.keepTokens}"></label>
          <label class="check-row"><input id="bulk-include-missing" type="checkbox" ${d.includeMissing ? "checked" : ""}>Create missing caption files</label>
          <label class="check-row"><input id="bulk-case-sensitive" type="checkbox" ${d.caseSensitive ? "checked" : ""}>Case sensitive</label>
        </div>
        <div class="action-row"><button class="primary-button" id="apply-bulk-operation" type="button">Apply Bulk Operation</button></div>
      </section>
      <section class="dataset-workspace">
        <div class="panel dataset-list-panel">
          <div class="section-title"><h2>Images</h2><span class="pill">${scan.shownItemCount}</span></div>
          <div class="dataset-list">
            ${scan.items.map((item) => {
              const issues = item.issues.length ? item.issues.map((issue) => `<span class="issue-pill">${escapeHtml(issue)}</span>`).join("") : `<span class="issue-pill issue-ok">ok</span>`;
              return `<button class="dataset-row ${item.imagePath === d.selectedImagePath ? "selected" : ""}" type="button" data-image-path="${escapeHtml(item.imagePath)}">
                <img src="${escapeHtml(convertFileSrc(item.imagePath))}" alt="" loading="lazy">
                <span class="dataset-row-main"><strong>${escapeHtml(item.relativePath)}</strong><span>${escapeHtml(imageSizeLabel(item))} / ${item.tagCount} tags</span><span class="issue-list">${issues}</span></span>
              </button>`;
            }).join("")}
          </div>
        </div>
        <section class="panel editor-panel">
          ${selected ? `
            <div class="section-title"><div><h2>Caption Editor</h2><p class="muted mono">${escapeHtml(selected.captionPath)}</p></div>${statusPill(selected.captionExists, "Exists", "Will create")}</div>
            <div class="preview-block"><img src="${escapeHtml(convertFileSrc(selected.imagePath))}" alt=""><dl class="details"><div><dt>Image</dt><dd>${escapeHtml(selected.relativePath)}</dd></div><div><dt>Size</dt><dd>${escapeHtml(imageSizeLabel(selected))}</dd></div><div><dt>Tags</dt><dd>${selected.tagCount}</dd></div></dl></div>
            <textarea id="caption-editor" spellcheck="false">${escapeHtml(d.draftCaption)}</textarea>
            <div class="action-row"><button class="primary-button" id="save-caption" type="button">Save Caption</button><button class="secondary-button" id="reload-dataset" type="button">Rescan</button></div>
          ` : `<h2>Caption Editor</h2><p class="muted">Select an image.</p>`}
        </section>
      </section>
      ${scan.orphanCaptions.length ? `<section class="panel"><h2>Orphan Caption Files</h2><div class="orphan-list">${scan.orphanCaptions.map((caption) => `<p class="mono">${escapeHtml(caption.relativePath)}</p>`).join("")}</div></section>` : ""}
    ` : `<section class="panel"><p class="muted">Choose a dataset folder to start.</p></section>`}`;
}

function input(path: string, label: string, type = "text", placeholder = ""): string {
  const value = getPath(state.jobDraft, path, "");
  return `<label>${label}<input data-job-field="${path}" type="${type}" value="${escapeHtml(value)}" placeholder="${escapeHtml(placeholder)}"></label>`;
}

function checkbox(path: string, label: string): string {
  const value = Boolean(getPath(state.jobDraft, path, false));
  return `<label class="check-row"><input data-job-field="${path}" type="checkbox" ${value ? "checked" : ""}>${label}</label>`;
}

function select(path: string, label: string, values: string[]): string {
  const value = String(getPath(state.jobDraft, path, values[0]));
  return `<label>${label}<select data-job-field="${path}">${values.map((item) => `<option value="${item}" ${item === value ? "selected" : ""}>${item}</option>`).join("")}</select></label>`;
}

function textarea(path: string, label: string, placeholder = ""): string {
  const value = getPath(state.jobDraft, path, "");
  return `<label class="textarea-label">${label}<textarea data-job-field="${path}" placeholder="${escapeHtml(placeholder)}">${escapeHtml(value)}</textarea></label>`;
}

function renderLogs(processId: string): string {
  const process = state.processes[processId];
  if (!process) return `<pre class="log-box">No process started.</pre>`;
  return `<pre class="log-box">${escapeHtml((process.logs || []).slice(-300).join("\n"))}</pre>`;
}

function renderJobEditor(): string {
  if (!state.jobDraft) return `<section class="panel"><p class="muted">Select or create a job.</p></section>`;
  const job = state.jobDraft;
  const trainId = `train:${job.name}`;
  const tbId = `tensorboard:${job.name}`;
  const sampleId = `sample:${job.name}`;
  return `
    <section class="panel">
      <div class="section-title"><h2>Job Editor</h2><span class="pill">${escapeHtml(job.name)}</span></div>
      <div class="form-grid">
        ${input("displayName", "Display name")}
        <label>Job id<input type="text" value="${escapeHtml(job.name)}" readonly></label>
        ${select("engineId", "Engine", (state.settings?.engines || []).map((engine) => engine.id))}
        ${select("architecture", "Architecture", ["anima", "sd15", "sd2", "sdxl"])}
        ${input("modelPaths.baseModelPath", "sd-scripts base model path")}
        ${input("modelPaths.ditPath", "DiT model path")}
        ${input("modelPaths.qwen3Path", "Qwen3 text encoder path")}
        ${input("modelPaths.vaePath", "VAE path")}
        ${input("dataset.imageDir", "Dataset image dir")}
        ${input("dataset.captionExtension", "Caption extension")}
        ${input("dataset.resolution.0", "Resolution W", "number")}
        ${input("dataset.resolution.1", "Resolution H", "number")}
        ${input("dataset.batchSize", "Batch size", "number")}
        ${input("dataset.numRepeats", "Repeats", "number")}
        ${input("training.outputName", "Output name")}
        ${input("training.maxTrainSteps", "Max steps", "number")}
        ${input("training.maxTrainEpochs", "Epochs", "number")}
        ${input("training.saveEveryNEpochs", "Save every epochs", "number")}
        ${input("training.saveEveryNSteps", "Save every steps", "number")}
        ${input("training.learningRate", "Learning rate", "number")}
        ${input("training.unetLr", "UNet LR", "number")}
        ${input("training.textEncoderLr", "Text encoder LR", "number")}
        ${input("training.optimizerType", "Optimizer")}
        ${input("training.lrScheduler", "LR scheduler")}
        ${input("training.lrWarmupSteps", "LR warmup steps", "number")}
        ${select("training.mixedPrecision", "Mixed precision", ["bf16", "fp16", "no"])}
        ${input("training.clipSkip", "Clip skip", "number")}
        ${input("network.module", "Network module")}
        ${input("network.dim", "Network dim", "number")}
        ${input("network.alpha", "Network alpha", "number")}
        ${input("sample.steps", "Sample steps", "number")}
        ${input("sample.sampler", "Sample sampler")}
        ${input("sample.scale", "Sample CFG scale", "number")}
        ${input("gpu.ids", "GPU IDs")}
        ${select("gpu.mode", "Multi GPU mode", ["single", "ddp", "fsdp", "fsdp2", "deepspeed"])}
        ${select("wandb.mode", "WanDB", ["disabled", "offline", "online"])}
        ${input("wandb.project", "WanDB project")}
        ${checkbox("training.gradientCheckpointing", "Gradient checkpointing")}
        ${checkbox("training.cacheLatentsToDisk", "Cache latents to disk")}
        ${checkbox("training.cacheTextEncoderOutputsToDisk", "Cache text encoder outputs to disk")}
        ${checkbox("training.sdpa", "SDPA")}
        ${checkbox("training.xformers", "xFormers")}
        ${checkbox("network.trainUnetOnly", "Train UNet only")}
        ${checkbox("sdScripts.v2", "SD v2")}
        ${checkbox("sdScripts.vParameterization", "v-parameterization")}
      </div>
      <div class="grid two">
        ${textarea("training.optimizerArgs", "Optimizer args", "weight_decay=0.01")}
        ${textarea("network.args", "Network args", "conv_dim=4\\nconv_alpha=1")}
      </div>
      <label class="textarea-label">Sample prompts<textarea id="job-sample-prompts">${escapeHtml(getPath(job, "sample.prompts", ""))}</textarea></label>
      <div class="grid two">
        ${textarea("sdScripts.extraArgs", "Extra sd-scripts train args", "--min_snr_gamma=5\\n--noise_offset=0.05")}
        ${textarea("sample.extraArgs", "Extra sd-scripts sample args", "--images_per_prompt 2")}
      </div>
      <div class="action-row">
        <button class="primary-button" id="save-job" type="button">Save Job</button>
        <button class="secondary-button" id="build-job-files" type="button">Write TOML</button>
        <button class="secondary-button" id="clone-job" type="button">Clone</button>
        <button class="danger-button" id="delete-job" type="button">Delete</button>
      </div>
      ${state.jobMessage ? `<p class="success-text">${escapeHtml(state.jobMessage)}</p>` : ""}${state.jobError ? `<p class="error-text">${escapeHtml(state.jobError)}</p>` : ""}
    </section>
    <section class="panel">
      <div class="section-title"><h2>Run</h2><span class="pill">launch plans and logs</span></div>
      <div class="action-row">
        <button class="primary-button" id="plan-train" type="button">Plan Training</button>
        <button class="primary-button" id="start-train" type="button" ${state.plans.train ? "" : "disabled"}>Start Training</button>
        <button class="secondary-button" data-stop-process="${trainId}" type="button">Stop Training</button>
        <button class="secondary-button" id="plan-tensorboard" type="button">Plan TensorBoard</button>
        <button class="secondary-button" id="start-tensorboard" type="button" ${state.plans.tensorboard ? "" : "disabled"}>Start TensorBoard</button>
        ${state.plans.tensorboard?.url ? `<a class="link-button" href="${escapeHtml(state.plans.tensorboard.url)}" target="_blank">Open TensorBoard</a>` : ""}
        <button class="secondary-button" id="plan-sample" type="button">Plan Sample</button>
        <button class="secondary-button" id="start-sample" type="button" ${state.plans.sample ? "" : "disabled"}>Generate Sample</button>
      </div>
      <div class="grid two">
        <div><h3>Training</h3>${state.plans.train ? `<pre>${escapeHtml(state.plans.train.displayCommand)}</pre>` : ""}${renderLogs(trainId)}</div>
        <div><h3>TensorBoard / Sample</h3>${state.plans.tensorboard ? `<pre>${escapeHtml(state.plans.tensorboard.displayCommand)}</pre>` : ""}${state.plans.sample ? `<pre>${escapeHtml(state.plans.sample.displayCommand)}</pre>` : ""}${renderLogs(tbId)}${renderLogs(sampleId)}</div>
      </div>
    </section>`;
}

function renderJobs(): string {
  return `
    <section class="hero-band">
      <div><p class="eyebrow">Jobs</p><h1>Training Jobs</h1><p class="lead">Create jobs, write TOML, launch training, start TensorBoard, and run samples.</p></div>
      <div class="action-row"><input id="new-job-name" class="inline-input" placeholder="my_lora"><button class="primary-button" id="create-job" type="button">Create Job</button></div>
    </section>
    <section class="jobs-layout">
      <div class="panel job-list-panel">
        <div class="section-title"><h2>Jobs</h2><button class="secondary-button" id="reload-jobs" type="button">Reload</button></div>
        <div class="job-list">${state.jobs.map((job) => `<button class="job-row ${job.name === state.selectedJobName ? "selected" : ""}" data-job-name="${escapeHtml(job.name)}" type="button"><strong>${escapeHtml(job.displayName)}</strong><span>${escapeHtml(job.datasetImageDir || "no dataset")}</span><span class="mono">${escapeHtml(job.updatedAt || "")}</span></button>`).join("")}</div>
      </div>
      ${renderJobEditor()}
    </section>`;
}

function renderTagger(): string {
  const status = state.tagger.modelStatus || {};
  const installId = "tagger:install_deps";
  const downloadId = "tagger:download";
  const runId = `tagger:${(state.tagger.datasetRoot.split(/[\\/]/).pop() || "dataset").replaceAll(" ", "_")}`;
  return `
    <section class="hero-band">
      <div><p class="eyebrow">WD14 Tagger</p><h1>Auto Tagging</h1><p class="lead">Installs runtime dependencies, downloads WD14 model files, and writes tags into caption txt files.</p></div>
      <div class="action-row"><button class="secondary-button" id="refresh-tagger" type="button">Refresh</button><button class="primary-button" id="install-tagger-deps" type="button">Install Deps</button><button class="primary-button" id="download-tagger-model" type="button">Download Model</button></div>
    </section>
    <section class="panel">
      <div class="section-title"><h2>Model</h2>${statusPill(Boolean(status.modelExists && status.tagsExists), "Ready", "Missing")}</div>
      <div class="form-grid">
        <label>Model dir<input id="tagger-model-dir" value="${escapeHtml(state.tagger.modelDir || status.modelDir || state.settings?.taggerModelDir || "")}"></label>
      </div>
      <div class="action-row">
        <button class="secondary-button" id="choose-tagger-model-dir" type="button">Choose Model Dir</button>
        <button class="primary-button" id="save-tagger-model-dir" type="button">Save Model Dir</button>
      </div>
      <dl class="details">
        <div><dt>Model dir</dt><dd class="mono">${escapeHtml(status.modelDir || state.settings?.taggerModelDir || "")}</dd></div>
        <div><dt>model.onnx</dt><dd>${statusPill(Boolean(status.modelExists), "Found", "Missing")}</dd></div>
        <div><dt>selected_tags.csv</dt><dd>${statusPill(Boolean(status.tagsExists), "Found", "Missing")}</dd></div>
      </dl>
    </section>
    <section class="panel">
      <div class="form-grid">
        <label>Dataset root<input id="tagger-dataset-root" value="${escapeHtml(state.tagger.datasetRoot)}"></label>
        <label>Threshold<input id="tagger-threshold" type="number" step="0.01" min="0" max="1" value="${state.tagger.threshold}"></label>
        <label>Character threshold<input id="tagger-character-threshold" type="number" step="0.01" min="0" max="1" value="${state.tagger.characterThreshold}"></label>
        <label>Mode<select id="tagger-mode"><option value="merge" ${state.tagger.mode === "merge" ? "selected" : ""}>merge</option><option value="overwrite" ${state.tagger.mode === "overwrite" ? "selected" : ""}>overwrite</option></select></label>
      </div>
      <div class="action-row"><button class="secondary-button" id="choose-tagger-root" type="button">Choose Dataset</button><button class="primary-button" id="run-tagger" type="button">Run Tagger</button><button class="secondary-button" data-stop-process="${runId}" type="button">Stop Tagger</button></div>
      ${state.tagger.message ? `<p class="success-text">${escapeHtml(state.tagger.message)}</p>` : ""}${state.tagger.error ? `<p class="error-text">${escapeHtml(state.tagger.error)}</p>` : ""}
    </section>
    <section class="grid two">
      <div class="panel"><h2>Install / Download Logs</h2>${renderLogs(installId)}${renderLogs(downloadId)}</div>
      <div class="panel"><h2>Tagger Logs</h2>${renderLogs(runId)}</div>
    </section>`;
}

function render(): void {
  rootElement.innerHTML = `
    <main>
      <nav class="sidebar" aria-label="Primary">
        <div class="brand"><span class="brand-mark">LT</span><div><strong>LoraTraining</strong><span>Desktop GUI</span></div></div>
        ${(["dashboard", "dataset", "jobs", "tagger"] as ViewId[]).map((view) => `<button class="nav-item ${state.view === view ? "active" : ""}" data-view="${view}" type="button">${view === "dashboard" ? "Dashboard" : view === "dataset" ? "Dataset Studio" : view === "jobs" ? "Jobs" : "WD14 Tagger"}</button>`).join("")}
      </nav>
      <div class="content">${state.view === "dataset" ? renderDatasetStudio() : state.view === "jobs" ? renderJobs() : state.view === "tagger" ? renderTagger() : renderDashboard()}</div>
    </main>`;
  bindEvents();
}

function textValue(id: string): string {
  return document.querySelector<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(`#${id}`)?.value ?? "";
}

function numberValue(id: string, fallback: number): number {
  const parsed = Number(textValue(id));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function syncDatasetControls(): void {
  state.dataset.root = textValue("dataset-root").trim();
  state.dataset.captionExtension = textValue("caption-extension").trim() || ".txt";
  state.dataset.minPixels = Math.max(0, numberValue("min-pixels", 0));
}

function syncBulkControls(): void {
  const operation = textValue("bulk-operation");
  if (["add", "remove", "replace", "move_front", "shuffle"].includes(operation)) state.dataset.operation = operation as any;
  state.dataset.tag = textValue("bulk-tag").trim();
  state.dataset.replacement = textValue("bulk-replacement").trim();
  state.dataset.keepTokens = Math.max(0, numberValue("bulk-keep-tokens", 0));
  state.dataset.includeMissing = Boolean(document.querySelector<HTMLInputElement>("#bulk-include-missing")?.checked);
  state.dataset.caseSensitive = Boolean(document.querySelector<HTMLInputElement>("#bulk-case-sensitive")?.checked);
}

function syncJobDraft(): void {
  if (!state.jobDraft) return;
  document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>("[data-job-field]").forEach((input) => {
    const field = input.dataset.jobField;
    if (!field) return;
    const current = getPath(state.jobDraft, field, "");
    let value: unknown = input instanceof HTMLInputElement && input.type === "checkbox" ? input.checked : input.value;
    if (typeof current === "number") value = Number(input.value);
    setPath(state.jobDraft, field, value);
  });
  setPath(state.jobDraft, "sample.prompts", textValue("job-sample-prompts"));
}

function syncTaggerControls(): void {
  state.tagger.modelDir = textValue("tagger-model-dir").trim();
  state.tagger.datasetRoot = textValue("tagger-dataset-root").trim();
  state.tagger.threshold = numberValue("tagger-threshold", 0.35);
  state.tagger.characterThreshold = numberValue("tagger-character-threshold", 0.35);
  state.tagger.mode = textValue("tagger-mode") === "overwrite" ? "overwrite" : "merge";
}

function bindEvents(): void {
  document.querySelectorAll<HTMLElement>("[data-view]").forEach((element) => element.addEventListener("click", () => {
    const view = element.dataset.view as ViewId;
    state.view = view;
    render();
  }));
  document.querySelector("#refresh-health")?.addEventListener("click", () => void loadInitial());
  document.querySelector("#choose-dataset-root")?.addEventListener("click", () => void chooseDatasetRoot());
  document.querySelector("#scan-dataset")?.addEventListener("click", () => { syncDatasetControls(); void scanDataset(); });
  document.querySelector("#reload-dataset")?.addEventListener("click", () => void scanDataset());
  document.querySelectorAll<HTMLButtonElement>(".dataset-row").forEach((button) => button.addEventListener("click", () => {
    const item = state.dataset.scan?.items.find((entry) => entry.imagePath === button.dataset.imagePath);
    if (!item) return;
    state.dataset.selectedImagePath = item.imagePath;
    state.dataset.draftCaption = item.captionText;
    render();
  }));
  document.querySelector("#caption-editor")?.addEventListener("input", (event) => state.dataset.draftCaption = (event.target as HTMLTextAreaElement).value);
  document.querySelector("#save-caption")?.addEventListener("click", () => void saveSelectedCaption());
  document.querySelector("#apply-bulk-operation")?.addEventListener("click", () => { syncBulkControls(); void applyBulkOperation(); });
  document.querySelector("#create-job")?.addEventListener("click", () => void createJob());
  document.querySelector("#reload-jobs")?.addEventListener("click", () => void loadJobs());
  document.querySelectorAll<HTMLButtonElement>(".job-row").forEach((button) => button.addEventListener("click", () => void selectJob(button.dataset.jobName || "")));
  document.querySelector("#save-job")?.addEventListener("click", () => { syncJobDraft(); void saveJob(); });
  document.querySelector("#build-job-files")?.addEventListener("click", () => { syncJobDraft(); void buildJobFiles(); });
  document.querySelector("#clone-job")?.addEventListener("click", () => { syncJobDraft(); void cloneJob(); });
  document.querySelector("#delete-job")?.addEventListener("click", () => void deleteJob());
  document.querySelector("#plan-train")?.addEventListener("click", () => { syncJobDraft(); void planTrain(); });
  document.querySelector("#start-train")?.addEventListener("click", () => void startPlan("train"));
  document.querySelector("#plan-tensorboard")?.addEventListener("click", () => void planTensorBoard());
  document.querySelector("#start-tensorboard")?.addEventListener("click", () => void startPlan("tensorboard"));
  document.querySelector("#plan-sample")?.addEventListener("click", () => void planSample());
  document.querySelector("#start-sample")?.addEventListener("click", () => void startPlan("sample"));
  document.querySelectorAll<HTMLButtonElement>("[data-stop-process]").forEach((button) => button.addEventListener("click", () => void stopProcess(button.dataset.stopProcess || "")));
  document.querySelector("#refresh-tagger")?.addEventListener("click", () => { syncTaggerControls(); void loadTaggerStatus(); });
  document.querySelector("#install-tagger-deps")?.addEventListener("click", () => void startTaggerPlan("tagger_install_deps_plan"));
  document.querySelector("#download-tagger-model")?.addEventListener("click", () => { syncTaggerControls(); void startTaggerPlan("tagger_download_plan"); });
  document.querySelector("#choose-tagger-model-dir")?.addEventListener("click", () => void chooseTaggerModelDir());
  document.querySelector("#save-tagger-model-dir")?.addEventListener("click", () => { syncTaggerControls(); void saveTaggerModelDir(); });
  document.querySelector("#choose-tagger-root")?.addEventListener("click", () => void chooseTaggerRoot());
  document.querySelector("#run-tagger")?.addEventListener("click", () => { syncTaggerControls(); void runTagger(); });
}

async function loadInitial(): Promise<void> {
  state.healthLoading = true;
  render();
  try {
    const [settings, health] = await Promise.all([bridgeData<Settings>("settings_get"), bridgeData<HealthData>("health_check")]);
    state.settings = settings;
    state.tagger.modelDir = settings.taggerModelDir || state.tagger.modelDir;
    state.health = health;
    state.healthError = undefined;
    await Promise.all([loadJobs(), loadTaggerStatus()]);
  } catch (error) {
    state.healthError = error instanceof Error ? error.message : String(error);
  } finally {
    state.healthLoading = false;
    render();
  }
}

async function chooseDatasetRoot(): Promise<void> {
  const result = await open({ directory: true, multiple: false, title: "Select dataset folder" });
  if (typeof result !== "string") return;
  state.dataset.root = result;
  state.tagger.datasetRoot = result;
  await scanDataset();
}

async function scanDataset(): Promise<void> {
  if (!state.dataset.root) return;
  state.dataset.loading = true;
  state.dataset.error = undefined;
  render();
  try {
    const scan = await bridgeData<DatasetScan>("dataset_scan", { root: state.dataset.root, captionExtension: state.dataset.captionExtension, minPixels: state.dataset.minPixels, maxItems: 5000 });
    state.dataset.scan = scan;
    state.dataset.root = scan.root;
    const selected = scan.items.find((item) => item.imagePath === state.dataset.selectedImagePath) || scan.items[0];
    state.dataset.selectedImagePath = selected?.imagePath;
    state.dataset.draftCaption = selected?.captionText || "";
    state.dataset.message = `Scanned ${scan.imageCount} images.`;
  } catch (error) {
    state.dataset.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.dataset.loading = false;
    render();
  }
}

async function saveSelectedCaption(): Promise<void> {
  const selected = selectedDatasetItem();
  if (!selected) return;
  await bridgeData("dataset_save_caption", { captionPath: selected.captionPath, text: state.dataset.draftCaption });
  await scanDataset();
}

async function applyBulkOperation(): Promise<void> {
  const result = await bridgeData<TagOperationResult>("dataset_apply_tag_operation", {
    root: state.dataset.root,
    captionExtension: state.dataset.captionExtension,
    operation: state.dataset.operation,
    tag: state.dataset.tag,
    replacement: state.dataset.replacement,
    keepTokens: state.dataset.keepTokens,
    includeMissing: state.dataset.includeMissing,
    caseSensitive: state.dataset.caseSensitive,
  });
  state.dataset.message = `Updated ${result.changedCount} captions.`;
  await scanDataset();
}

async function loadJobs(): Promise<void> {
  const result = await bridgeData<{ jobsRoot: string; jobs: JobSummary[] }>("jobs_list");
  state.jobs = result.jobs;
}

async function createJob(): Promise<void> {
  const name = textValue("new-job-name") || "new_lora";
  const result = await bridgeData<{ job: Job }>("job_create", { name });
  state.jobDraft = result.job;
  state.selectedJobName = result.job.name;
  await loadJobs();
  render();
}

async function selectJob(name: string): Promise<void> {
  const result = await bridgeData<{ job: Job }>("job_get", { name });
  state.selectedJobName = name;
  state.jobDraft = result.job;
  state.jobMessage = undefined;
  state.jobError = undefined;
  render();
}

async function saveJob(): Promise<void> {
  if (!state.jobDraft) return;
  const result = await bridgeData<{ job: Job }>("job_save", { job: state.jobDraft });
  state.jobDraft = result.job;
  state.selectedJobName = result.job.name;
  state.jobMessage = "Saved job.";
  await loadJobs();
  render();
}

async function buildJobFiles(): Promise<void> {
  await saveJob();
  if (!state.jobDraft) return;
  await bridgeData("job_build_files", { name: state.jobDraft.name });
  state.jobMessage = "Wrote job.json, dataset.toml, sample_prompts.txt, and _merged_config.toml.";
  render();
}

async function cloneJob(): Promise<void> {
  if (!state.jobDraft) return;
  const target = `${state.jobDraft.name}_copy`;
  const result = await bridgeData<{ job: Job }>("job_clone", { source: state.jobDraft.name, target });
  state.jobDraft = result.job;
  state.selectedJobName = result.job.name;
  await loadJobs();
  render();
}

async function deleteJob(): Promise<void> {
  if (!state.jobDraft) return;
  await bridgeData("job_delete", { name: state.jobDraft.name });
  state.jobDraft = undefined;
  state.selectedJobName = undefined;
  await loadJobs();
  render();
}

async function planTrain(): Promise<void> {
  await saveJob();
  if (!state.jobDraft) return;
  const result = await bridgeData<{ plan: LaunchPlan; errors?: string[] }>("train_launch_plan", { name: state.jobDraft.name, allowInvalid: true });
  state.plans.train = result.plan;
  state.jobError = result.errors?.length ? result.errors.join("\n") : undefined;
  render();
}

async function planTensorBoard(): Promise<void> {
  await saveJob();
  if (!state.jobDraft) return;
  const result = await bridgeData<{ plan: LaunchPlan }>("tensorboard_launch_plan", { name: state.jobDraft.name });
  state.plans.tensorboard = result.plan;
  render();
}

async function planSample(): Promise<void> {
  await saveJob();
  if (!state.jobDraft) return;
  const result = await bridgeData<{ plan: LaunchPlan; errors?: string[] }>("sample_launch_plan", { name: state.jobDraft.name, allowInvalid: true });
  state.plans.sample = result.plan;
  state.jobError = result.errors?.length ? result.errors.join("\n") : undefined;
  render();
}

async function startPlan(kind: "train" | "tensorboard" | "sample"): Promise<void> {
  const plan = state.plans[kind];
  if (!plan) return;
  await startProcess(plan);
}

async function loadTaggerStatus(): Promise<void> {
  state.tagger.modelStatus = await bridgeData<Record<string, unknown>>("tagger_model_status", { modelDir: state.tagger.modelDir || state.settings?.taggerModelDir || "" });
  if (!state.tagger.modelDir) state.tagger.modelDir = String(state.tagger.modelStatus.modelDir || state.settings?.taggerModelDir || "");
  render();
}

async function startTaggerPlan(job: string): Promise<void> {
  const payload = job === "tagger_download_plan" ? { modelDir: state.tagger.modelDir || state.settings?.taggerModelDir || "" } : {};
  const result = await bridgeData<{ plan: LaunchPlan }>(job, payload);
  await startProcess(result.plan);
}

async function chooseTaggerModelDir(): Promise<void> {
  const result = await open({ directory: true, multiple: false, title: "Select WD14 model folder" });
  if (typeof result !== "string") return;
  state.tagger.modelDir = result;
  await saveTaggerModelDir();
}

async function saveTaggerModelDir(): Promise<void> {
  if (!state.settings) return;
  const modelDir = state.tagger.modelDir.trim();
  if (!modelDir) return;
  state.settings = await bridgeData<Settings>("settings_save", { ...state.settings, taggerModelDir: modelDir });
  state.tagger.modelDir = state.settings.taggerModelDir;
  state.tagger.message = "Saved model directory.";
  await loadTaggerStatus();
}

async function chooseTaggerRoot(): Promise<void> {
  const result = await open({ directory: true, multiple: false, title: "Select dataset folder" });
  if (typeof result !== "string") return;
  state.tagger.datasetRoot = result;
  render();
}

async function runTagger(): Promise<void> {
  syncTaggerControls();
  const result = await bridgeData<{ plan: LaunchPlan }>("tagger_launch_plan", {
    modelDir: state.tagger.modelDir || state.settings?.taggerModelDir || "",
    datasetRoot: state.tagger.datasetRoot,
    threshold: state.tagger.threshold,
    characterThreshold: state.tagger.characterThreshold,
    mode: state.tagger.mode,
  });
  await startProcess(result.plan);
}

setInterval(() => {
  const ids = Object.keys(state.processes);
  if (!ids.length) return;
  Promise.all(ids.map(refreshProcess)).then(render).catch(() => undefined);
}, 2000);

render();
void loadInitial();
