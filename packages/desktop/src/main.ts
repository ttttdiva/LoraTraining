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
type AgentProvider = {
  id: string;
  label: string;
  kind: "api" | "cli";
  available: boolean;
  apiKeyEnv?: string;
  apiKeyConfigured?: boolean;
  model: string;
  baseUrl?: string;
  bin?: string;
  path?: string;
  error?: string;
  defaultModel?: string;
  defaultBaseUrl?: string;
};
type HealthData = {
  app: { projectRoot: string; bridgeRoot: string };
  system: { platform: string; pythonVersion: string; executable: string; cwd: string };
  tools: ToolStatus[];
  gpu: { available: boolean; gpus: GpuInfo[]; error?: string };
  engines: EngineCandidate[];
  agentProviders?: AgentProvider[];
};

type Settings = {
  jobsRoot: string;
  datasetsRoot: string;
  defaultEngineId: string;
  taggerModelDir: string;
  datasets?: DatasetProfile[];
  engines: Array<{ id: string; name: string; type: string; root: string; venv: string }>;
  defaults: Record<string, unknown>;
  agent?: {
    provider?: string;
    model?: string;
    baseUrl?: string;
    apiKeyEnv?: string;
    temperature?: number;
    imageMaxSide?: number;
    outputFormat?: string;
    captionMode?: string;
    taggerThreshold?: number;
    characterThreshold?: number;
  };
  ui?: {
    view?: string;
    datasetRoot?: string;
    activeDatasetId?: string;
    datasetCaptionExtension?: string;
    datasetMinPixels?: number;
    datasetSelectedImagePath?: string;
    taggerDatasetRoot?: string;
    taggerThreshold?: number;
    taggerCharacterThreshold?: number;
    taggerMode?: string;
    agentSourceRoot?: string;
    agentJobName?: string;
    agentGoal?: string;
    agentIntent?: string;
    agentTriggerTag?: string;
    agentArchitecture?: string;
    agentEngineId?: string;
    agentGpuIds?: string;
    agentMultiGpuMode?: string;
    selectedJobName?: string;
  };
};

type DatasetProfile = {
  id: string;
  name: string;
  root: string;
  captionExtension: string;
  minPixels: number;
  lastSelectedImagePath?: string;
  updatedAt?: string;
};

type DatasetItem = {
  imagePath: string;
  captionPath: string;
  relativePath: string;
  captionExists: boolean;
  captionText: string;
  tags: string[];
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
  runRecordPath?: string;
};
type ProcessStatus = {
  kind?: string;
  command?: string;
  running: boolean;
  exitCode?: number | null;
  logs: string[];
};
type ProcessEnvelope = { id: string; status: ProcessStatus };
type JobContextMenu = { jobName: string; x: number; y: number };
type TrainingMetric = { index: number; step?: number; epoch?: number; loss: number; lr?: number; raw: string };

type ViewId = "dashboard" | "agent" | "dataset" | "jobs" | "tagger";

type DatasetState = {
  profiles: DatasetProfile[];
  activeProfileId?: string;
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

type AgentAutopilotResult = {
  job: Job;
  jobPath: string;
  sourceSummary: Record<string, any>;
  recommendation: Record<string, any>;
  external: { called: boolean; ok: boolean; text?: string; error?: string; parsedJson?: Record<string, any>; provider?: AgentProvider };
  files: Record<string, string>;
  preprocessPlan?: LaunchPlan;
  taggerPlan?: LaunchPlan;
  trainPlan?: LaunchPlan;
  trainErrors?: string[];
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
  jobContextMenu?: JobContextMenu;
  jobMessage?: string;
  jobError?: string;
  plans: Record<string, LaunchPlan | undefined>;
  processes: Record<string, ProcessStatus | undefined>;
  agent: {
    sourceRoot: string;
    jobName: string;
    goal: string;
    intent: "character" | "style" | "concept";
    triggerTag: string;
    provider: string;
    model: string;
    baseUrl: string;
    apiKeyEnv: string;
    architecture: string;
    engineId: string;
    gpuIds: string;
    multiGpuMode: string;
    imageMaxSide: number;
    outputFormat: "keep" | "png" | "jpg";
    callModel: boolean;
    loading: boolean;
    message?: string;
    error?: string;
    result?: AgentAutopilotResult;
  };
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
    profiles: [],
    activeProfileId: undefined,
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
  agent: {
    sourceRoot: "",
    jobName: "",
    goal: "Build a high quality LoRA from this unorganized image folder.",
    intent: "character",
    triggerTag: "",
    provider: "codex-cli",
    model: "gpt-5-codex",
    baseUrl: "",
    apiKeyEnv: "",
    architecture: "anima",
    engineId: "",
    gpuIds: "0",
    multiGpuMode: "single",
    imageMaxSide: 1536,
    outputFormat: "png",
    callModel: true,
    loading: false,
  },
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
let persistUiTimer: number | undefined;

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

function agentProviders(): AgentProvider[] {
  return state.health?.agentProviders || [];
}

function selectedAgentProvider(): AgentProvider | undefined {
  return agentProviders().find((provider) => provider.id === state.agent.provider);
}

function providerStatus(provider: AgentProvider): string {
  if (provider.kind === "cli") return provider.path || provider.error || provider.bin || "";
  if (provider.id === "ollama" || provider.id === "openai_compatible_local") return provider.error || provider.baseUrl || "";
  return provider.apiKeyConfigured ? provider.apiKeyEnv || "" : provider.error || provider.apiKeyEnv || "";
}

function planCommand(plan?: LaunchPlan): string {
  if (!plan) return "";
  const record = plan.runRecordPath ? `\n\nRun record: ${plan.runRecordPath}` : "";
  return `${plan.displayCommand}${record}`;
}

function imageSrc(path: string): string {
  return convertFileSrc(path, "asset");
}

function splitCaptionTags(text: string): string[] {
  const cleaned = text.replaceAll("\r\n", "\n").replaceAll("\r", "\n").trim();
  if (!cleaned) return [];
  const parts = cleaned.includes(",") ? cleaned.split(",") : cleaned.split("\n");
  return parts.map((tag) => tag.trim()).filter(Boolean);
}

function joinCaptionTags(tags: string[]): string {
  return tags.map((tag) => tag.trim()).filter(Boolean).join(", ");
}

function classifyTag(tag: string): string {
  const value = tag.trim().toLowerCase();
  if (value.startsWith("@") || ["1girl", "1boy", "girl", "boy", "solo"].includes(value)) return "tag-char";
  if (["hair", "eyes", "skin", "body", "face"].some((part) => value.includes(part))) return "tag-char";
  if (["shirt", "skirt", "pants", "dress", "uniform", "clothes", "wearing", "jacket"].some((part) => value.includes(part))) return "tag-clothes";
  if (["background", "outdoor", "indoor", "room", "sky", "tree", "nature"].some((part) => value.includes(part))) return "tag-bg";
  if (["masterpiece", "best quality", "highres", "year", "score", "rating"].some((part) => value.includes(part))) return "tag-meta";
  return "tag-general";
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

function validView(value: unknown): value is ViewId {
  return ["dashboard", "agent", "dataset", "jobs", "tagger"].includes(String(value));
}

function validAgentIntent(value: unknown): value is "character" | "style" | "concept" {
  return ["character", "style", "concept"].includes(String(value));
}

function validTaggerMode(value: unknown): value is "merge" | "overwrite" {
  return ["merge", "overwrite"].includes(String(value));
}

function slug(value: string): string {
  return value.trim().replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[._]+|[._]+$/g, "") || `dataset_${Date.now()}`;
}

function folderName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() || "dataset";
}

function normalizeDatasetProfile(raw: Partial<DatasetProfile>, fallbackRoot = ""): DatasetProfile | undefined {
  const root = String(raw.root || fallbackRoot || "").trim();
  if (!root) return undefined;
  const name = String(raw.name || folderName(root)).trim() || folderName(root);
  return {
    id: String(raw.id || slug(name || root)),
    name,
    root,
    captionExtension: String(raw.captionExtension || ".txt"),
    minPixels: Math.max(0, Number(raw.minPixels || 0)),
    lastSelectedImagePath: String(raw.lastSelectedImagePath || ""),
    updatedAt: String(raw.updatedAt || new Date().toISOString()),
  };
}

function loadDatasetProfiles(settings: Settings): DatasetProfile[] {
  const profiles = Array.isArray(settings.datasets)
    ? settings.datasets.map((item) => normalizeDatasetProfile(item)).filter((item): item is DatasetProfile => Boolean(item))
    : [];
  const legacy = normalizeDatasetProfile({
    name: settings.ui?.datasetRoot ? folderName(settings.ui.datasetRoot) : "",
    root: settings.ui?.datasetRoot || "",
    captionExtension: settings.ui?.datasetCaptionExtension || ".txt",
    minPixels: settings.ui?.datasetMinPixels || 0,
    lastSelectedImagePath: settings.ui?.datasetSelectedImagePath || "",
  });
  if (legacy && !profiles.some((profile) => profile.root.toLowerCase() === legacy.root.toLowerCase())) {
    profiles.unshift(legacy);
  }
  return profiles;
}

function activeDatasetProfile(): DatasetProfile | undefined {
  return state.dataset.profiles.find((profile) => profile.id === state.dataset.activeProfileId);
}

function setDatasetFromProfile(profile: DatasetProfile): void {
  state.dataset.activeProfileId = profile.id;
  state.dataset.root = profile.root;
  state.dataset.captionExtension = profile.captionExtension || ".txt";
  state.dataset.minPixels = Math.max(0, Number(profile.minPixels || 0));
  state.dataset.selectedImagePath = profile.lastSelectedImagePath || undefined;
  state.dataset.draftCaption = "";
  state.dataset.scan = undefined;
  state.tagger.datasetRoot = profile.root;
}

function upsertDatasetProfile(profile: DatasetProfile): DatasetProfile {
  const normalized = normalizeDatasetProfile({ ...profile, updatedAt: new Date().toISOString() });
  if (!normalized) throw new Error("dataset root is required");
  const index = state.dataset.profiles.findIndex((item) => item.id === normalized.id || item.root.toLowerCase() === normalized.root.toLowerCase());
  if (index >= 0) {
    state.dataset.profiles[index] = { ...state.dataset.profiles[index], ...normalized };
  } else {
    state.dataset.profiles.unshift(normalized);
  }
  return normalized;
}

function updateActiveDatasetProfile(): void {
  if (!state.dataset.root) return;
  const existing = activeDatasetProfile();
  const profile = upsertDatasetProfile({
    id: existing?.id || slug(folderName(state.dataset.root)),
    name: existing?.name || folderName(state.dataset.root),
    root: state.dataset.root,
    captionExtension: state.dataset.captionExtension,
    minPixels: state.dataset.minPixels,
    lastSelectedImagePath: state.dataset.selectedImagePath || "",
  });
  state.dataset.activeProfileId = profile.id;
}

function applyPersistedUi(settings: Settings): void {
  const ui = settings.ui || {};
  state.dataset.profiles = loadDatasetProfiles(settings);
  if (validView(ui.view)) state.view = ui.view;
  if (typeof ui.activeDatasetId === "string") state.dataset.activeProfileId = ui.activeDatasetId;
  const activeProfile = activeDatasetProfile() || state.dataset.profiles[0];
  if (activeProfile) setDatasetFromProfile(activeProfile);
  else if (typeof ui.datasetRoot === "string") state.dataset.root = ui.datasetRoot;
  if (typeof ui.datasetCaptionExtension === "string" && ui.datasetCaptionExtension.trim()) state.dataset.captionExtension = ui.datasetCaptionExtension;
  if (Number.isFinite(Number(ui.datasetMinPixels))) state.dataset.minPixels = Math.max(0, Number(ui.datasetMinPixels));
  if (typeof ui.datasetSelectedImagePath === "string") state.dataset.selectedImagePath = ui.datasetSelectedImagePath;
  if (typeof ui.taggerDatasetRoot === "string") state.tagger.datasetRoot = ui.taggerDatasetRoot;
  if (Number.isFinite(Number(ui.taggerThreshold))) state.tagger.threshold = Number(ui.taggerThreshold);
  if (Number.isFinite(Number(ui.taggerCharacterThreshold))) state.tagger.characterThreshold = Number(ui.taggerCharacterThreshold);
  if (validTaggerMode(ui.taggerMode)) state.tagger.mode = ui.taggerMode;
  if (typeof ui.agentSourceRoot === "string") state.agent.sourceRoot = ui.agentSourceRoot;
  if (typeof ui.agentJobName === "string") state.agent.jobName = ui.agentJobName;
  if (typeof ui.agentGoal === "string" && ui.agentGoal.trim()) state.agent.goal = ui.agentGoal;
  if (validAgentIntent(ui.agentIntent)) state.agent.intent = ui.agentIntent;
  if (typeof ui.agentTriggerTag === "string") state.agent.triggerTag = ui.agentTriggerTag;
  if (typeof ui.agentArchitecture === "string" && ui.agentArchitecture) state.agent.architecture = ui.agentArchitecture;
  if (typeof ui.agentEngineId === "string") state.agent.engineId = ui.agentEngineId;
  if (typeof ui.agentGpuIds === "string" && ui.agentGpuIds.trim()) state.agent.gpuIds = ui.agentGpuIds;
  if (typeof ui.agentMultiGpuMode === "string" && ui.agentMultiGpuMode) state.agent.multiGpuMode = ui.agentMultiGpuMode;
  if (typeof ui.selectedJobName === "string") state.selectedJobName = ui.selectedJobName;
}

function currentUiSettings(): NonNullable<Settings["ui"]> {
  return {
    view: state.view,
    datasetRoot: state.dataset.root,
    activeDatasetId: state.dataset.activeProfileId || "",
    datasetCaptionExtension: state.dataset.captionExtension,
    datasetMinPixels: state.dataset.minPixels,
    datasetSelectedImagePath: state.dataset.selectedImagePath || "",
    taggerDatasetRoot: state.tagger.datasetRoot,
    taggerThreshold: state.tagger.threshold,
    taggerCharacterThreshold: state.tagger.characterThreshold,
    taggerMode: state.tagger.mode,
    agentSourceRoot: state.agent.sourceRoot,
    agentJobName: state.agent.jobName,
    agentGoal: state.agent.goal,
    agentIntent: state.agent.intent,
    agentTriggerTag: state.agent.triggerTag,
    agentArchitecture: state.agent.architecture,
    agentEngineId: state.agent.engineId,
    agentGpuIds: state.agent.gpuIds,
    agentMultiGpuMode: state.agent.multiGpuMode,
    selectedJobName: state.selectedJobName || "",
  };
}

function queuePersistUiState(): void {
  if (!state.settings) return;
  if (persistUiTimer !== undefined) window.clearTimeout(persistUiTimer);
  persistUiTimer = window.setTimeout(() => void persistUiState(), 300);
}

async function persistUiState(): Promise<void> {
  if (!state.settings) return;
  persistUiTimer = undefined;
  const next: Settings = {
    ...state.settings,
    datasets: state.dataset.profiles,
    agent: {
      ...state.settings.agent,
      provider: state.agent.provider,
      model: state.agent.model,
      baseUrl: state.agent.baseUrl,
      apiKeyEnv: state.agent.apiKeyEnv,
      imageMaxSide: state.agent.imageMaxSide,
      outputFormat: state.agent.outputFormat,
      taggerThreshold: state.tagger.threshold,
      characterThreshold: state.tagger.characterThreshold,
    },
    ui: {
      ...state.settings.ui,
      ...currentUiSettings(),
    },
  };
  try {
    state.settings = await bridgeData<Settings>("settings_save", next as unknown as Record<string, unknown>);
  } catch (error) {
    console.error("failed to persist UI state", error);
  }
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
          <div><dt>Datasets</dt><dd class="mono">${escapeHtml(state.settings?.datasetsRoot || "-")}</dd></div>
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

function datasetTagSuggestions(scan?: DatasetScan): string {
  const tags = new Set<string>();
  scan?.items.forEach((item) => {
    const itemTags = item.tags?.length ? item.tags : splitCaptionTags(item.captionText);
    itemTags.forEach((tag) => tags.add(tag));
  });
  return Array.from(tags).sort((left, right) => left.localeCompare(right)).map((tag) => `<option value="${escapeHtml(tag)}"></option>`).join("");
}

function renderTagChip(tag: string, index: number): string {
  return `<button class="caption-tag-chip ${classifyTag(tag)}" type="button" data-remove-tag-index="${index}" title="Remove tag"><span>${escapeHtml(tag)}</span><span class="tag-chip-remove">x</span></button>`;
}

function renderDatasetStatusMessages(): string {
  const d = state.dataset;
  return `${d.loading ? `<p class="muted">Working...</p>` : ""}${d.message ? `<p class="success-text">${escapeHtml(d.message)}</p>` : ""}${d.error ? `<p class="error-text">${escapeHtml(d.error)}</p>` : ""}`;
}

function renderSelectedTagsEditor(selected: DatasetItem, scan?: DatasetScan): string {
  const tags = splitCaptionTags(state.dataset.draftCaption);
  const chips = tags.length ? tags.map(renderTagChip).join("") : `<span class="tag-empty">No tags</span>`;
  return `
    <div class="visual-caption-editor">
      <div class="tag-chip-grid" id="tag-chip-grid">
        ${chips}
        <input id="tag-add-input" class="tag-add-input" list="dataset-tag-suggestions" placeholder="+ add tag" autocomplete="off">
      </div>
      <datalist id="dataset-tag-suggestions">${datasetTagSuggestions(scan)}</datalist>
    </div>
    <details class="raw-caption-details">
      <summary>Raw caption text</summary>
      <textarea id="caption-editor" class="raw-caption-editor" spellcheck="false">${escapeHtml(state.dataset.draftCaption)}</textarea>
    </details>`;
}

function renderDatasetIssueList(item: DatasetItem): string {
  return item.issues.length
    ? item.issues.map((issue) => `<span class="issue-pill">${escapeHtml(issue)}</span>`).join("")
    : `<span class="issue-pill issue-ok">ok</span>`;
}

function renderDatasetRowTags(item: DatasetItem): string {
  const rowTags = (item.tags?.length ? item.tags : splitCaptionTags(item.captionText)).slice(0, 5);
  return rowTags.length
    ? `<span class="dataset-row-tags">${rowTags.map((tag) => `<span class="mini-tag ${classifyTag(tag)}">${escapeHtml(tag)}</span>`).join("")}</span>`
    : `<span class="dataset-row-tags"><span class="mini-tag empty">empty</span></span>`;
}

function renderDatasetRowContent(item: DatasetItem): string {
  return `
    <img src="${escapeHtml(imageSrc(item.imagePath))}" alt="" loading="lazy">
    <span class="dataset-row-main"><strong>${escapeHtml(item.relativePath)}</strong><span>${escapeHtml(imageSizeLabel(item))} / ${item.tagCount} tags</span>${renderDatasetRowTags(item)}<span class="issue-list">${renderDatasetIssueList(item)}</span></span>`;
}

function renderDatasetRow(item: DatasetItem, selectedImagePath?: string): string {
  return `<button class="dataset-row ${item.imagePath === selectedImagePath ? "selected" : ""}" type="button" data-image-path="${escapeHtml(item.imagePath)}">
    ${renderDatasetRowContent(item)}
  </button>`;
}

function renderDatasetEditorPanel(selected: DatasetItem | undefined, scan?: DatasetScan): string {
  if (!selected) return `<h2>Caption Editor</h2><p class="muted">Select an image.</p>`;
  return `
    <div class="section-title"><div><h2>Visual Tag Editor</h2><p class="muted mono">${escapeHtml(selected.captionPath)}</p></div>${statusPill(selected.captionExists, "Exists", "Will create")}</div>
    <div class="preview-block"><img src="${escapeHtml(imageSrc(selected.imagePath))}" alt=""><dl class="details"><div><dt>Image</dt><dd>${escapeHtml(selected.relativePath)}</dd></div><div><dt>Size</dt><dd>${escapeHtml(imageSizeLabel(selected))}</dd></div><div><dt>Tags</dt><dd>${selected.tagCount}</dd></div></dl></div>
    ${renderSelectedTagsEditor(selected, scan)}
    <div class="action-row"><button class="primary-button" id="save-caption" type="button">Save Caption</button><button class="secondary-button" id="reload-dataset" type="button">Rescan</button><button class="secondary-button" id="run-tagger-from-dataset" type="button">Run WD14 Tagger</button></div>`;
}

function renderDatasetStudio(): string {
  const d = state.dataset;
  const scan = d.scan;
  const selected = selectedDatasetItem();
  const profile = activeDatasetProfile();
  const profileOptions = d.profiles.length
    ? d.profiles.map((item) => `<option value="${escapeHtml(item.id)}" ${item.id === d.activeProfileId ? "selected" : ""}>${escapeHtml(item.name)}</option>`).join("")
    : `<option value="">No registered datasets</option>`;
  return `
    <section class="hero-band">
      <div><p class="eyebrow">Dataset Studio</p><h1>Caption and Tag Editor</h1><p class="lead">Scan folders, edit captions, and apply batch tag operations.</p></div>
      <div class="action-row"><button class="secondary-button" id="register-dataset" type="button">Add Dataset</button><button class="primary-button" id="scan-dataset" type="button" ${d.root ? "" : "disabled"}>Scan</button></div>
    </section>
    <section class="panel">
      <div class="section-title"><div><h2>Dataset Library</h2><p class="muted mono">${escapeHtml(state.settings?.datasetsRoot || "")}</p></div><span class="pill">${d.profiles.length}</span></div>
      <div class="dataset-library-controls">
        <label>Dataset<select id="dataset-profile">${profileOptions}</select></label>
        <button class="secondary-button" id="rename-dataset-profile" type="button" ${profile ? "" : "disabled"}>Rename</button>
        <button class="secondary-button" id="save-dataset-profile" type="button" ${d.root ? "" : "disabled"}>Save Current</button>
        <button class="danger-button" id="forget-dataset-profile" type="button" ${profile ? "" : "disabled"}>Forget</button>
      </div>
      <div class="form-grid dataset-controls">
        <label>Dataset root<input id="dataset-root" value="${escapeHtml(d.root)}" placeholder="data/datasets/my_lora"></label>
        <label>Caption extension<input id="caption-extension" value="${escapeHtml(d.captionExtension)}"></label>
        <label>Low-res threshold pixels<input id="min-pixels" type="number" min="0" step="1" value="${d.minPixels}"></label>
      </div>
      <div id="dataset-status-messages">${renderDatasetStatusMessages()}</div>
    </section>
    ${scan ? `
      <section class="grid cards summary-grid">
        <article class="card compact"><p class="eyebrow">Images</p><h3 id="dataset-summary-images">${scan.imageCount}</h3></article>
        <article class="card compact"><p class="eyebrow">Captions</p><h3 id="dataset-summary-captions">${scan.captionCount}</h3></article>
        <article class="card compact"><p class="eyebrow">Missing</p><h3 id="dataset-summary-missing">${scan.missingCaptionCount}</h3></article>
        <article class="card compact"><p class="eyebrow">Orphans</p><h3 id="dataset-summary-orphans">${scan.orphanCaptionCount}</h3></article>
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
            ${scan.items.map((item) => renderDatasetRow(item, d.selectedImagePath)).join("")}
          </div>
        </div>
        <section class="panel editor-panel" id="dataset-editor-panel">
          ${renderDatasetEditorPanel(selected, scan)}
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

function parseNumberToken(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function extractTrainingMetrics(logs: string[] = []): TrainingMetric[] {
  const metrics: TrainingMetric[] = [];
  const lossPattern = /(?:^|[\s,{[(])(?:train[_/-]?loss|avg[_/-]?loss|loss)\s*[=:]\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)/i;
  const lrPattern = /(?:^|[\s,{[(])(?:learning[_-]?rate|lr)\s*[=:]\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)/i;
  const stepPattern = /(?:global[_-]?step|step)\s*[=:]\s*(\d+)/i;
  const progressPattern = /(?:^|\s)(\d+)\s*\/\s*(\d+)(?:\s|$)/;
  const epochPattern = /epoch\s*[=:]?\s*([-+]?\d*\.?\d+)/i;

  logs.forEach((raw, index) => {
    const line = raw.replace(/^ERR:\s*/, "").trim();
    const loss = parseNumberToken(line.match(lossPattern)?.[1]);
    if (loss === undefined) return;
    const lr = parseNumberToken(line.match(lrPattern)?.[1]);
    const step = parseNumberToken(line.match(stepPattern)?.[1]) ?? parseNumberToken(line.match(progressPattern)?.[1]);
    const epoch = parseNumberToken(line.match(epochPattern)?.[1]);
    metrics.push({ index, step, epoch, loss, lr, raw });
  });

  return metrics.slice(-240);
}

function metricPoints(values: number[], width: number, height: number, padding: number): string {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.max(Math.abs(max), 1);
  return values.map((value, index) => {
    const x = padding + (values.length === 1 ? 0 : (index / (values.length - 1)) * (width - padding * 2));
    const y = height - padding - ((value - min) / span) * (height - padding * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function renderTrainingMetrics(processId: string): string {
  const process = state.processes[processId];
  const metrics = extractTrainingMetrics(process?.logs || []);
  if (!metrics.length) {
    return `<div class="metrics-panel empty"><p class="muted">No loss values parsed yet. Training logs and TensorBoard are still available below.</p></div>`;
  }

  const latest = metrics[metrics.length - 1];
  const values = metrics.map((item) => item.loss);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const chartWidth = 520;
  const chartHeight = 150;
  const points = metricPoints(values, chartWidth, chartHeight, 16);
  const stepLabel = latest.step !== undefined ? `step ${latest.step}` : `log ${latest.index + 1}`;
  const lrLabel = latest.lr !== undefined ? latest.lr.toExponential(2) : "-";

  return `
    <div class="metrics-panel">
      <dl class="metric-strip">
        <div><dt>Loss</dt><dd>${latest.loss.toFixed(5)}</dd></div>
        <div><dt>Min</dt><dd>${min.toFixed(5)}</dd></div>
        <div><dt>Max</dt><dd>${max.toFixed(5)}</dd></div>
        <div><dt>LR</dt><dd>${escapeHtml(lrLabel)}</dd></div>
        <div><dt>Point</dt><dd>${escapeHtml(stepLabel)}</dd></div>
      </dl>
      <svg class="metric-chart" viewBox="0 0 ${chartWidth} ${chartHeight}" role="img" aria-label="Training loss chart" preserveAspectRatio="none">
        <line x1="16" y1="${chartHeight - 16}" x2="${chartWidth - 16}" y2="${chartHeight - 16}"></line>
        <polyline points="${points}"></polyline>
      </svg>
    </div>`;
}

function renderLogs(processId: string): string {
  const process = state.processes[processId];
  if (!process) return `<pre class="log-box">No process started.</pre>`;
  return `<pre class="log-box">${escapeHtml((process.logs || []).slice(-300).join("\n"))}</pre>`;
}

function renderAgentAutopilot(): string {
  const a = state.agent;
  const provider = selectedAgentProvider();
  const result = a.result;
  const preprocessId = result?.preprocessPlan?.id || `agent-preprocess:${result?.job?.name || a.jobName || "autopilot_job"}`;
  const taggerId = result?.taggerPlan?.id || `tagger:${(result?.job?.dataset?.imageDir || "dataset").split(/[\\/]/).pop() || "dataset"}`;
  const trainId = result?.trainPlan?.id || `train:${result?.job?.name || a.jobName || "autopilot_job"}`;
  const providerRows = agentProviders().map((item) => `
    <tr>
      <td>${escapeHtml(item.label)}</td>
      <td>${statusPill(item.available, item.kind === "cli" ? "Found" : "Ready", "Check")}</td>
      <td class="mono">${escapeHtml(item.model || item.defaultModel || "")}</td>
      <td class="mono">${escapeHtml(providerStatus(item))}</td>
    </tr>`).join("");
  const summary = result?.sourceSummary;
  const recommendation = result?.recommendation;
  return `
    <section class="hero-band">
      <div><p class="eyebrow">Agent Autopilot</p><h1>Dataset Agent</h1><p class="lead">Point it at an unorganized image folder, then generate preprocessing, tagging, and training settings.</p></div>
      <div class="action-row"><button class="secondary-button" id="choose-agent-source" type="button">Choose Folder</button><button class="primary-button" id="build-agent-plan" type="button" ${a.sourceRoot || a.loading ? "" : "disabled"}>${a.loading ? "Working..." : "Build Plan"}</button></div>
    </section>
    <section class="grid two">
      <article class="panel">
        <div class="section-title"><h2>Input</h2>${provider ? statusPill(provider.available, "Provider OK", "Provider Check") : ""}</div>
        <div class="form-grid">
          <label>Source images<input id="agent-source-root" value="${escapeHtml(a.sourceRoot)}" placeholder="data/datasets/raw_images"></label>
          <label>Job name<input id="agent-job-name" value="${escapeHtml(a.jobName)}" placeholder="my_lora_agent"></label>
          <label>Intent<select id="agent-intent">
            ${["character", "style", "concept"].map((item) => `<option value="${item}" ${a.intent === item ? "selected" : ""}>${item}</option>`).join("")}
          </select></label>
          <label>Trigger tag<input id="agent-trigger-tag" value="${escapeHtml(a.triggerTag)}" placeholder="character_token"></label>
          <label>Architecture<select id="agent-architecture">
            ${["anima", "sd15", "sd2", "sdxl"].map((item) => `<option value="${item}" ${a.architecture === item ? "selected" : ""}>${item}</option>`).join("")}
          </select></label>
          <label>Engine<select id="agent-engine-id">
            ${(state.settings?.engines || []).map((engine) => `<option value="${escapeHtml(engine.id)}" ${a.engineId === engine.id ? "selected" : ""}>${escapeHtml(engine.id)}</option>`).join("")}
          </select></label>
          <label>GPU IDs<input id="agent-gpu-ids" value="${escapeHtml(a.gpuIds)}" placeholder="0,1"></label>
          <label>Multi GPU<select id="agent-multi-gpu-mode">
            ${["single", "ddp", "fsdp", "fsdp2", "deepspeed"].map((item) => `<option value="${item}" ${a.multiGpuMode === item ? "selected" : ""}>${item}</option>`).join("")}
          </select></label>
          <label>Image max side<input id="agent-image-max-side" type="number" min="512" step="64" value="${a.imageMaxSide}"></label>
          <label>Output format<select id="agent-output-format">
            ${["png", "jpg", "keep"].map((item) => `<option value="${item}" ${a.outputFormat === item ? "selected" : ""}>${item}</option>`).join("")}
          </select></label>
        </div>
        <label class="textarea-label">Goal<textarea id="agent-goal">${escapeHtml(a.goal)}</textarea></label>
      </article>
      <article class="panel">
        <div class="section-title"><h2>Agent Provider</h2><span class="pill">${escapeHtml(provider?.kind || "-")}</span></div>
        <div class="form-grid">
          <label>Provider<select id="agent-provider">
            ${agentProviders().map((item) => `<option value="${escapeHtml(item.id)}" ${a.provider === item.id ? "selected" : ""}>${escapeHtml(item.label)}</option>`).join("")}
          </select></label>
          <label>Model<input id="agent-model" value="${escapeHtml(a.model)}"></label>
          <label>Base URL<input id="agent-base-url" value="${escapeHtml(a.baseUrl)}" placeholder="${escapeHtml(provider?.defaultBaseUrl || "")}"></label>
          <label>API key env<input id="agent-api-key-env" value="${escapeHtml(a.apiKeyEnv)}" placeholder="${escapeHtml(provider?.apiKeyEnv || "")}"></label>
          <label class="check-row"><input id="agent-call-model" type="checkbox" ${a.callModel ? "checked" : ""}>Ask selected agent</label>
        </div>
        <table class="provider-table"><thead><tr><th>Provider</th><th>Status</th><th>Model</th><th>Detail</th></tr></thead><tbody>${providerRows}</tbody></table>
      </article>
    </section>
    ${a.message ? `<p class="success-text">${escapeHtml(a.message)}</p>` : ""}${a.error ? `<p class="error-text">${escapeHtml(a.error)}</p>` : ""}
    ${result ? `
      <section class="grid cards summary-grid">
        <article class="card compact"><p class="eyebrow">Images</p><h3>${escapeHtml(summary?.imageCount ?? 0)}</h3></article>
        <article class="card compact"><p class="eyebrow">Missing captions</p><h3>${escapeHtml(summary?.missingCaptionCount ?? 0)}</h3></article>
        <article class="card compact"><p class="eyebrow">Median long edge</p><h3>${escapeHtml(summary?.medianLongEdge ?? 0)}</h3></article>
        <article class="card compact"><p class="eyebrow">Duplicates</p><h3>${escapeHtml(summary?.duplicateHashCount ?? 0)}</h3></article>
      </section>
      <section class="panel">
        <div class="section-title"><div><h2>Generated Job</h2><p class="muted mono">${escapeHtml(result.jobPath)}</p></div><span class="pill">${escapeHtml(result.job.name)}</span></div>
        <div class="action-row">
          <button class="primary-button" id="open-agent-job" type="button">Open Job Editor</button>
          <button class="secondary-button" id="start-agent-preprocess" type="button" ${result.preprocessPlan ? "" : "disabled"}>Run Preprocess</button>
          <button class="secondary-button" id="start-agent-tagger" type="button" ${result.taggerPlan ? "" : "disabled"}>Run WD14 Tagger</button>
          <button class="secondary-button" id="start-agent-train" type="button" ${result.trainPlan ? "" : "disabled"}>Start Training</button>
          <button class="secondary-button" data-stop-process="${preprocessId}" type="button">Stop Preprocess</button>
          <button class="secondary-button" data-stop-process="${taggerId}" type="button">Stop Tagger</button>
          <button class="secondary-button" data-stop-process="${trainId}" type="button">Stop Training</button>
        </div>
        ${result.trainErrors?.length ? `<p class="error-text">${escapeHtml(result.trainErrors.join("\n"))}</p>` : ""}
      </section>
      <section class="grid two">
        <article class="panel"><h2>Recommendation</h2><pre class="json-box">${escapeHtml(JSON.stringify(recommendation, null, 2))}</pre></article>
        <article class="panel"><h2>Agent Response</h2><pre class="json-box">${escapeHtml(result.external?.text || result.external?.error || "Heuristic only.")}</pre></article>
      </section>
      <section class="grid two">
        <article class="panel"><h2>Preprocess / Tagger Plans</h2>${result.preprocessPlan ? `<pre>${escapeHtml(planCommand(result.preprocessPlan))}</pre>` : ""}${result.taggerPlan ? `<pre>${escapeHtml(planCommand(result.taggerPlan))}</pre>` : ""}${renderLogs(preprocessId)}${renderLogs(taggerId)}</article>
        <article class="panel"><h2>Training Plan</h2>${result.trainPlan ? `<pre>${escapeHtml(planCommand(result.trainPlan))}</pre>` : ""}${renderTrainingMetrics(trainId)}${renderLogs(trainId)}</article>
      </section>
    ` : `<section class="panel"><p class="muted">No Autopilot plan yet.</p></section>`}`;
}

function renderJobEditor(): string {
  if (!state.jobDraft) return `<section class="panel"><p class="muted">Select or create a job.</p></section>`;
  const job = state.jobDraft;
  const trainId = `train:${job.name}`;
  const setupId = `engine_setup:${String(job.engineId || "engine").replaceAll(" ", "_")}`;
  const convertId = `convert:${job.name}`;
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
        ${checkbox("postActions.convertToComfy", "Convert to ComfyUI after training")}
        ${checkbox("postActions.keepUnet", "Keep source UNET LoRA after conversion")}
        ${checkbox("postActions.shutdown", "Shutdown Windows after training")}
        ${input("postActions.shutdownDelaySeconds", "Shutdown delay seconds", "number")}
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
        <button class="secondary-button" id="plan-engine-setup" type="button">Plan Engine Setup</button>
        <button class="secondary-button" id="start-engine-setup" type="button" ${state.plans.engineSetup ? "" : "disabled"}>Run Setup</button>
        <button class="primary-button" id="plan-train" type="button">Plan Training</button>
        <button class="primary-button" id="start-train" type="button" ${state.plans.train ? "" : "disabled"}>Start Training</button>
        <button class="secondary-button" data-stop-process="${trainId}" type="button">Stop Training</button>
        <button class="secondary-button" id="plan-convert" type="button">Plan ComfyUI Convert</button>
        <button class="secondary-button" id="start-convert" type="button" ${state.plans.convert ? "" : "disabled"}>Run Convert</button>
        <button class="secondary-button" id="plan-tensorboard" type="button">Plan TensorBoard</button>
        <button class="secondary-button" id="start-tensorboard" type="button" ${state.plans.tensorboard ? "" : "disabled"}>Start TensorBoard</button>
        ${state.plans.tensorboard?.url ? `<a class="link-button" href="${escapeHtml(state.plans.tensorboard.url)}" target="_blank">Open TensorBoard</a>` : ""}
        <button class="secondary-button" id="plan-sample" type="button">Plan Sample</button>
        <button class="secondary-button" id="start-sample" type="button" ${state.plans.sample ? "" : "disabled"}>Generate Sample</button>
      </div>
      <div class="grid two">
        <div><h3>Setup / Training</h3>${state.plans.engineSetup ? `<pre>${escapeHtml(state.plans.engineSetup.displayCommand)}</pre>` : ""}${renderLogs(setupId)}${state.plans.train ? `<pre>${escapeHtml(state.plans.train.displayCommand)}</pre>` : ""}${renderTrainingMetrics(trainId)}${renderLogs(trainId)}</div>
        <div><h3>Convert / TensorBoard / Sample</h3>${state.plans.convert ? `<pre>${escapeHtml(state.plans.convert.displayCommand)}</pre>` : ""}${renderLogs(convertId)}${state.plans.tensorboard ? `<pre>${escapeHtml(state.plans.tensorboard.displayCommand)}</pre>` : ""}${state.plans.sample ? `<pre>${escapeHtml(state.plans.sample.displayCommand)}</pre>` : ""}${renderLogs(tbId)}${renderLogs(sampleId)}</div>
      </div>
    </section>`;
}

function renderJobContextMenu(): string {
  const menu = state.jobContextMenu;
  if (!menu) return "";
  const left = Math.max(8, Math.min(menu.x, window.innerWidth - 220));
  const top = Math.max(8, Math.min(menu.y, window.innerHeight - 190));
  const jobName = escapeHtml(menu.jobName);
  return `
    <div class="context-menu" style="left: ${left}px; top: ${top}px;" role="menu" aria-label="Job actions">
      <button type="button" role="menuitem" data-job-action="open" data-job-name="${jobName}">Open</button>
      <button type="button" role="menuitem" data-job-action="rename" data-job-name="${jobName}">Rename F2</button>
      <button type="button" role="menuitem" data-job-action="clone" data-job-name="${jobName}">Clone</button>
      <button type="button" role="menuitem" class="danger-menu-item" data-job-action="delete" data-job-name="${jobName}">Delete</button>
    </div>`;
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
        <div class="job-list">${state.jobs.map((job) => `<button class="job-row ${job.name === state.selectedJobName ? "selected" : ""}" data-job-name="${escapeHtml(job.name)}" type="button" title="Right-click for job actions. F2 renames the selected job."><strong>${escapeHtml(job.displayName)}</strong><span>${escapeHtml(job.datasetImageDir || "no dataset")}</span><span class="mono">${escapeHtml(job.updatedAt || "")}</span></button>`).join("")}</div>
      </div>
      ${renderJobEditor()}
    </section>
    ${renderJobContextMenu()}`;
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
      <div class="path-picker">
        <label>Model dir<input id="tagger-model-dir" value="${escapeHtml(state.tagger.modelDir || status.modelDir || state.settings?.taggerModelDir || "")}"></label>
        <button class="secondary-button compact-button" id="choose-tagger-model-dir" type="button">Browse</button>
      </div>
      <dl class="details model-status-list">
        <div><dt>Model</dt><dd><strong>${escapeHtml(status.modelName || "WD14 ConvNeXt Tagger v2")}</strong><br><span class="mono">${escapeHtml(status.modelId || "SmilingWolf/wd-v1-4-convnext-tagger-v2")}</span></dd></div>
        <div><dt>${escapeHtml(status.modelFileName || "wd-v1-4-convnext-tagger-v2.onnx")}</dt><dd>${statusPill(Boolean(status.modelExists), "Found", "Missing")}</dd></div>
        <div><dt>${escapeHtml(status.tagsFileName || "wd-v1-4-convnext-tagger-v2-selected_tags.csv")}</dt><dd>${statusPill(Boolean(status.tagsExists), "Found", "Missing")}</dd></div>
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
        ${(["dashboard", "agent", "dataset", "jobs", "tagger"] as ViewId[]).map((view) => `<button class="nav-item ${state.view === view ? "active" : ""}" data-view="${view}" type="button">${view === "dashboard" ? "Dashboard" : view === "agent" ? "Agent Autopilot" : view === "dataset" ? "Dataset Studio" : view === "jobs" ? "Jobs" : "WD14 Tagger"}</button>`).join("")}
      </nav>
      <div class="content">${state.view === "agent" ? renderAgentAutopilot() : state.view === "dataset" ? renderDatasetStudio() : state.view === "jobs" ? renderJobs() : state.view === "tagger" ? renderTagger() : renderDashboard()}</div>
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

function refreshDatasetStatusMessages(): void {
  const messages = document.querySelector<HTMLElement>("#dataset-status-messages");
  if (messages) messages.innerHTML = renderDatasetStatusMessages();
}

function setElementText(id: string, value: string | number): void {
  const element = document.querySelector<HTMLElement>(`#${id}`);
  if (element) element.textContent = String(value);
}

function refreshDatasetSummary(): void {
  const scan = state.dataset.scan;
  if (!scan) return;
  setElementText("dataset-summary-images", scan.imageCount);
  setElementText("dataset-summary-captions", scan.captionCount);
  setElementText("dataset-summary-missing", scan.missingCaptionCount);
  setElementText("dataset-summary-orphans", scan.orphanCaptionCount);
}

function datasetRowButtons(): HTMLButtonElement[] {
  return Array.from(document.querySelectorAll<HTMLButtonElement>(".dataset-row"));
}

function refreshDatasetRowSelection(): void {
  datasetRowButtons().forEach((button) => {
    button.classList.toggle("selected", button.dataset.imagePath === state.dataset.selectedImagePath);
  });
}

function refreshDatasetRow(item: DatasetItem): void {
  const row = datasetRowButtons().find((button) => button.dataset.imagePath === item.imagePath);
  if (!row) return;
  row.innerHTML = renderDatasetRowContent(item);
  row.classList.toggle("selected", item.imagePath === state.dataset.selectedImagePath);
}

function refreshDatasetSelectionView(): void {
  refreshDatasetRowSelection();
  const panel = document.querySelector<HTMLElement>("#dataset-editor-panel");
  if (!panel) {
    render();
    return;
  }
  panel.innerHTML = renderDatasetEditorPanel(selectedDatasetItem(), state.dataset.scan);
  bindDatasetEditorEvents();
}

function selectDatasetImage(imagePath: string): void {
  const item = state.dataset.scan?.items.find((entry) => entry.imagePath === imagePath);
  if (!item) return;
  state.dataset.selectedImagePath = item.imagePath;
  state.dataset.draftCaption = item.captionText;
  updateActiveDatasetProfile();
  queuePersistUiState();
  refreshDatasetSelectionView();
}

function bindDatasetRowEvents(): void {
  document.querySelectorAll<HTMLButtonElement>(".dataset-row").forEach((button) => {
    button.addEventListener("click", () => selectDatasetImage(button.dataset.imagePath || ""));
  });
}

function bindDatasetEditorEvents(): void {
  document.querySelector("#caption-editor")?.addEventListener("input", (event) => state.dataset.draftCaption = (event.target as HTMLTextAreaElement).value);
  document.querySelector("#tag-add-input")?.addEventListener("keydown", (event) => void handleTagAddInput(event as KeyboardEvent));
  document.querySelectorAll<HTMLButtonElement>("[data-remove-tag-index]").forEach((button) => {
    button.addEventListener("click", () => void removeSelectedTag(Number(button.dataset.removeTagIndex)));
  });
  document.querySelector("#save-caption")?.addEventListener("click", () => void saveSelectedCaption());
  document.querySelector("#run-tagger-from-dataset")?.addEventListener("click", () => void runTaggerForCurrentDataset());
  document.querySelector("#reload-dataset")?.addEventListener("click", () => void scanDataset());
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

function applyAgentProviderDefaults(providerId: string): void {
  const provider = agentProviders().find((item) => item.id === providerId);
  if (!provider) return;
  state.agent.provider = provider.id;
  state.agent.model = provider.model || provider.defaultModel || state.agent.model;
  state.agent.baseUrl = provider.baseUrl || provider.defaultBaseUrl || "";
  state.agent.apiKeyEnv = provider.apiKeyEnv || "";
}

function syncAgentControls(): void {
  state.agent.sourceRoot = textValue("agent-source-root").trim();
  state.agent.jobName = textValue("agent-job-name").trim();
  state.agent.goal = textValue("agent-goal").trim();
  const intent = textValue("agent-intent");
  if (["character", "style", "concept"].includes(intent)) state.agent.intent = intent as any;
  state.agent.triggerTag = textValue("agent-trigger-tag").trim();
  state.agent.provider = textValue("agent-provider") || state.agent.provider;
  state.agent.model = textValue("agent-model").trim();
  state.agent.baseUrl = textValue("agent-base-url").trim();
  state.agent.apiKeyEnv = textValue("agent-api-key-env").trim();
  state.agent.architecture = textValue("agent-architecture") || "anima";
  state.agent.engineId = textValue("agent-engine-id") || state.settings?.defaultEngineId || "";
  state.agent.gpuIds = textValue("agent-gpu-ids").trim() || "0";
  state.agent.multiGpuMode = textValue("agent-multi-gpu-mode") || "single";
  state.agent.imageMaxSide = Math.max(512, numberValue("agent-image-max-side", 1536));
  const outputFormat = textValue("agent-output-format");
  if (["keep", "png", "jpg"].includes(outputFormat)) state.agent.outputFormat = outputFormat as any;
  state.agent.callModel = Boolean(document.querySelector<HTMLInputElement>("#agent-call-model")?.checked);
}

function bindEvents(): void {
  document.querySelectorAll<HTMLElement>("[data-view]").forEach((element) => element.addEventListener("click", () => {
    const view = element.dataset.view as ViewId;
    state.view = view;
    queuePersistUiState();
    render();
  }));
  document.querySelector("#refresh-health")?.addEventListener("click", () => void loadInitial());
  document.querySelector("#choose-agent-source")?.addEventListener("click", () => void chooseAgentSourceRoot());
  document.querySelector("#build-agent-plan")?.addEventListener("click", () => { syncAgentControls(); queuePersistUiState(); void buildAgentPlan(); });
  document.querySelector("#agent-provider")?.addEventListener("change", () => {
    applyAgentProviderDefaults(textValue("agent-provider"));
    queuePersistUiState();
    render();
  });
  document.querySelector("#open-agent-job")?.addEventListener("click", () => void openAgentJob());
  document.querySelector("#start-agent-preprocess")?.addEventListener("click", () => void startAgentPlan("preprocess"));
  document.querySelector("#start-agent-tagger")?.addEventListener("click", () => void startAgentPlan("tagger"));
  document.querySelector("#start-agent-train")?.addEventListener("click", () => void startAgentPlan("train"));
  document.querySelector("#register-dataset")?.addEventListener("click", () => void registerDataset());
  document.querySelector("#dataset-profile")?.addEventListener("change", () => { void selectDatasetProfile(textValue("dataset-profile")); });
  document.querySelector("#rename-dataset-profile")?.addEventListener("click", () => void renameDatasetProfile());
  document.querySelector("#save-dataset-profile")?.addEventListener("click", () => { syncDatasetControls(); void saveCurrentDatasetProfile(); });
  document.querySelector("#forget-dataset-profile")?.addEventListener("click", () => void forgetDatasetProfile());
  document.querySelector("#scan-dataset")?.addEventListener("click", () => { syncDatasetControls(); queuePersistUiState(); void scanDataset(); });
  bindDatasetRowEvents();
  bindDatasetEditorEvents();
  document.querySelector("#apply-bulk-operation")?.addEventListener("click", () => { syncBulkControls(); void applyBulkOperation(); });
  document.querySelector("#create-job")?.addEventListener("click", () => void createJob());
  document.querySelector("#reload-jobs")?.addEventListener("click", () => void loadJobs());
  document.querySelectorAll<HTMLButtonElement>(".job-row").forEach((button) => {
    button.addEventListener("click", () => {
      state.jobContextMenu = undefined;
      void selectJob(button.dataset.jobName || "");
    });
    button.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      void openJobContextMenu(button.dataset.jobName || "", event.clientX, event.clientY);
    });
  });
  document.querySelectorAll<HTMLButtonElement>("[data-job-action]").forEach((button) => {
    button.addEventListener("click", () => void runJobMenuAction(button.dataset.jobAction || "", button.dataset.jobName || ""));
  });
  document.querySelector("#save-job")?.addEventListener("click", () => { syncJobDraft(); void saveJob(); });
  document.querySelector("#build-job-files")?.addEventListener("click", () => { syncJobDraft(); void buildJobFiles(); });
  document.querySelector("#clone-job")?.addEventListener("click", () => { syncJobDraft(); void cloneJob(); });
  document.querySelector("#delete-job")?.addEventListener("click", () => void deleteJob());
  document.querySelector("#plan-engine-setup")?.addEventListener("click", () => { syncJobDraft(); void planEngineSetup(); });
  document.querySelector("#start-engine-setup")?.addEventListener("click", () => void startPlan("engineSetup"));
  document.querySelector("#plan-train")?.addEventListener("click", () => { syncJobDraft(); void planTrain(); });
  document.querySelector("#start-train")?.addEventListener("click", () => void startPlan("train"));
  document.querySelector("#plan-convert")?.addEventListener("click", () => { syncJobDraft(); void planComfyConvert(); });
  document.querySelector("#start-convert")?.addEventListener("click", () => void startPlan("convert"));
  document.querySelector("#plan-tensorboard")?.addEventListener("click", () => void planTensorBoard());
  document.querySelector("#start-tensorboard")?.addEventListener("click", () => void startPlan("tensorboard"));
  document.querySelector("#plan-sample")?.addEventListener("click", () => void planSample());
  document.querySelector("#start-sample")?.addEventListener("click", () => void startPlan("sample"));
  document.querySelectorAll<HTMLButtonElement>("[data-stop-process]").forEach((button) => button.addEventListener("click", () => void stopProcess(button.dataset.stopProcess || "")));
  document.querySelector("#refresh-tagger")?.addEventListener("click", () => { syncTaggerControls(); queuePersistUiState(); void loadTaggerStatus(); });
  document.querySelector("#install-tagger-deps")?.addEventListener("click", () => void startTaggerPlan("tagger_install_deps_plan"));
  document.querySelector("#download-tagger-model")?.addEventListener("click", () => void downloadTaggerModel());
  document.querySelector("#choose-tagger-model-dir")?.addEventListener("click", () => void chooseTaggerModelDir());
  document.querySelector("#tagger-model-dir")?.addEventListener("change", () => { syncTaggerControls(); void persistTaggerModelDir().then(loadTaggerStatus); });
  document.querySelector("#choose-tagger-root")?.addEventListener("click", () => void chooseTaggerRoot());
  document.querySelector("#run-tagger")?.addEventListener("click", () => { syncTaggerControls(); queuePersistUiState(); void runTagger(); });
  document.onkeydown = handleGlobalKeyDown;
  document.onclick = handleDocumentClick;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

function handleGlobalKeyDown(event: KeyboardEvent): void {
  if (event.key === "Escape" && state.jobContextMenu) {
    state.jobContextMenu = undefined;
    render();
    return;
  }
  if (state.view !== "jobs" || isEditableTarget(event.target)) return;
  if (event.key === "F2" && state.selectedJobName) {
    event.preventDefault();
    void renameJob(state.selectedJobName);
  }
  if (event.key === "Delete" && state.selectedJobName) {
    event.preventDefault();
    void deleteJobByName(state.selectedJobName);
  }
}

function handleDocumentClick(event: MouseEvent): void {
  if (!state.jobContextMenu) return;
  if (event.target instanceof HTMLElement && event.target.closest(".context-menu, .job-row")) return;
  state.jobContextMenu = undefined;
  render();
}

async function loadInitial(): Promise<void> {
  state.healthLoading = true;
  render();
  try {
    const [settings, health] = await Promise.all([bridgeData<Settings>("settings_get"), bridgeData<HealthData>("health_check")]);
    state.settings = settings;
    state.tagger.modelDir = settings.taggerModelDir || state.tagger.modelDir;
    state.agent.provider = settings.agent?.provider || state.agent.provider;
    state.agent.model = settings.agent?.model || state.agent.model;
    state.agent.baseUrl = settings.agent?.baseUrl || state.agent.baseUrl;
    state.agent.apiKeyEnv = settings.agent?.apiKeyEnv || state.agent.apiKeyEnv;
    state.agent.imageMaxSide = Number(settings.agent?.imageMaxSide || state.agent.imageMaxSide);
    if (["keep", "png", "jpg"].includes(String(settings.agent?.outputFormat || ""))) state.agent.outputFormat = settings.agent?.outputFormat as any;
    state.agent.engineId = settings.defaultEngineId || state.agent.engineId;
    state.agent.architecture = String(settings.defaults?.architecture || state.agent.architecture);
    state.agent.gpuIds = String(settings.defaults?.gpuIds || state.agent.gpuIds);
    state.agent.multiGpuMode = String(settings.defaults?.multiGpuMode || state.agent.multiGpuMode);
    applyPersistedUi(settings);
    state.health = health;
    if (!state.agent.model || !state.agent.apiKeyEnv || !state.agent.baseUrl) {
      applyAgentProviderDefaults(state.agent.provider);
    }
    state.healthError = undefined;
    await Promise.all([loadJobs(), loadTaggerStatus()]);
    if (state.selectedJobName && state.jobs.some((job) => job.name === state.selectedJobName)) {
      await selectJob(state.selectedJobName);
    }
    if (state.view === "dataset" && state.dataset.root) {
      await scanDataset();
    }
  } catch (error) {
    state.healthError = error instanceof Error ? error.message : String(error);
  } finally {
    state.healthLoading = false;
    render();
  }
}

async function chooseAgentSourceRoot(): Promise<void> {
  const result = await open({ directory: true, multiple: false, title: "Select source image folder" });
  if (typeof result !== "string") return;
  state.agent.sourceRoot = result;
  if (!state.agent.jobName) {
    state.agent.jobName = (result.split(/[\\/]/).pop() || "autopilot_lora").replaceAll(" ", "_");
  }
  state.dataset.root = result;
  state.tagger.datasetRoot = result;
  queuePersistUiState();
  render();
}

async function buildAgentPlan(): Promise<void> {
  if (!state.agent.sourceRoot) return;
  state.agent.loading = true;
  state.agent.error = undefined;
  state.agent.message = undefined;
  render();
  try {
    const result = await bridgeData<AgentAutopilotResult>("agent_autopilot_plan", {
      sourceRoot: state.agent.sourceRoot,
      jobName: state.agent.jobName || undefined,
      goal: state.agent.goal,
      intent: state.agent.intent,
      triggerTag: state.agent.triggerTag,
      provider: state.agent.provider,
      model: state.agent.model,
      baseUrl: state.agent.baseUrl,
      apiKeyEnv: state.agent.apiKeyEnv,
      architecture: state.agent.architecture,
      engineId: state.agent.engineId,
      gpuIds: state.agent.gpuIds,
      multiGpuMode: state.agent.multiGpuMode,
      imageMaxSide: state.agent.imageMaxSide,
      outputFormat: state.agent.outputFormat,
      callModel: state.agent.callModel,
    });
    state.agent.result = result;
    state.agent.jobName = result.job.name;
    state.agent.message = `Generated ${result.job.name}.`;
    state.plans.agentPreprocess = result.preprocessPlan;
    state.plans.agentTagger = result.taggerPlan;
    state.plans.agentTrain = result.trainPlan;
    await loadJobs();
    queuePersistUiState();
  } catch (error) {
    state.agent.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.agent.loading = false;
    render();
  }
}

async function openAgentJob(): Promise<void> {
  const jobName = state.agent.result?.job?.name;
  if (!jobName) return;
  await selectJob(jobName);
  state.view = "jobs";
  queuePersistUiState();
  render();
}

async function startAgentPlan(kind: "preprocess" | "tagger" | "train"): Promise<void> {
  const plan = kind === "preprocess" ? state.agent.result?.preprocessPlan : kind === "tagger" ? state.agent.result?.taggerPlan : state.agent.result?.trainPlan;
  if (!plan) return;
  await startProcess(plan);
}

async function registerDataset(): Promise<void> {
  const result = await open({ directory: true, multiple: false, title: "Select dataset folder" });
  if (typeof result !== "string") return;
  const name = window.prompt("Dataset name", folderName(result))?.trim();
  if (!name) return;
  state.dataset.root = result;
  state.dataset.captionExtension = state.dataset.captionExtension || ".txt";
  state.tagger.datasetRoot = result;
  const profile = upsertDatasetProfile({
    id: slug(name),
    name,
    root: result,
    captionExtension: state.dataset.captionExtension,
    minPixels: state.dataset.minPixels,
    lastSelectedImagePath: "",
  });
  setDatasetFromProfile(profile);
  queuePersistUiState();
  await scanDataset();
}

async function selectDatasetProfile(id: string): Promise<void> {
  const profile = state.dataset.profiles.find((item) => item.id === id);
  if (!profile) return;
  setDatasetFromProfile(profile);
  queuePersistUiState();
  await scanDataset();
}

async function saveCurrentDatasetProfile(): Promise<void> {
  if (!state.dataset.root) return;
  const existing = activeDatasetProfile();
  const name = existing?.name || window.prompt("Dataset name", folderName(state.dataset.root))?.trim();
  if (!name) return;
  const profile = upsertDatasetProfile({
    id: existing?.id || slug(name),
    name,
    root: state.dataset.root,
    captionExtension: state.dataset.captionExtension,
    minPixels: state.dataset.minPixels,
    lastSelectedImagePath: state.dataset.selectedImagePath || "",
  });
  state.dataset.activeProfileId = profile.id;
  state.tagger.datasetRoot = profile.root;
  state.dataset.message = `Saved dataset "${profile.name}".`;
  queuePersistUiState();
  render();
}

async function renameDatasetProfile(): Promise<void> {
  const profile = activeDatasetProfile();
  if (!profile) return;
  const name = window.prompt("Dataset name", profile.name)?.trim();
  if (!name || name === profile.name) return;
  profile.name = name;
  profile.updatedAt = new Date().toISOString();
  state.dataset.message = `Renamed dataset to "${name}".`;
  queuePersistUiState();
  render();
}

async function forgetDatasetProfile(): Promise<void> {
  const profile = activeDatasetProfile();
  if (!profile) return;
  if (!window.confirm(`Forget dataset "${profile.name}"? Files will not be deleted.`)) return;
  state.dataset.profiles = state.dataset.profiles.filter((item) => item.id !== profile.id);
  const next = state.dataset.profiles[0];
  if (next) {
    setDatasetFromProfile(next);
    await scanDataset();
  } else {
    state.dataset.activeProfileId = undefined;
    state.dataset.root = "";
    state.dataset.scan = undefined;
    state.dataset.selectedImagePath = undefined;
    state.dataset.draftCaption = "";
    render();
  }
  queuePersistUiState();
}

async function scanDataset(): Promise<void> {
  if (!state.dataset.root) return;
  updateActiveDatasetProfile();
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
    updateActiveDatasetProfile();
    queuePersistUiState();
  } catch (error) {
    state.dataset.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.dataset.loading = false;
    render();
  }
}

function updateSelectedCaptionLocally(result: SaveCaptionResult): void {
  const selected = selectedDatasetItem();
  state.dataset.draftCaption = result.captionText;
  if (!selected) return;

  const scan = state.dataset.scan;
  const hadCaption = selected.captionExists;
  const hadMissingIssue = selected.issues.includes("missing_caption");
  const hadEmptyIssue = selected.issues.includes("empty_caption");
  const isEmpty = !result.captionText.trim();

  selected.captionExists = true;
  selected.captionText = result.captionText;
  selected.tags = result.tags;
  selected.tagCount = result.tagCount;
  selected.issues = selected.issues.filter((issue) => issue !== "missing_caption" && issue !== "empty_caption");
  if (isEmpty) selected.issues.push("empty_caption");

  if (scan) {
    if (!hadCaption) scan.captionCount += 1;
    if (hadMissingIssue) scan.missingCaptionCount = Math.max(0, scan.missingCaptionCount - 1);
    if (hadEmptyIssue && !isEmpty) scan.emptyCaptionCount = Math.max(0, scan.emptyCaptionCount - 1);
    if (!hadEmptyIssue && isEmpty) scan.emptyCaptionCount += 1;
    scan.issueCount = scan.items.filter((item) => item.issues.length).length;
  }
}

async function saveSelectedCaption(): Promise<void> {
  const selected = selectedDatasetItem();
  if (!selected) return;
  const editor = document.querySelector<HTMLTextAreaElement>("#caption-editor");
  if (editor) state.dataset.draftCaption = editor.value;
  try {
    const result = await bridgeData<SaveCaptionResult>("dataset_save_caption", { captionPath: selected.captionPath, text: state.dataset.draftCaption });
    updateSelectedCaptionLocally(result);
    state.dataset.message = `Saved ${result.tagCount} tags.`;
    state.dataset.error = undefined;
    refreshDatasetStatusMessages();
    refreshDatasetSummary();
    refreshDatasetRow(selected);
    refreshDatasetSelectionView();
  } catch (error) {
    state.dataset.error = error instanceof Error ? error.message : String(error);
    refreshDatasetStatusMessages();
  }
}

function setSelectedTags(tags: string[]): void {
  const normalized = tags.map((tag) => tag.trim()).filter(Boolean);
  const text = joinCaptionTags(normalized);
  state.dataset.draftCaption = text;
  const selected = selectedDatasetItem();
  if (selected) {
    selected.captionText = text;
    selected.tags = normalized;
    selected.tagCount = normalized.length;
  }
}

async function removeSelectedTag(index: number): Promise<void> {
  if (!Number.isInteger(index)) return;
  const tags = splitCaptionTags(state.dataset.draftCaption);
  if (index < 0 || index >= tags.length) return;
  tags.splice(index, 1);
  setSelectedTags(tags);
  await saveSelectedCaption();
}

function tagsFromInput(value: string): string[] {
  return value.split(/[,\n]/).map((tag) => tag.trim()).filter(Boolean);
}

async function addTagsToSelected(raw: string): Promise<void> {
  const incoming = tagsFromInput(raw);
  if (!incoming.length) return;

  const tags = splitCaptionTags(state.dataset.draftCaption);
  const existing = new Set(tags.map((tag) => tag.toLowerCase()));
  let changed = false;
  for (const tag of incoming) {
    const key = tag.toLowerCase();
    if (existing.has(key)) continue;
    tags.push(tag);
    existing.add(key);
    changed = true;
  }

  if (!changed) return;
  setSelectedTags(tags);
  await saveSelectedCaption();
}

async function handleTagAddInput(event: KeyboardEvent): Promise<void> {
  const input = event.target instanceof HTMLInputElement ? event.target : undefined;
  if (!input) return;
  if (event.key !== "Enter" && event.key !== ",") return;
  event.preventDefault();
  const value = input.value.trim();
  input.value = "";
  await addTagsToSelected(value);
}

async function runTaggerForCurrentDataset(): Promise<void> {
  if (!state.dataset.root) return;
  state.tagger.datasetRoot = state.dataset.root;
  state.view = "tagger";
  queuePersistUiState();
  render();
  await runTagger();
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
  queuePersistUiState();
  await loadJobs();
  render();
}

async function selectJob(name: string): Promise<void> {
  if (!name) return;
  const result = await bridgeData<{ job: Job }>("job_get", { name });
  state.selectedJobName = name;
  state.jobDraft = result.job;
  state.jobMessage = undefined;
  state.jobError = undefined;
  queuePersistUiState();
  render();
}

async function openJobContextMenu(name: string, x: number, y: number): Promise<void> {
  if (!name) return;
  state.jobContextMenu = { jobName: name, x, y };
  if (state.selectedJobName !== name || state.jobDraft?.name !== name) {
    await selectJob(name);
  }
  state.jobContextMenu = { jobName: name, x, y };
  render();
}

async function runJobMenuAction(action: string, name: string): Promise<void> {
  if (!name) return;
  if (action === "open") {
    state.jobContextMenu = undefined;
    await selectJob(name);
    return;
  }
  if (action === "rename") {
    await renameJob(name);
    return;
  }
  if (action === "clone") {
    await cloneJobByName(name);
    return;
  }
  if (action === "delete") {
    await deleteJobByName(name);
  }
}

async function saveJob(): Promise<void> {
  if (!state.jobDraft) return;
  const result = await bridgeData<{ job: Job }>("job_save", { job: state.jobDraft });
  state.jobDraft = result.job;
  state.selectedJobName = result.job.name;
  state.jobMessage = "Saved job.";
  await loadJobs();
  queuePersistUiState();
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
  await cloneJobByName(state.jobDraft.name);
}

async function deleteJob(): Promise<void> {
  await deleteJobByName(state.jobDraft?.name);
}

async function cloneJobByName(source: string): Promise<void> {
  if (!source) return;
  try {
    const target = `${source}_copy`;
    const result = await bridgeData<{ job: Job }>("job_clone", { source, target });
    state.jobDraft = result.job;
    state.selectedJobName = result.job.name;
    state.jobContextMenu = undefined;
    state.jobMessage = `Cloned ${source}.`;
    state.jobError = undefined;
    await loadJobs();
    queuePersistUiState();
  } catch (error) {
    state.jobError = error instanceof Error ? error.message : String(error);
  } finally {
    render();
  }
}

async function renameJob(source: string): Promise<void> {
  if (!source) return;
  const target = window.prompt("Rename job", source)?.trim();
  if (!target || target === source) {
    state.jobContextMenu = undefined;
    render();
    return;
  }
  try {
    const result = await bridgeData<{ job: Job }>("job_rename", { source, target });
    state.jobDraft = result.job;
    state.selectedJobName = result.job.name;
    state.jobContextMenu = undefined;
    state.jobMessage = `Renamed to ${result.job.name}.`;
    state.jobError = undefined;
    await loadJobs();
    queuePersistUiState();
  } catch (error) {
    state.jobError = error instanceof Error ? error.message : String(error);
  } finally {
    render();
  }
}

async function deleteJobByName(name?: string): Promise<void> {
  if (!name) return;
  if (!window.confirm(`Delete job "${name}"?`)) {
    state.jobContextMenu = undefined;
    render();
    return;
  }
  try {
    await bridgeData("job_delete", { name });
    if (state.selectedJobName === name) {
      state.jobDraft = undefined;
      state.selectedJobName = undefined;
    }
    state.jobContextMenu = undefined;
    state.jobMessage = `Deleted ${name}.`;
    state.jobError = undefined;
    await loadJobs();
    queuePersistUiState();
  } catch (error) {
    state.jobError = error instanceof Error ? error.message : String(error);
  } finally {
    render();
  }
}

async function planEngineSetup(): Promise<void> {
  await saveJob();
  if (!state.jobDraft) return;
  const result = await bridgeData<{ plan: LaunchPlan; errors?: string[] }>("engine_setup_plan", { name: state.jobDraft.name, allowInvalid: true });
  state.plans.engineSetup = result.plan;
  state.jobError = result.errors?.length ? result.errors.join("\n") : undefined;
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

async function planComfyConvert(): Promise<void> {
  await saveJob();
  if (!state.jobDraft) return;
  const result = await bridgeData<{ plan: LaunchPlan; errors?: string[] }>("comfy_convert_plan", { name: state.jobDraft.name, allowInvalid: true });
  state.plans.convert = result.plan;
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

async function startPlan(kind: "engineSetup" | "train" | "convert" | "tensorboard" | "sample"): Promise<void> {
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
  await persistTaggerModelDir();
  await loadTaggerStatus();
}

async function persistTaggerModelDir(): Promise<void> {
  if (!state.settings) return;
  const modelDir = state.tagger.modelDir.trim();
  if (!modelDir) return;
  if (modelDir === state.settings.taggerModelDir) return;
  state.settings = await bridgeData<Settings>("settings_save", { ...state.settings, taggerModelDir: modelDir });
  state.tagger.modelDir = state.settings.taggerModelDir;
}

async function downloadTaggerModel(): Promise<void> {
  syncTaggerControls();
  await persistTaggerModelDir();
  await startTaggerPlan("tagger_download_plan");
}

async function chooseTaggerRoot(): Promise<void> {
  const result = await open({ directory: true, multiple: false, title: "Select dataset folder" });
  if (typeof result !== "string") return;
  state.tagger.datasetRoot = result;
  queuePersistUiState();
  render();
}

async function runTagger(): Promise<void> {
  syncTaggerControls();
  queuePersistUiState();
  await persistTaggerModelDir();
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
