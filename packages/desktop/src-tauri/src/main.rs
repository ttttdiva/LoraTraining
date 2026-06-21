#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

const MAX_LOG_LINES: usize = 5_000;

#[derive(Default)]
struct ProcessRegistry {
    processes: Mutex<HashMap<String, ProcessEntry>>,
}

#[derive(Clone)]
struct ProcessEntry {
    kind: String,
    command: String,
    child: Arc<Mutex<Child>>,
    logs: Arc<Mutex<VecDeque<String>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LaunchPlan {
    id: String,
    kind: String,
    cwd: String,
    command: String,
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

fn project_root() -> PathBuf {
    if let Ok(root) = std::env::var("LORA_TRAINING_ROOT") {
        return PathBuf::from(root);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = manifest_dir
        .parent()
        .and_then(|desktop| desktop.parent())
        .and_then(|packages| packages.parent())
    {
        return root.to_path_buf();
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn python_path_env(project_root: &PathBuf) -> String {
    let bridge_root = project_root.join("python");
    let existing = std::env::var("PYTHONPATH").unwrap_or_default();
    if existing.is_empty() {
        return bridge_root.to_string_lossy().to_string();
    }

    let separator = if cfg!(windows) { ";" } else { ":" };
    format!("{}{}{}", bridge_root.to_string_lossy(), separator, existing)
}

fn resolve_python(project_root: &PathBuf) -> String {
    if let Ok(python) = std::env::var("LORA_TRAINING_PYTHON") {
        if !python.trim().is_empty() {
            return python;
        }
    }

    let local_python = project_root
        .join(".venv")
        .join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" });
    if local_python.exists() {
        return local_python.to_string_lossy().to_string();
    }

    "python".to_string()
}

fn read_pipe(mut pipe: impl Read) -> String {
    let mut output = String::new();
    let _ = pipe.read_to_string(&mut output);
    output
}

fn push_log(logs: &Arc<Mutex<VecDeque<String>>>, line: String) {
    let mut locked = logs.lock().unwrap();
    locked.push_back(line);
    while locked.len() > MAX_LOG_LINES {
        locked.pop_front();
    }
}

fn spawn_log_reader<R: Read + Send + 'static>(reader: R, logs: Arc<Mutex<VecDeque<String>>>, prefix: &'static str) {
    std::thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            push_log(&logs, format!("{prefix}{line}"));
        }
    });
}

fn run_bridge_sync(job: String, payload: Value) -> Result<Value, String> {
    let root = project_root();
    let python = resolve_python(&root);
    let payload_json = payload.to_string();

    let mut command = Command::new(&python);
    command
        .args([
            "-m",
            "lora_training_gui.bridge",
            "--job",
            &job,
            "--payload-json",
            &payload_json,
        ])
        .current_dir(&root)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONPATH", python_path_env(&root))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to start Python bridge: {error}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture bridge stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture bridge stderr".to_string())?;

    let stdout_thread = std::thread::spawn(move || read_pipe(stdout));
    let stderr_thread = std::thread::spawn(move || read_pipe(stderr));

    let status = child
        .wait()
        .map_err(|error| format!("failed to wait for Python bridge: {error}"))?;
    let stdout_text = stdout_thread.join().unwrap_or_default();
    let stderr_text = stderr_thread.join().unwrap_or_default();

    let trimmed = stdout_text.trim();
    if trimmed.is_empty() {
        return Ok(json!({
            "ok": false,
            "bridge": {
                "status": "error",
                "job": job,
                "errors": [format!("bridge produced no stdout; exit={status}")]
            },
            "stderr": stderr_text
        }));
    }

    let parsed: Value = serde_json::from_str(trimmed)
        .map_err(|error| format!("bridge returned invalid JSON: {error}; stdout={trimmed}"))?;
    let ok = status.success()
        && parsed
            .get("status")
            .and_then(|status| status.as_str())
            == Some("ok");

    Ok(json!({
        "ok": ok,
        "bridge": parsed,
        "stderr": stderr_text
    }))
}

fn process_snapshot(entry: &ProcessEntry) -> Value {
    let mut child = entry.child.lock().unwrap();
    let status = child.try_wait().ok().flatten();
    let running = status.is_none();
    let exit_code = status.and_then(|status| status.code());
    let logs: Vec<String> = entry.logs.lock().unwrap().iter().cloned().collect();
    json!({
        "kind": entry.kind,
        "command": entry.command,
        "running": running,
        "exitCode": exit_code,
        "logs": logs
    })
}

fn kill_child_tree(child: &mut Child) {
    #[cfg(target_os = "windows")]
    {
        let pid = child.id().to_string();
        let _ = Command::new("taskkill")
            .args(["/PID", &pid, "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .and_then(|mut killer| killer.wait());
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = child.kill();
    }
}

#[tauri::command]
async fn run_bridge(job: String, payload: Value) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || run_bridge_sync(job, payload))
        .await
        .map_err(|error| format!("bridge task failed: {error}"))?
}

#[tauri::command]
fn start_process(registry: tauri::State<ProcessRegistry>, plan: Value) -> Result<Value, String> {
    let plan: LaunchPlan = serde_json::from_value(plan).map_err(|error| format!("invalid launch plan: {error}"))?;
    let mut registry_map = registry.processes.lock().unwrap();

    if let Some(existing) = registry_map.get(&plan.id) {
        let snapshot = process_snapshot(existing);
        if snapshot.get("running").and_then(|value| value.as_bool()) == Some(true) {
            return Err(format!("process is already running: {}", plan.id));
        }
    }

    let mut command = Command::new(&plan.command);
    command
        .args(&plan.args)
        .current_dir(&plan.cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in &plan.env {
        command.env(key, value);
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to start process {}: {error}", plan.id))?;

    let logs = Arc::new(Mutex::new(VecDeque::new()));
    push_log(&logs, format!("started: {} {}", plan.command, plan.args.join(" ")));

    if let Some(stdout) = child.stdout.take() {
        spawn_log_reader(stdout, logs.clone(), "");
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_log_reader(stderr, logs.clone(), "ERR: ");
    }

    let entry = ProcessEntry {
        kind: plan.kind,
        command: format!("{} {}", plan.command, plan.args.join(" ")),
        child: Arc::new(Mutex::new(child)),
        logs,
    };
    let snapshot = process_snapshot(&entry);
    registry_map.insert(plan.id.clone(), entry);
    Ok(json!({
        "id": plan.id,
        "status": snapshot
    }))
}

#[tauri::command]
fn process_status(registry: tauri::State<ProcessRegistry>, id: String) -> Result<Value, String> {
    let registry_map = registry.processes.lock().unwrap();
    let Some(entry) = registry_map.get(&id) else {
        return Ok(json!({
            "id": id,
            "status": {
                "running": false,
                "exitCode": null,
                "logs": []
            }
        }));
    };
    let snapshot = process_snapshot(entry);
    Ok(json!({
        "id": id,
        "status": snapshot
    }))
}

#[tauri::command]
fn stop_process(registry: tauri::State<ProcessRegistry>, id: String) -> Result<Value, String> {
    let registry_map = registry.processes.lock().unwrap();
    let Some(entry) = registry_map.get(&id) else {
        return Ok(json!({ "id": id, "stopped": false }));
    };
    {
        let mut child = entry.child.lock().unwrap();
        if child.try_wait().ok().flatten().is_none() {
            kill_child_tree(&mut child);
        }
    }
    push_log(&entry.logs, "stop requested".to_string());
    Ok(json!({ "id": id, "stopped": true }))
}

fn main() {
    tauri::Builder::default()
        .manage(ProcessRegistry::default())
        .invoke_handler(tauri::generate_handler![
            run_bridge,
            start_process,
            process_status,
            stop_process
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
