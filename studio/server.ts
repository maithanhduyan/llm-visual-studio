/**
 * server.ts — Máy chủ Bun cho Mini DeepSeek Studio.
 *
 * Nó làm hai việc:
 *   1. Trả trang web (Bun tự đóng gói TypeScript + CSS, không cần Vite).
 *   2. Giữ một tiến trình Python thường trực (analyze.py --serve) để chạy model.
 *
 * Giữ tiến trình thường trực là mẹo quan trọng: nạp PyTorch mất vài giây,
 * nếu mỗi lần phân tích lại nạp thì trang web sẽ rất chậm.
 *
 * Tiến trình Python đó KHÔNG nạp sẵn model nào. Model chỉ được nạp khi
 * trang web hỏi tới, và được giữ lại trong bộ nhớ cho lần sau.
 *
 * Chạy:  bun run server.ts
 */

import index from "./index.html";
import type { AnalyzeResult, ArchData, ModelOption, SpaceData, StudioMeta } from "./types";

const HERE = import.meta.dir;
const PYTHON = process.env.PYTHON ?? "python";
const FIRST_PORT = Number(process.env.PORT ?? 4173);
const TIMEOUT_MS = 120_000;

type Slot = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

let proc: ReturnType<typeof Bun.spawn> | null = null;
let readyPromise: Promise<ModelOption[]> | null = null;
let resolveReady: ((models: ModelOption[]) => void) | null = null;
let rejectReady: ((error: Error) => void) | null = null;
let workerSourceMtime = 0;

const pending: Slot[] = [];
let logs: string[] = [];

const describe = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** Một dòng JSON từ Python vừa tới. */
function handleLine(line: string) {
  let message: any;
  try {
    message = JSON.parse(line);
  } catch {
    return; // Python in ra thứ khác (ví dụ cảnh báo) -> bỏ qua
  }

  // Dòng báo "tiến trình đã sẵn sàng", kèm danh sách model.
  if (typeof message.ready === "boolean") {
    if (message.ready) resolveReady?.(message.models as ModelOption[]);
    else rejectReady?.(new Error(message.error ?? "Không khởi động được analyze.py."));
    return;
  }

  // Các dòng còn lại là câu trả lời, trả về đúng thứ tự đã hỏi.
  const slot = pending.shift();
  if (!slot) return;
  clearTimeout(slot.timer);
  if (message.ok) slot.resolve(message.result);
  else slot.reject(new Error(message.error ?? "Lỗi không rõ trong analyze.py"));
}

/** Tiến trình Python chết -> mọi câu đang chờ đều thất bại. */
function fail(error: Error) {
  rejectReady?.(error);
  while (pending.length) {
    const slot = pending.shift()!;
    clearTimeout(slot.timer);
    slot.reject(error);
  }
}

function drain(stream: ReadableStream<Uint8Array>, onLine: (line: string) => void) {
  (async () => {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let cut: number;
        while ((cut = buffer.indexOf("\n")) !== -1) {
          const line = buffer.slice(0, cut).trim();
          buffer = buffer.slice(cut + 1);
          if (line) onLine(line);
        }
      }
    } catch {
      // ống đã đóng
    }
  })();
}

function startWorker() {
  const child = Bun.spawn([PYTHON, `${HERE}/analyze.py`, "--serve"], {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    cwd: HERE,
  });
  proc = child;

  drain(child.stdout as ReadableStream<Uint8Array>, handleLine);

  // Phải đọc stderr, nếu không ống đầy và Python sẽ đứng im.
  drain(child.stderr as ReadableStream<Uint8Array>, (line) => {
    logs.push(line);
    logs = logs.slice(-20);
  });

  child.exited.then((code) => {
    // Nếu tiến trình này đã bị thay bằng tiến trình mới (do analyze.py
    // vừa được sửa) thì cái chết của nó là chủ ý, không phải lỗi.
    if (proc !== child) return;

    proc = null;
    fail(new Error(`Tiến trình Python đã dừng (mã ${code}). ${logs.at(-1) ?? ""}`.trim()));
  });
}

/** Nạp tiến trình Python nếu chưa có. Nạp lỗi thì lần sau thử lại được.
 *
 *  Nếu analyze.py vừa được sửa thì khởi động lại tiến trình Python — Python
 *  chỉ đọc source một lần lúc khởi động, nên không làm vậy thì sửa xong
 *  phải tắt máy chủ mở lại mới thấy tác dụng.
 */
async function ensureWorker(): Promise<ModelOption[]> {
  const source = Bun.file(`${HERE}/analyze.py`);
  const mtime = (await source.stat()).mtimeMs;

  if (readyPromise && mtime !== workerSourceMtime) {
    console.log("  analyze.py vừa thay đổi -> khởi động lại tiến trình Python");
    proc?.kill();
    readyPromise = null;
    resolveReady = null;
    rejectReady = null;
    pending.length = 0;
  }
  workerSourceMtime = mtime;

  if (readyPromise) return readyPromise;

  readyPromise = new Promise<ModelOption[]>((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
    try {
      startWorker();
    } catch (error) {
      reject(new Error(`Không chạy được "${PYTHON}". Đã cài Python chưa? (${describe(error)})`));
    }
  });

  readyPromise.catch(() => {
    readyPromise = null;
    resolveReady = null;
    rejectReady = null;
  });

  return readyPromise;
}

// Python xử lý lần lượt, nên ta xếp hàng các câu hỏi lại cho khớp thứ tự.
let chain: Promise<unknown> = Promise.resolve();

function request<T>(payload: Record<string, unknown>): Promise<T> {
  const run = chain.then(() => send<T>(payload));
  chain = run.catch(() => {});
  return run;
}

async function send<T>(payload: Record<string, unknown>): Promise<T> {
  await ensureWorker();

  const child = proc;
  if (!child) throw new Error("Chưa có tiến trình Python.");

  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      const at = pending.findIndex((slot) => slot.timer === timer);
      if (at >= 0) pending.splice(at, 1);
      reject(new Error("Model trả lời quá lâu."));
    }, TIMEOUT_MS);

    pending.push({ resolve: resolve as (value: unknown) => void, reject, timer });

    child.stdin.write(`${JSON.stringify(payload)}\n`);
    child.stdin.flush();
  });
}

const json = (data: unknown, status = 200) =>
  Response.json(data, { status, headers: { "cache-control": "no-store" } });

function modelOf(url: string): string {
  return new URL(url).searchParams.get("model") ?? "deepseek_lite";
}

function guarded(handler: () => Promise<unknown>) {
  return async () => {
    try {
      return json(await handler());
    } catch (error) {
      return json({ error: describe(error) }, 400);
    }
  };
}

async function listen() {
  for (let attempt = 0; attempt < 6; attempt += 1) {
    const port = FIRST_PORT + attempt;
    try {
      const server = Bun.serve({
        port,
        development: true,
        routes: {
          "/": index,

          "/api/models": {
            GET: guarded(async () => ({ models: await ensureWorker() })),
          },

          "/api/meta": {
            GET: (req) => guarded(() => request<StudioMeta>({ action: "meta", model: modelOf(req.url) }))(),
          },

          "/api/space": {
            GET: (req) => guarded(() => request<SpaceData>({ action: "space", model: modelOf(req.url) }))(),
          },

          "/api/architecture": {
            GET: (req) =>
              guarded(() => request<ArchData>({ action: "architecture", model: modelOf(req.url) }))(),
          },

          "/api/analyze": {
            POST: async (req) => {
              let body: any = {};
              try {
                body = await req.json();
              } catch {
                // thân request rỗng
              }
              try {
                const result = await request<AnalyzeResult>({
                  action: "analyze",
                  model: String(body?.model ?? "deepseek_lite"),
                  text: String(body?.text ?? ""),
                });
                return json(result);
              } catch (error) {
                return json({ error: describe(error) }, 400);
              }
            },
          },
        },
      });

      console.log(`\n  Mini DeepSeek Studio  ->  ${server.url}\n`);
      return;
    } catch (error) {
      if (attempt === 5) throw error;
    }
  }
}

process.on("SIGINT", () => {
  proc?.kill();
  process.exit(0);
});

await listen();
