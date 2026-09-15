/**
 * server.ts — Máy chủ nhỏ để xem lại các chuyến bay.
 *
 * Nó chỉ làm hai việc:
 *   1. Trả trang web (Bun tự đóng gói TypeScript + CSS)
 *   2. Trả danh sách chuyến bay và nội dung từng chuyến
 *
 * Chuyến bay do Python mô phỏng rồi ghi ra file JSON. Trình duyệt chỉ việc
 * đọc lại mà phát. Không cần Python chạy nền — mở trang là xem được ngay.
 *
 * Chạy:  bun run server.ts
 */

import index from "./index.html";

const HERE = import.meta.dir;
const FLIGHTS_DIR = `${HERE}/../flights`;
const FIRST_PORT = Number(process.env.PORT ?? 4180);

const json = (data: unknown, status = 200) =>
  Response.json(data, { status, headers: { "cache-control": "no-store" } });

async function listFlights() {
  const glob = new Bun.Glob("*.json");
  const names: string[] = [];

  for await (const name of glob.scan({ cwd: FLIGHTS_DIR })) {
    names.push(name.replace(/\.json$/, ""));
  }

  const flights = [];

  for (const name of names.sort()) {
    const file = Bun.file(`${FLIGHTS_DIR}/${name}.json`);

    try {
      const data = await file.json();
      flights.push({
        name,
        summary: data.summary ?? {},
        meta: data.meta ?? {},
        phases: (data.phases ?? []).map((p: { name: string }) => p.name),
      });
    } catch {
      // file hỏng thì bỏ qua
    }
  }

  return flights;
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

          // Trọng số model cho chế độ chạy sống. Trình duyệt tự chạy model
          // bằng JavaScript, nên nó cần file này — không có Python nào chạy
          // phía sau để trả lời từng câu.
          "/public/model.json": () => {
            const file = Bun.file(`${HERE}/public/model.json`);

            return new Response(file, {
              headers: { "content-type": "application/json" },
            });
          },

          "/api/flights": {
            GET: async () => {
              try {
                return json({ flights: await listFlights() });
              } catch (error) {
                return json({ error: String(error) }, 500);
              }
            },
          },

          "/api/flight/:name": {
            GET: async (req) => {
              const name = req.params.name.replace(/[^a-zA-Z0-9_-]/g, "");
              const file = Bun.file(`${FLIGHTS_DIR}/${name}.json`);

              if (!(await file.exists())) {
                return json({ error: `Không có chuyến bay ${name}` }, 404);
              }

              return json(await file.json());
            },
          },
        },
      });

      console.log(`\n  Trình xem chuyến bay  ->  ${server.url}\n`);
      return;
    } catch (error) {
      if (attempt === 5) throw error;
    }
  }
}

await listen();
