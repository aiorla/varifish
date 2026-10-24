// varifish in a worker of its own, for a page that speaks to the engine only
// by messages: started as a module worker from the folder that web/build.sh
// fills, it loads the build and the files that the build's manifest names,
// starts the engine with varifish.js, and passes UCI between them.
//
// The page sends {load: {base, manifest, memoryMB}} once, where `base` is the
// manifest's URL, then {command} for each UCI command. The worker answers
// {loading: what, share, bytes} as it loads "engine", "network" and
// "tablebases", {ready: {build, ms}} once the engine reads commands, then
// {line, stream} for each line it writes, and {exit, error} when it stops.
// The page must be cross-origin isolated.

import { relaxedSimd, simd, start } from './varifish.js';

let engine = null;
// Commands sent before the engine started, in order
const queued = [];

onmessage = ({ data }) => {
  if (data.load !== undefined) {
    load(data.load).catch(error => postMessage({ exit: 1, error: String(error?.message ?? error) }));
  } else if (data.command !== undefined) {
    if (engine === null) queued.push(data.command);
    else engine.send(data.command);
  }
};

/** Loads the build this browser runs and the files it opens, and starts it. */
async function load({ base, manifest, memoryMB }) {
  const started = performance.now();
  if (!simd()) throw new Error('this browser has no WebAssembly SIMD, which varifish needs');
  postMessage({ loading: 'engine' });
  const build = manifest.relaxed !== undefined && relaxedSimd() ? manifest.relaxed : manifest.wasm;
  const wasm = await fetched(new URL(build, base));

  const files = {};
  const { network } = manifest;
  postMessage({ loading: 'network', share: 0, bytes: network.bytes });
  files[network.name] = await fetchedWithProgress(new URL(network.url, base), 'network', network.bytes);
  postMessage({ loading: 'tablebases' });
  const tables = await Promise.all(manifest.tables.map(table => fetched(new URL(table.url, base))));
  manifest.tables.forEach((table, i) => (files[table.name] = tables[i]));

  engine = await start({
    wasm,
    files,
    memoryMB,
    onLine: (line, stream) => postMessage({ line, stream }),
    onExit: (exit, error) => postMessage({ exit, error }),
  });
  for (const command of queued.splice(0)) engine.send(command);
  postMessage({ ready: { build, ms: Math.round(performance.now() - started) } });
}

async function fetched(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.arrayBuffer();
}

/**
 * Fetches a large file, reporting its share received as {loading: what,
 * share, bytes} every tenth of a second: `bytes` is what crosses the network,
 * compressed if the server compresses it, and the share counts the file as
 * received, against its size `total`.
 */
async function fetchedWithProgress(url, what, total) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = Number(response.headers.get('Content-Length')) || total;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  let reported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (performance.now() - reported > 100) {
      reported = performance.now();
      postMessage({ loading: what, share: total > 0 ? Math.min(1, received / total) : null, bytes });
    }
  }
  const whole = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    whole.set(chunk, offset);
    offset += chunk.length;
  }
  return whole.buffer;
}
