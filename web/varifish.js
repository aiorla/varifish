// varifish in a web page: the engine's WebAssembly build, from `make build
// ARCH=wasm32` or `ARCH=wasm32-relaxed-simd`, in workers that share its
// memory, one for each of its threads, spoken to in UCI.
//
//     import { start, relaxedSimd } from './varifish.js';
//     const engine = await start({
//       wasm: relaxedSimd() ? 'varifish-wasm32-relaxed-simd.wasm' : 'varifish-wasm32.wasm',
//       files: { 'atomic_run3b_e202_l05.nnue': await (await fetch(url)).arrayBuffer() },
//       onLine: line => console.log(line),
//     });
//     engine.send('uci');
//
// Threads share memory only on a cross-origin isolated page: one served
// with the headers `Cross-Origin-Opener-Policy: same-origin` and
// `Cross-Origin-Embedder-Policy: require-corp`.

const PAGE = 65536;
const MB = 1 << 20;

// Each of varifish's threads spawns at most one more at a time: its thread
// pool's, and the transposition table's, which clears the table with as
// many threads.
const workersFor = threads => 2 * threads + 2;

/** Whether this browser has WebAssembly's SIMD, which every build needs. */
export const simd = () =>
  WebAssembly.validate(
    new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]),
  );

/** Whether this browser has relaxed SIMD, which the faster build needs. */
export const relaxedSimd = () =>
  WebAssembly.validate(
    new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11]),
  );

/**
 * Starts the engine.
 *
 * - `wasm`: the build's URL, or its bytes.
 * - `files`: the files the engine may open, by name, as ArrayBuffers: its
 *   networks, and tablebases in the folder "." that SyzygyPath names.
 * - `memoryMB`: the most memory the engine may take, at most 4096, within
 *   which Hash must fit with the rest; the default is the most the browser
 *   gives, of 4096, 2048, 1024 and 512.
 * - `onLine(line, stream)`: each line the engine writes, on "stdout" or
 *   "stderr".
 * - `onExit(code, error)`: when the engine exits, or a thread fails with
 *   `error`; its workers are stopped then.
 *
 * Returns the engine: `send(command)` gives it a UCI command, `exited` is a
 * promise of its exit code, and `terminate()` stops it at once.
 */
export async function start({ wasm, files = {}, memoryMB, onLine = () => {}, onExit = () => {} }) {
  if (!globalThis.crossOriginIsolated)
    throw new Error('varifish needs a cross-origin isolated page for its threads');
  const bytes = typeof wasm === 'string' || wasm instanceof URL
    ? await (await fetch(wasm)).arrayBuffer()
    : wasm;
  const module = await WebAssembly.compile(bytes);
  const memory = sharedMemory(memoryLimits(new Uint8Array(bytes)), memoryMB);

  // The files, copied once into memory that every worker shares
  const shared = {};
  for (const [name, data] of Object.entries(files)) {
    const copy = new SharedArrayBuffer(data.byteLength);
    new Uint8Array(copy).set(new Uint8Array(data));
    shared[name] = copy;
  }

  // Standard input: a ring of bytes, with [written, read, closed, sequence]
  const stdin = new SharedArrayBuffer(16 + MB);
  const control = new Int32Array(stdin, 0, 4);
  const ring = new Uint8Array(stdin, 16);

  // The pool: for each worker [state, thread id, start argument, unused],
  // then the last thread id given. Room for 1024 threads.
  const slots = workersFor(1024);
  const pool = new SharedArrayBuffer(16 * slots + 16);

  const workers = [];
  let done = false;
  let resolveExit;
  const exited = new Promise(resolve => (resolveExit = resolve));
  function finish(code, error) {
    if (done) return;
    done = true;
    for (const worker of workers) worker.terminate();
    for (const [, resolve] of waiting) resolve();
    onExit(code, error);
    resolveExit(code);
  }

  const url = new URL('./varifish-worker.js', import.meta.url);
  function startWorker(slot) {
    const worker = new Worker(url, { type: 'module', name: slot < 0 ? 'varifish' : `varifish ${slot}` });
    workers.push(worker);
    worker.onmessage = ({ data }) => {
      if (data.line !== undefined) onLine(data.line, data.stream);
      else if (data.ready !== undefined) ready();
      else if (data.error !== undefined) finish(1, data.error);
      else if (data.exit !== undefined) finish(data.exit);
    };
    worker.onerror = event => finish(1, event.message);
    worker.postMessage({ module, memory, stdin, pool, slots, slot, files: shared });
    return worker;
  }

  // Pool workers start before the engine needs them, as spawning a thread
  // cannot wait for a worker to load.
  let poolSize = 0;
  let readyCount = 0;
  let waiting = [];
  function ready() {
    readyCount++;
    waiting = waiting.filter(([size, resolve]) => readyCount < size || resolve());
  }
  function grow(size) {
    size = Math.min(size, slots);
    while (poolSize < size) startWorker(poolSize++);
    return readyCount >= size || done
      ? Promise.resolve()
      : new Promise(resolve => waiting.push([size, resolve]));
  }

  // Commands go to the engine in order, each after the workers its threads
  // need are ready.
  let queue = grow(workersFor(1)).then(() => startWorker(-1));
  function send(command) {
    queue = queue.then(async () => {
      const threads = threadsOf(command);
      if (threads) await grow(workersFor(threads));
      if (done) return;
      write(new TextEncoder().encode(command + '\n'));
    });
  }

  function write(data) {
    let written = Atomics.load(control, 0);
    if (written + data.length - Atomics.load(control, 1) > ring.length)
      throw new Error('varifish has not read its input');
    for (const byte of data) ring[written++ % ring.length] = byte;
    Atomics.store(control, 0, written);
    Atomics.add(control, 3, 1);
    Atomics.notify(control, 3);
  }

  await queue;
  return { send, exited, terminate: () => finish(null) };
}

/** The threads a command sets: Threads, or bench's second argument. */
function threadsOf(command) {
  const words = command.trim().split(/\s+/);
  if (words[0] === 'setoption' && words[1] === 'name') {
    const value = words.indexOf('value');
    if (value > 2 && words.slice(2, value).join(' ').toLowerCase() === 'threads')
      return Math.max(1, Number.parseInt(words[value + 1], 10) || 1);
  }
  if (words[0] === 'bench') return Math.max(1, Number.parseInt(words[2], 10) || 1);
  return 0;
}

/** A shared memory within the module's limits, as large as allowed. */
function sharedMemory([initial, maximum], memoryMB) {
  const sizes = memoryMB ? [memoryMB] : [4096, 2048, 1024, 512];
  let error;
  for (const mb of sizes) {
    const pages = Math.min(Math.floor((mb * MB) / PAGE), maximum ?? Infinity);
    if (pages < initial) continue;
    try {
      return new WebAssembly.Memory({ initial, maximum: pages, shared: true });
    } catch (e) {
      error = e;
    }
  }
  throw error ?? new Error('the memory asked for is below what varifish needs');
}

/** The limits, in pages, of the module's imported memory. */
function memoryLimits(b) {
  let i = 8;
  const leb = () => {
    let result = 0, shift = 0, byte;
    do {
      byte = b[i++];
      result += (byte & 0x7f) * 2 ** shift;
      shift += 7;
    } while (byte & 0x80);
    return result;
  };
  while (i < b.length) {
    const id = b[i++];
    const size = leb();
    const end = i + size;
    if (id === 2) {
      for (let n = leb(); n > 0; n--) {
        for (let names = 0; names < 2; names++) {
          const len = leb(); // the module's name, then the import's
          i += len;
        }
        const kind = b[i++];
        if (kind === 2) {
          const flags = b[i++];
          const min = leb();
          return [min, flags & 1 ? leb() : undefined];
        }
        if (kind === 0) leb();
        else throw new Error(`varifish imports something of kind ${kind}`);
      }
    }
    i = end;
  }
  throw new Error('varifish imports no memory');
}
