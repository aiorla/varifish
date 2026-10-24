// One of varifish's threads, in a worker that varifish.js starts: the main
// thread runs the engine's UCI loop on standard input; a pool worker waits
// for the engine to spawn a thread, runs it, and waits again. Every worker
// shares the engine's one memory, and the pool's slots and standard input
// are shared arrays that varifish.js describes.

const SUCCESS = 0, ACCES = 2, AGAIN = 6, BADF = 8, INVAL = 28, NOENT = 44, NOSYS = 52;
const CHARACTER_DEVICE = 2, DIRECTORY = 3, REGULAR_FILE = 4;
const ALL_RIGHTS = 0xffffffffffffffffn;
const PREOPEN = 3;

// Slot states, as varifish.js numbers them
const READY = 1, RUNNING = 2, CLAIMED = 3;

class Exit extends Error {
  constructor(code) {
    super(`exit ${code}`);
    this.code = code;
  }
}

onmessage = ({ data }) => {
  onmessage = null;
  run(data);
};

function run({ module, memory, stdin, pool, slots, slot, files }) {
  const control = new Int32Array(stdin, 0, 4); // written, read, closed, sequence
  const ring = new Uint8Array(stdin, 16);
  const slotsArray = new Int32Array(pool);
  const lastTid = 4 * slots;

  const view = () => new DataView(memory.buffer);
  const bytes = (ptr, len) => new Uint8Array(memory.buffer, ptr, len);
  // Decoders and getRandomValues take no views of shared memory: copies.
  const text = (ptr, len) => new TextDecoder().decode(bytes(ptr, len).slice());

  // Each thread's own descriptors, numbered apart from other threads' so
  // that one passed between threads is refused rather than misread.
  const open = new Map();
  let nextFd = 1000 * (slot + 2);

  const lines = [[], []];
  function output(fd, chunk) {
    const pending = lines[fd - 1];
    let start = 0;
    for (let i = 0; i < chunk.length; i++) {
      if (chunk[i] !== 10) continue;
      pending.push(chunk.subarray(start, i));
      const line = new TextDecoder().decode(concat(pending));
      pending.length = 0;
      postMessage({ line, stream: fd === 1 ? 'stdout' : 'stderr' });
      start = i + 1;
    }
    if (start < chunk.length) pending.push(chunk.slice(start));
  }

  function iovecs(iovs, count) {
    const v = view();
    return Array.from({ length: count }, (_, i) => [
      v.getUint32(iovs + 8 * i, true),
      v.getUint32(iovs + 8 * i + 4, true),
    ]);
  }

  // Copies what standard input holds into the buffers, waiting for a byte
  // or the end.
  function readStdin(buffers) {
    for (;;) {
      const sequence = Atomics.load(control, 3);
      const written = Atomics.load(control, 0);
      let read = Atomics.load(control, 1);
      if (written !== read) {
        let n = 0;
        for (const [ptr, len] of buffers) {
          const target = bytes(ptr, len);
          let k = 0;
          while (k < len && read !== written) target[k++] = ring[read++ % ring.length];
          n += k;
          if (read === written) break;
        }
        Atomics.store(control, 1, read);
        return n;
      }
      if (Atomics.load(control, 2)) return 0;
      Atomics.wait(control, 3, sequence);
    }
  }

  const wasi = {
    args_sizes_get(argc, size) {
      view().setUint32(argc, 1, true);
      view().setUint32(size, 9, true);
      return SUCCESS;
    },
    args_get(argv, buffer) {
      view().setUint32(argv, buffer, true);
      bytes(buffer, 9).set(new TextEncoder().encode('varifish\0'));
      return SUCCESS;
    },
    environ_sizes_get(count, size) {
      view().setUint32(count, 0, true);
      view().setUint32(size, 0, true);
      return SUCCESS;
    },
    environ_get() {
      return SUCCESS;
    },
    random_get(buffer, len) {
      const random = new Uint8Array(len);
      for (let i = 0; i < len; i += 65536) crypto.getRandomValues(random.subarray(i, i + 65536));
      bytes(buffer, len).set(random);
      return SUCCESS;
    },
    // The monotonic clock is the same in every worker: each worker's
    // performance.now() counts from its own start.
    clock_time_get(id, _precision, out) {
      const ms = id === 0 ? Date.now() : performance.timeOrigin + performance.now();
      view().setBigUint64(out, BigInt(Math.round(ms * 1e6)), true);
      return SUCCESS;
    },
    fd_close(fd) {
      return open.delete(fd) ? SUCCESS : BADF;
    },
    fd_fdstat_get(fd, out) {
      const type = fd <= 2 ? CHARACTER_DEVICE : fd === PREOPEN ? DIRECTORY : open.has(fd) ? REGULAR_FILE : 0;
      if (!type) return BADF;
      const v = view();
      v.setUint8(out, type);
      v.setUint16(out + 2, 0, true);
      v.setBigUint64(out + 8, ALL_RIGHTS, true);
      v.setBigUint64(out + 16, ALL_RIGHTS, true);
      return SUCCESS;
    },
    fd_filestat_get(fd, out) {
      const file = open.get(fd);
      if (fd > PREOPEN && !file) return BADF;
      bytes(out, 64).fill(0);
      const v = view();
      v.setUint8(out + 16, fd <= 2 ? CHARACTER_DEVICE : fd === PREOPEN ? DIRECTORY : REGULAR_FILE);
      v.setBigUint64(out + 24, 1n, true);
      v.setBigUint64(out + 32, BigInt(file?.data.length ?? 0), true);
      return SUCCESS;
    },
    fd_prestat_get(fd, out) {
      if (fd !== PREOPEN) return BADF;
      view().setUint8(out, 0);
      view().setUint32(out + 4, 1, true);
      return SUCCESS;
    },
    fd_prestat_dir_name(fd, path, len) {
      if (fd !== PREOPEN) return BADF;
      bytes(path, len).set(new TextEncoder().encode('.').subarray(0, len));
      return SUCCESS;
    },
    fd_read(fd, iovs, count, nread) {
      const buffers = iovecs(iovs, count);
      let total = 0;
      if (fd === 0) total = readStdin(buffers);
      else {
        const file = open.get(fd);
        if (!file) return BADF;
        for (const [ptr, len] of buffers) {
          const chunk = file.data.subarray(file.position, file.position + len);
          bytes(ptr, chunk.length).set(chunk);
          file.position += chunk.length;
          total += chunk.length;
          if (chunk.length < len) break;
        }
      }
      view().setUint32(nread, total, true);
      return SUCCESS;
    },
    fd_write(fd, iovs, count, nwritten) {
      if (fd !== 1 && fd !== 2) return BADF;
      let total = 0;
      for (const [ptr, len] of iovecs(iovs, count)) {
        output(fd, bytes(ptr, len).slice());
        total += len;
      }
      view().setUint32(nwritten, total, true);
      return SUCCESS;
    },
    // Opens only the files varifish.js was given, for reading.
    path_open(fd, _dirflags, path, len, oflags, _rights, _inheriting, _fdflags, out) {
      if (fd !== PREOPEN) return BADF;
      if (oflags & (1 | 4 | 8)) return ACCES; // creating, exclusive, truncating
      const name = text(path, len).replace(/^(\.\/)+/, '');
      if (oflags & 2) return name === '.' || name === '' ? NOSYS : NOENT; // a directory
      const data = files[name];
      if (!data) return NOENT;
      const newFd = nextFd++;
      open.set(newFd, { data: new Uint8Array(data), position: 0 });
      view().setUint32(out, newFd, true);
      return SUCCESS;
    },
    proc_exit(code) {
      throw new Exit(code);
    },
    sched_yield() {
      return SUCCESS;
    },
  };

  // A system call the engine starts using without this file knowing it
  // fails rather than the engine's instantiation.
  const imports = { env: { memory }, wasi_snapshot_preview1: {}, wasi: { 'thread-spawn': spawn } };
  for (const { module: from, name, kind } of WebAssembly.Module.imports(module)) {
    if (from === 'wasi_snapshot_preview1' && kind === 'function')
      imports[from][name] = wasi[name] ?? (() => NOSYS);
  }

  // thread-spawn: hands the new thread to a ready pool worker.
  function spawn(startArg) {
    for (let i = 0; i < slots; i++) {
      if (Atomics.compareExchange(slotsArray, 4 * i, READY, CLAIMED) !== READY) continue;
      const tid = Atomics.add(slotsArray, lastTid, 1) + 1;
      Atomics.store(slotsArray, 4 * i + 1, tid);
      Atomics.store(slotsArray, 4 * i + 2, startArg);
      Atomics.store(slotsArray, 4 * i, RUNNING);
      Atomics.notify(slotsArray, 4 * i);
      return tid;
    }
    return -AGAIN;
  }

  const instance = new WebAssembly.Instance(module, imports);

  // Runs a thread's function: null when it returns, else the exit code.
  function call(f) {
    try {
      f();
      return null;
    } catch (e) {
      if (e instanceof Exit) return e.code;
      postMessage({ error: String(e?.stack ?? e) });
      return 1;
    }
  }

  if (slot < 0) {
    postMessage({ exit: call(() => instance.exports._start()) ?? 0 });
    return;
  }
  const state = 4 * slot;
  Atomics.store(slotsArray, state, READY);
  postMessage({ ready: slot });
  for (;;) {
    for (let s; (s = Atomics.load(slotsArray, state)) !== RUNNING; ) Atomics.wait(slotsArray, state, s);
    const code = call(() =>
      instance.exports.wasi_thread_start(
        Atomics.load(slotsArray, state + 1),
        Atomics.load(slotsArray, state + 2),
      ),
    );
    if (code !== null) {
      postMessage({ exit: code });
      return;
    }
    Atomics.store(slotsArray, state, READY);
  }
}

function concat(chunks) {
  const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}
