//! The search threads: each runs the jobs given to it on its worker, and the
//! pool starts a search on all of them, the main thread leading.
//!
//! A translation into Rust, modified for varifish in 2026, of Stockfish's
//! `thread.h` and `thread.cpp`: Stockfish, a UCI chess playing engine derived
//! from Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! AUTHORS), under the GNU General Public License, version 3 or later.
//!
//! varifish does not bind threads to NUMA nodes nor replicate the network
//! and the shared histories across them: all threads share one copy.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::history::SharedHistories;
use crate::movegen::{ALL, MoveList};
use crate::position::Position;
use crate::search::{
    Counters, LimitsType, RootMove, SearchManager, SearchOptions, SharedState, UpdateContext,
    Worker,
};
use crate::types::*;
use crate::uci::to_move;
use crate::variant::Variant;

/// The stack of a search thread, which recursion reaches MAX_PLY deep. In
/// WebAssembly it comes out of the engine's memory, which is never given
/// back, and MAX_PLY plies take about 1.1 MB of it.
const THREAD_STACK_SIZE: usize = if cfg!(target_arch = "wasm32") {
    4 * 1024 * 1024
} else {
    64 * 1024 * 1024
};

/// Locks a mutex, which a panic while it was held does not poison here: a
/// panic aborts in release builds, and fails the test in others.
pub fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A job that a thread runs on its worker
pub type Job<V> = Box<dyn FnOnce(&mut Worker<V>) + Send>;

struct JobState<V: Variant> {
    job: Option<Job<V>>,
    searching: bool,
    exit: bool,
}

/// What other threads share of a thread: its job, and its worker, which
/// the thread holds while it runs a job.
pub struct ThreadHandle<V: Variant> {
    state: Mutex<JobState<V>>,
    cv: Condvar,
    pub worker: Mutex<Box<Worker<V>>>,
}

impl<V: Variant> ThreadHandle<V> {
    /// Launching a function in the thread
    pub fn run_job(&self, job: Job<V>) {
        let mut state = lock(&self.state);
        while state.searching {
            state = self.cv.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.job = Some(job);
        state.searching = true;
        drop(state);
        self.cv.notify_all();
    }

    /// Blocks on the condition variable until the thread has finished
    /// searching
    pub fn wait_for_search_finished(&self) {
        let mut state = lock(&self.state);
        while state.searching {
            state = self.cv.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Thread gets parked here, blocked on the condition variable when the
    /// thread has no work to do.
    fn idle_loop(&self) {
        loop {
            let job = {
                let mut state = lock(&self.state);
                state.searching = false;
                self.cv.notify_all(); // Wake up anyone waiting for search finished
                while !state.searching {
                    state = self.cv.wait(state).unwrap_or_else(|e| e.into_inner());
                }

                if state.exit {
                    return;
                }

                state.job.take()
            };

            if let Some(job) = job {
                job(&mut lock(&self.worker));
            }
        }
    }
}

/// A search thread, which runs jobs until it is dropped.
pub struct Thread<V: Variant> {
    pub handle: Arc<ThreadHandle<V>>,
    join: Option<JoinHandle<()>>,
}

impl<V: Variant> Thread<V> {
    /// Launches the thread and waits until it goes to sleep in idle_loop().
    fn new(worker: Box<Worker<V>>) -> Thread<V> {
        let handle = Arc::new(ThreadHandle {
            state: Mutex::new(JobState {
                job: None,
                searching: true,
                exit: false,
            }),
            cv: Condvar::new(),
            worker: Mutex::new(worker),
        });

        let thread_handle = Arc::clone(&handle);
        let join = std::thread::Builder::new()
            .stack_size(THREAD_STACK_SIZE)
            .spawn(move || thread_handle.idle_loop())
            .unwrap_or_else(|_| {
                eprintln!("Failed to create search thread");
                std::process::exit(1)
            });

        handle.wait_for_search_finished();

        Thread {
            handle,
            join: Some(join),
        }
    }
}

impl<V: Variant> Drop for Thread<V> {
    /// Wakes up the thread in idle_loop() and waits for its termination.
    fn drop(&mut self) {
        self.handle.wait_for_search_finished();
        {
            let mut state = lock(&self.handle.state);
            state.exit = true;
            state.searching = true;
        }
        self.handle.cv.notify_all();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// ThreadPool struct handles all the threads-related stuff like init,
/// starting, parking and, most importantly, launching a thread. All the
/// access to threads is done through this class.
pub struct ThreadPool<V: Variant> {
    // The main thread first
    threads: Vec<Thread<V>>,
}

impl<V: Variant> Default for ThreadPool<V> {
    fn default() -> Self {
        ThreadPool {
            threads: Vec::new(),
        }
    }
}

impl<V: Variant> ThreadPool<V> {
    pub fn len(&self) -> usize {
        self.threads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    pub fn main_thread(&self) -> &ThreadHandle<V> {
        &self.threads[0].handle
    }

    /// Creates/destroys threads to match the requested number. Created and
    /// launched threads will immediately go to sleep in idle_loop. Upon
    /// resizing, threads are recreated.
    pub fn set(&mut self, sh: &mut Arc<SharedState>, requested: usize) {
        if !self.threads.is_empty() {
            // destroy any existing thread(s)
            self.main_thread().wait_for_search_finished();
            self.threads.clear();
        }

        if requested == 0 {
            return;
        }

        // All threads are part of one NUMA node.
        {
            let shared = Arc::get_mut(sh).expect("no search is running");
            shared.histories = SharedHistories::new(requested.next_power_of_two());
        }

        let counters: Vec<Arc<Counters>> = (0..requested)
            .map(|_| Arc::new(Counters::default()))
            .collect();

        // The main thread's manager starts and waits for the others, which
        // are created first.
        let helpers: Vec<Thread<V>> = (1..requested)
            .map(|idx| {
                Thread::new(Worker::new(
                    idx,
                    requested,
                    Arc::clone(&counters[idx]),
                    None,
                    &sh.networks,
                ))
            })
            .collect();
        let handles = helpers.iter().map(|th| Arc::clone(&th.handle)).collect();
        let manager = Box::new(SearchManager::new(handles, counters.clone()));
        let main = Thread::new(Worker::new(
            0,
            requested,
            Arc::clone(&counters[0]),
            Some(manager),
            &sh.networks,
        ));

        self.threads.push(main);
        self.threads.extend(helpers);

        self.clear(sh);

        self.main_thread().wait_for_search_finished();
    }

    /// Sets threadPool data to initial values
    pub fn clear(&self, sh: &Arc<SharedState>) {
        if self.threads.is_empty() {
            return;
        }

        // The caller's reference is the only one: no thread searches.
        assert!(Arc::strong_count(sh) == 1, "no search is running");
        for th in &self.threads {
            let sh = Arc::clone(sh);
            // SAFETY: every thread clears in its job, and none searches
            // before all are done, as the pool waits for them below.
            th.handle
                .run_job(Box::new(move |worker| unsafe { worker.clear(&sh) }));
        }

        for th in &self.threads {
            th.handle.wait_for_search_finished();
        }

        let mut main = lock(&self.main_thread().worker);
        let manager = main.manager.as_deref_mut().expect("the main thread");

        // These two affect the time taken on the first move of a game:
        manager.best_previous_average_score = VALUE_INFINITE;
        manager.previous_time_reduction = 0.85;

        manager.calls_cnt = 0;
        manager.best_previous_score = VALUE_INFINITE;
        manager.original_time_adjust = -1.0;
        manager.tm.clear();
    }

    /// Wakes up main thread waiting in idle_loop() and returns immediately.
    /// Main thread will wake up other threads and start the search.
    pub fn start_thinking(
        &self,
        sh: &Arc<SharedState>,
        pos: &Position<V>,
        limits: LimitsType,
        options: SearchOptions,
        updates: &UpdateContext,
    ) {
        self.main_thread().wait_for_search_finished();

        {
            let mut main = lock(&self.main_thread().worker);
            let manager = main.manager.as_deref_mut().expect("the main thread");
            manager.stop_on_ponderhit = false;
            manager.updates = updates.clone();
        }
        sh.stop.store(false, Relaxed);
        sh.ponder.store(limits.ponder_mode, Relaxed);
        sh.increase_depth.store(true, Relaxed);

        let mut root_moves = Vec::new();

        for uci_move in &limits.searchmoves {
            let m = to_move(pos, uci_move);

            if m != Move::none() {
                root_moves.push(RootMove::new(m));
            }
        }

        if root_moves.is_empty() {
            for &m in MoveList::new::<ALL>(pos).iter() {
                root_moves.push(RootMove::new(m));
            }
        }

        // Atomic ranks the winning moves by their distance to zeroing, as
        // Atomic-Stockfish does, and chess ranks them equally, as Stockfish
        // does.
        let tb_config = V::tables(&sh.tablebases).rank_root_moves(
            &options.syzygy,
            &mut pos.clone(),
            &mut root_moves,
            V::ATOMIC,
            &|| false,
        );

        // Each thread searches its own copy of the root position.
        for th in &self.threads {
            let limits = limits.clone();
            let root_moves = root_moves.clone();
            let root_pos = pos.clone();
            th.handle.run_job(Box::new(move |worker| {
                worker.limits = limits;
                worker.counters.reset();
                worker.nmp_min_ply = 0;
                worker.root_depth = 0;
                worker.root_moves = root_moves;
                worker.root_pos = root_pos;
                worker.tb_config = tb_config;
                worker.options = options;
            }));
        }

        for th in &self.threads {
            th.handle.wait_for_search_finished();
        }

        let sh = Arc::clone(sh);
        self.main_thread()
            .run_job(Box::new(move |worker| worker.start_searching(&sh)));
    }

    /// Waits for the main thread, which waits for the others.
    pub fn wait_for_search_finished(&self) {
        if !self.threads.is_empty() {
            self.main_thread().wait_for_search_finished();
        }
    }
}

impl<V: Variant> Drop for ThreadPool<V> {
    fn drop(&mut self) {
        // destroy any existing thread(s)
        self.wait_for_search_finished();
        self.threads.clear();
    }
}
