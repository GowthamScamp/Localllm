//! Bounded ring buffer for child-process stdout/stderr lines.
//!
//! Each `SglangProcess` and `LlamaCppServer` owns an `Arc<LogRingBuffer>` that
//! its reader tasks push lines into. The `GET /api/logs/{alias}` endpoint and
//! `localllm logs <alias>` CLI command read a snapshot back.
//!
//! Capacity is fixed at construction; pushing past capacity drops the oldest
//! line. Internally a `Mutex<VecDeque<String>>` — there's no high-frequency
//! contention since only the reader task writes and only HTTP handlers read.

use std::collections::VecDeque;
use std::sync::Mutex;

pub const DEFAULT_CAPACITY: usize = 500;

#[derive(Debug)]
pub struct LogRingBuffer {
    inner: Mutex<VecDeque<String>>,
    capacity: usize,
}

impl LogRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Append a line, evicting the oldest if at capacity. Locks briefly;
    /// callers are background tokio tasks so no async blocking.
    pub fn push(&self, line: String) {
        let mut buf = match self.inner.lock() {
            Ok(g) => g,
            // On poisoning we recover and continue — losing a few lines is fine.
            Err(p) => p.into_inner(),
        };
        if buf.len() == self.capacity {
            buf.pop_front();
        }
        buf.push_back(line);
    }

    /// Snapshot last `n` lines (or all, if `n` >= current length).
    pub fn snapshot(&self, n: usize) -> Vec<String> {
        let buf = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let len = buf.len();
        let start = len.saturating_sub(n);
        buf.iter().skip(start).cloned().collect()
    }
}

impl Default for LogRingBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_oldest_at_capacity() {
        let r = LogRingBuffer::new(3);
        r.push("a".into());
        r.push("b".into());
        r.push("c".into());
        r.push("d".into());
        let snap = r.snapshot(10);
        assert_eq!(snap, vec!["b", "c", "d"]);
    }

    #[test]
    fn snapshot_n_returns_tail() {
        let r = LogRingBuffer::new(10);
        for i in 0..5 {
            r.push(format!("line{}", i));
        }
        assert_eq!(r.snapshot(2), vec!["line3", "line4"]);
    }
}
