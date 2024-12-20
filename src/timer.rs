use std::time::Instant;

#[derive(Copy, Clone, Debug)]
pub struct GameTimer {
    base_timer: Instant,
    frame_timer: Instant,

    stopped: bool,

    base_time: f64,
    paused_time: f64,
    stop_time: f64,

    prev_tick: f64,
    curr_tick: f64,
}

impl Default for GameTimer {
    fn default() -> Self {
        Self {
            base_timer: Instant::now(),
            frame_timer: Instant::now(),
            stopped: Default::default(),
            base_time: Default::default(),
            paused_time: Default::default(),
            stop_time: Default::default(),
            prev_tick: Default::default(),
            curr_tick: Default::default(),
        }
    }
}

impl GameTimer {
    const MILLIS_PER_SECS: f64 = 1000.0;
    pub fn total_time(&self) -> f32 {
        (self.base_timer.elapsed().as_secs_f64() * Self::MILLIS_PER_SECS
            - self.paused_time
            - self.base_time) as f32
            / 1000.0
    }

    pub fn delta_time(&self) -> f32 {
        (self.curr_tick - self.prev_tick) as f32
    }

    pub fn reset(&mut self) {
        self.stopped = false;
        self.stop_time = 0.0;
        self.base_time = self.base_timer.elapsed().as_secs_f64() * Self::MILLIS_PER_SECS;
        self.frame_timer = Instant::now();
    }

    pub fn start(&mut self) {
        if !self.stopped {
            return;
        }

        let start_time = self.base_timer.elapsed().as_secs_f64() * Self::MILLIS_PER_SECS;
        self.paused_time += start_time - self.stop_time;

        self.frame_timer = Instant::now();
        self.stop_time = 0.0;
        self.stopped = false;
    }

    pub fn stop(&mut self) {
        if self.stopped {
            return;
        }

        self.stop_time = self.base_timer.elapsed().as_secs_f64() * Self::MILLIS_PER_SECS;
        self.stopped = true;
    }

    pub fn tick(&mut self) {
        if self.stopped {
            self.prev_tick = 0.0;
            self.curr_tick = 0.0;
            return;
        }

        self.prev_tick = self.curr_tick;
        self.curr_tick = self.frame_timer.elapsed().as_secs_f64();
    }
}
