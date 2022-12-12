use std::sync::Arc;

use wasm_bindgen::prelude::*;
use rustfft::{FftPlanner, Fft, num_complex::Complex};

mod utils;

extern crate web_sys;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
macro_rules! log {
  ( $( $t:tt )* ) => {
    web_sys::console::log_1(&format!( $( $t )* ).into());
  }
}

fn decibel_to_gain(decibel: f32) -> f32
{
  (10.0 as f32).powf(decibel / 20.0)
}

fn gain_to_decibel(gain: f32, min_decibel: f32) -> f32
{
  if gain == 0.0 {
    return min_decibel;
  }

  let tmp = 20.0 * gain.abs().log10();
  tmp.max(min_decibel)
}

const LEVEL_REDUCTION_PER_SAMPLE: f32 = 1.0 / 24000.0;
const MIN_DECIBEL: f32 = -48.0;
static mut MIN_DECIBEL_GAIN: f32 = 0.0;

struct RingBuffer<T>
{
  buffer: Vec<T>;
  capacity: usize;
  read_pos: usize;
  write_pos: usize;
};

impl RingBuffer<T> {
  pub fn new(capacity: usize) {
    RingBuffer {
      buffer: Vec::new::<T>(0; capacity + 1),
      capacity: capacity,
      read_pos: 0,
      write_pos: 0,
    }
  }

  pub fn num_readable(&self) {
    if self.read_pos <=self.write_pos {
      self.write_pos - self.read_pos
    } else {
      self.write_pos + (self.capacity + 1) - self.read_pos
    }
  }

  pub fn num_writable(&self) {
    self.capacity - self.num_readable()
  }

  pub fn read(&mut self, dest: &mut [f32], length: usize) -> bool {
    if self.num_readable() < length {
      return false;
    }

    let buffer_length = self.capacity + 1;
    let copy1 = std::cmp::min(buffer_length - self.read_pos, length);


    dest[0..copy1].clone_from_slice(&self.buffer[self.read_pos..self.read_pos + copy1]);

    if copy1 == length {
      return true;
    }

    let copy2 = std::cmp::max(length, copy1) - copy1;
    dest[copy1..copy1+copy2].clone_from_slice(&self.buffer[0..copy2]);

    true
  }

  pub fn write(&mut self, src: &[f32], length: usize) -> bool {
    if self.num_readable() < length {
      return false;
    }

    let buffer_length = self.capacity + 1;
    let copy1 = std::cmp::min(buffer_length - self.write_pos, length);

    if copy1 == length {
      self.buffer[self.write_pos..self.write_pos + copy1].clone_from_slice(&src[0..copy1]);
      self.write_pos += copy1;
      return true;
    }

    let copy2 = length - copy1;
    self.buffer[0..copy2].clone_from_slice(&src[copy1..copy1 + copy2]);
    self.write_pos = copy2;
    true
  }

  pub fn overlap_add(&mut self, src: &[f32], length: usize, overlap_size: usize) -> boolean {
    if length < overlap_size {
      return false;
    }

    if self.num_readable() < overlap_size {
      return false;
    }

    let num_to_write_new = length - overlap_size;
    if self.num_writable() < num_to_write_new {
      return false;
    }

    let buffer_length = self.capacity + 1;

    let write_start =
    if self.write_pos >= overlap_size {
      self.write_pos - overlap_size
    } else {
      buffer_length - (overlap_size - self.write_pos)
    };

    let copy1 = std::cmp::min(buffer_length - write_start, overlap_size);

    self.buffer[write_start..write_start + copy1].clone_from_slice(&src[0..copy1]);

    if copy1 != overlap_size {
      let copy2 = overlap_size - copy1;
      self.buffer[0..copy2].clone_from_slice(&src[copy1..copy1 + copy2]);
    }

    self.write(&src[overlap_size..length], num_to_write_new)
  }

  pub fn fill(&mut self, length: usize, value: T) -> bool {
    if self.num_writable() < length {
      return false;
    }

    let buffer_length = self.capacity + 1;
    let copy1 = std::cmp::min(buffer_length - self.write_pos, length);

    self.buffer[self.write_pos..self.write_pos + copy1].fill(value);

    if copy1 == length {
      self.write_pos += copy1;
      return true;
    }

    let copy2 = buffer_length - copy1;
    self.buffer[0..copy2].fill(value);
    self.write_pos = copy2;

    true
  }

  pub fn discard(&mut self, length: usize) -> bool {
    if self.num_readable() < length {
      return false;
    }

    let buffer_length = self.capacity + 1;
    let discard1 = std::cmp::min(buffer_length - self.read_pos, length);

    if discard1 == length {
      self.read_pos += length;
      return true;
    }

    let discard2 = length - discard1;
    self.read_pos = discard2;

    true
  }

  pub fn discard_all(&mut self) {
    self.discard(self.num_readable());
  }
}

type Cmp = Complex<f32>;

#[wasm_bindgen]
pub struct WasmProcessor {
  sample_rate: usize,
  block_size: usize,
  fft: Arc<dyn Fft<f32>>,
  ifft: Arc<dyn Fft<f32>>,
  fft_size: usize,
  overlap_size: usize,
  input_ring_buffer: RingBuffer<f32>,
  output_ring_buffer: RingBuffer<f32>,
  signal_buffer: Vec<Cmp>,
  frequency_buffer: Vec<Cmp>,
  cepstrum_buffer: Vec<Cmp>,
  tmp_fft_buffer: Vec<Cmp>,
  tmp_fft_buffer2: Vec<Cmp>,
  tmp_phase_buffer: Vec<f32>,
  window: Vec<f32>,
  prev_input_phases: Vec<f32>,
  prev_output_phases: Vec<f32>,
  analysis_magnitude: Vec<f32>,
  analysis_frequencies: Vec<f32>,
  synthesis_magnitude: Vec<f32>,
  synthesis_frequencies: Vec<f32>,
  tmp_buffer: Vec<f32>,
  wet_buffer: Vec<f32>,
  dry_wet: f32,
  formant: f32,
  pitch: f32,
  output_gain: f32,
  envelope_order: usize,
  input_level: f32,
  output_level: f32,
};

#[wasm_bindgen]
impl WasmProcessor {
  pub fn new(sample_rate: usize, block_size: usize) -> WasmProcessor {
    utils::set_panic_hook();

    unsafe { MIN_DECIBEL_GAIN = decibel_to_gain(MIN_DECIBEL); };

    let fft_size = 4096;
    let overlap_size = 2048;

    let fft_planner = FftPlanner::new();
    let fft = fft_planner.plan_fft_forward(fft_size);
    let ifft = fft_planner.plan_fft_backward(fft_size);

    let signal_buffer = vec![Cmp::new(0, 0), fft_size];
    let frequency_buffer = vec![Cmp::new(0, 0), fft_size];
    let cepstrum_buffer = vec![Cmp::new(0, 0), fft_size];

    let window = vec![0.0f32, fft_size];
    for (i, elem) in window.iter_mut().enumerate() {
      let omega: f32 = 2.0f32 * std::f32::consts::PI * i / fft_size;
      elem = 0.5f32 * (1.0f32 - omega.cos());
    }

    let input_ring_buffer = RingBuffer::<f32>::new(fft_size);
    input_ring_buffer.discard_all();
    input_ring_buffer.fill(fft_size - overlap_size);

    let output_ring_buffer = RingBuffer::<f32>::new(fft_size + block_size);
    output_ring_buffer.discard_all();
    output_ring_buffer.fill(fft_size + block_size - overlap_size);

    let tmp_buffer = vec![0.0f32, fft_size];
    let wet_buffer = vec![0.0f32, block_size];

    let tmp_fft_buffer = vec![Cmp::new(0, 0), fft_size];
    let tmp_fft_buffer2 = vec![Cmp::new(0, 0), fft_size];
    let tmp_phase_buffer = vec![Cmp::new(0, 0), fft_size];
    let prev_input_phases = vec![Cmp::new(0, 0), fft_size];
    let prev_output_phases = vec![Cmp::new(0, 0), fft_size];
    let analysis_magnitude = vec![Cmp::new(0, 0), fft_size];
    let analysis_frequencies = vec![Cmp::new(0, 0), fft_size];
    let synthesis_magnitude = vec![Cmp::new(0, 0), fft_size];
    let synthesis_frequencies = vec![Cmp::new(0, 0), fft_size];

    let d = WasmProcessor {
      sample_rate,
      block_size,
      fft,
      ifft,
      fft_size,
      overlap_size,
      input_ring_buffer,
      output_ring_buffer,
      signal_buffer,
      frequency_buffer,
      cepstrum_buffer,
      tmp_fft_buffer,
      tmp_fft_buffer2,
      tmp_phase_buffer,
      window,
      prev_input_phases,
      prev_output_phases,
      analysis_magnitude,
      analysis_frequencies,
      synthesis_magnitude,
      synthesis_frequencies,
      tmp_buffer,
      wet_buffer,
      dry_wet: 0.5,
      formant: 0.5,
      pitch: 0.5,
      output_gain: 0.0,
      input_level: MIN_DECIBEL,
      output_level: MIN_DECIBEL,
    };

//     log!("{}", d.delay.dump());
    d
  }

  pub fn process(&mut self, buffer: &mut [f32], length: usize, levels: &mut [f32]) {
//    log!("len of levels: {}", levels.len());

    let omega = 440.0 * 2.0 * std::f32::consts::PI / self.sample_rate as f32;

//    let before = buffer[0];
    let new_input_level_gain = buffer.iter().max_by(|x, y| x.abs().total_cmp(&y.abs())).expect("buffer should not be empty");
    let reduced_level = MIN_DECIBEL.max(self.input_level - LEVEL_REDUCTION_PER_SAMPLE * buffer.len() as f32);
    levels[0] = gain_to_decibel(*new_input_level_gain, MIN_DECIBEL).max(reduced_level);

    for x in buffer[..length].iter_mut() {
      *x = self.delay.process(*x);
      self.phase += omega;
      if self.phase >= 2.0 * std::f32::consts::PI {
        self.phase -= 2.0 * std::f32::consts::PI;
      }
    }

    let new_output_level_gain = buffer.iter().max_by(|x, y| x.abs().total_cmp(&y.abs())).expect("buffer should not be empty");
    let reduced_level = MIN_DECIBEL.max(self.output_level - LEVEL_REDUCTION_PER_SAMPLE * buffer.len() as f32);
    levels[1] = gain_to_decibel(*new_output_level_gain, MIN_DECIBEL).max(reduced_level);
//    let after = buffer[0];

//     log!("before {}, after {}", before, after);
  }

  pub fn set_wet_amount(&mut self, value: f32) {
    self.delay.wet_amount = value;
  }

  pub fn set_feedback_amount(&mut self, value: f32) {
    log!("wet-amount: {}, feedback: {}, length: {}", self.delay.wet_amount, self.delay.feedback_amount, self.delay.length);

    self.delay.feedback_amount = std::cmp::min(value, 0.95);
  }

  pub fn set_delay_length(&mut self, value: f32) {
    let mut len = unsafe { (value * self.sample_rate as f32).to_int_unchecked::<usize>() };
    len = std::cmp::max(len, 10);
    self.delay = DelayLine::new(len, self.delay.wet_amount, self.delay.feedback_amount);
  }
}
