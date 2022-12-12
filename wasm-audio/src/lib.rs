use std::sync::Arc;

use wasm_bindgen::prelude::*;
use rustfft::{FftPlanner, Fft, num_complex::Complex};
use std::convert::{TryFrom, TryInto};

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
  (10.0f32).powf(decibel / 20.0)
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

struct RingBuffer<T: Clone>
{
  buffer: Vec<T>,
  capacity: usize,
  read_pos: usize,
  write_pos: usize,
}

impl<T: Clone> RingBuffer<T> {
  pub fn new(capacity: usize, value: T) -> RingBuffer<T> {
    RingBuffer {
      buffer: vec![value; capacity + 1],
      capacity,
      read_pos: 0,
      write_pos: 0,
    }
  }

  pub fn capacity(&self) -> usize {
    self.capacity
  }

  pub fn num_readable(&self) -> usize {
    if self.read_pos <=self.write_pos {
      self.write_pos - self.read_pos
    } else {
      self.write_pos + (self.capacity + 1) - self.read_pos
    }
  }

  pub fn num_writable(&self) -> usize {
    self.capacity - self.num_readable()
  }

  pub fn is_full(&self) -> bool {
    self.num_writable() == 0
  }

  pub fn is_empty(&self) -> bool {
    self.num_readable() == 0
  }

  pub fn read(&mut self, dest: &mut [T]) -> bool {
    let length = dest.len();

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

  pub fn write(&mut self, src: &[T]) -> bool {
    let length = src.len();

    if self.num_writable() < length {
      return false;
    }

    let buffer_length = self.capacity + 1;
    let copy1 = std::cmp::min(buffer_length - self.write_pos, length);

    self.buffer[self.write_pos..self.write_pos + copy1].clone_from_slice(&src[0..copy1]);
    
    if copy1 == length {
      self.write_pos += copy1;
      return true;
    }

    let copy2 = length - copy1;
    self.buffer[0..copy2].clone_from_slice(&src[copy1..copy1 + copy2]);
    self.write_pos = copy2;
    true
  }

  pub fn overlap_add(&mut self, src: &[T], overlap_size: usize) -> bool {
    let length = src.len();

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

    self.write(&src[overlap_size..length])
  }

  pub fn fill(&mut self, length: usize, value: T) -> bool {
    if self.num_writable() < length {
      return false;
    }

    let buffer_length = self.capacity + 1;
    let copy1 = std::cmp::min(buffer_length - self.write_pos, length);

    self.buffer[self.write_pos..self.write_pos + copy1].fill(value.clone());

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
  output_gain_decibel: f32,
  envelope_order: usize,
  input_level: f32,
  output_level: f32,
}

#[wasm_bindgen]
impl WasmProcessor {
  pub fn new(sample_rate: usize, block_size: usize) -> WasmProcessor {
    utils::set_panic_hook();

    unsafe { MIN_DECIBEL_GAIN = decibel_to_gain(MIN_DECIBEL); };

    let fft_size = 2048;
    let overlap_size = 512;

    let mut fft_planner = FftPlanner::new();
    let fft = fft_planner.plan_fft_forward(fft_size);
    let ifft = fft_planner.plan_fft_inverse(fft_size);

    let orig = Cmp::new(0.0, 0.0);

    let signal_buffer = vec![orig; fft_size];
    let frequency_buffer = vec![orig; fft_size];
    let cepstrum_buffer = vec![orig; fft_size];

    let mut window = vec![0.0f32; fft_size];
    for (i, elem) in window.iter_mut().enumerate() {
      let omega: f32 = 2.0f32 * std::f32::consts::PI * i as f32 / fft_size as f32;
      *elem = 0.5f32 * (1.0f32 - omega.cos());
    }

    let mut input_ring_buffer = RingBuffer::<f32>::new(fft_size, 0.0f32);
    input_ring_buffer.discard_all();
    input_ring_buffer.fill(fft_size - overlap_size, 0.0f32);
    log!("input ring buffer: readable size {}", input_ring_buffer.num_readable());

    let mut output_ring_buffer = RingBuffer::<f32>::new(fft_size + 2 * block_size, 0.0f32);
    output_ring_buffer.discard_all();
    output_ring_buffer.fill(fft_size + block_size - overlap_size, 0.0f32);
    log!("output ring buffer: capacity {}", output_ring_buffer.capacity());
    log!("output ring buffer: readable size {}", output_ring_buffer.num_readable());

    let tmp_buffer = vec![0.0f32; fft_size];
    let wet_buffer = vec![0.0f32; block_size];

    let tmp_fft_buffer = vec![orig; fft_size];
    let tmp_fft_buffer2 = vec![orig; fft_size];
    let tmp_phase_buffer = vec![0.0f32; fft_size];
    let prev_input_phases = vec![0.0f32; fft_size];
    let prev_output_phases = vec![0.0f32; fft_size];
    let analysis_magnitude = vec![0.0f32; fft_size];
    let analysis_frequencies = vec![0.0f32; fft_size];
    let synthesis_magnitude = vec![0.0f32; fft_size];
    let synthesis_frequencies = vec![0.0f32; fft_size];

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
      dry_wet: 0.8,
      formant: 0.5,
      pitch: 0.5,
      output_gain_decibel: 0.0,
      envelope_order: 5,
      input_level: MIN_DECIBEL,
      output_level: MIN_DECIBEL,
    };

    d
  }

  pub fn process(&mut self, buffer: &mut [f32], length: usize, levels: &mut [f32]) {
    // log!("process {}", length);
    let dry_level = self.dry_wet;
    let wet_level = 1.0 - dry_level;

    // log!("input sum: {}", buffer.iter().sum::<f32>());

    levels.fill(0.0f32);

    let mut processed = 0;

    loop {
      if processed == length { 
        break;
      }

      let num_writable = self.input_ring_buffer.num_writable();
      if num_writable == 0 {
        // log!("[ERROR] unexpected writable size");
      }

      let num_to_write = std::cmp::min(num_writable, length - processed);
      // log!("num_to_write: {}", &num_to_write);

      let write_result = self.input_ring_buffer.write(&buffer[processed..processed + num_to_write]); 
      if !write_result {
        // log!("[ERROR] failed to write into input ring buffer");
      }

      // log!("input ring buffer before {}", &self.input_ring_buffer.num_readable());
      if self.input_ring_buffer.is_full() {
        self.process_fft_block();
      }
      // log!("input ring buffer after {}", &self.input_ring_buffer.num_readable());

      // log!("read output ring buffer {}", &num_to_write);
      let read_result = self.output_ring_buffer.read(&mut self.wet_buffer[processed..processed + num_to_write]);

      if !read_result {
        // log!("[ERROR] failed to read from output ring buffer {}", self.output_ring_buffer.num_readable());
      }

      let discard_result = self.output_ring_buffer.discard(num_to_write);
      if !discard_result {
        // log!("[ERROR] failed to discard output ring buffer {}", self.output_ring_buffer.num_readable());
      }

      processed += num_to_write;

      // log!("output ring buffer readable size at loop end {}", self.output_ring_buffer.num_readable());
    }

    // log!("wet sum: {}", self.wet_buffer.iter().sum::<f32>());

    let output_gain = decibel_to_gain(self.output_gain_decibel);
    // log!("dry: {}, wet: {}, gain: {}", dry_level, wet_level, output_gain);

    for (i, elem) in buffer.iter_mut().enumerate() {
      let tmp = (*elem * dry_level) + self.wet_buffer[i] * wet_level;
      // log!("tmp: {}", tmp);
      *elem = (-1.0f32).max(1.0f32.min(tmp * output_gain));
    }

    // log!("output sum: {}", buffer.iter().sum::<f32>());

  }

  fn process_fft_block(&mut self) {
    let read_result = self.input_ring_buffer.read(&mut self.tmp_buffer);
    if !read_result {
      // log!("[ERROR] failed to read from input ring buffer.");
    }
    let discard_result = self.input_ring_buffer.discard(self.fft_size);
    if !discard_result {
      // log!("[ERROR] failed to discard input ring buffer.");
    }

    // log!("write output ring buffer: {}", &self.tmp_buffer.len());
    // log!("output ring buffer writable size {}", self.output_ring_buffer.num_writable());
    let write_result = self.output_ring_buffer.write(&self.tmp_buffer);
    if !write_result {
      // log!("[ERROR] failed to write into output ring buffer.");
    }

    // log!("output ring buffer readable size: {}", &self.output_ring_buffer.num_readable());
  }

  fn wrap_phase(phase_in: f32) -> f32 {
    let pi = std::f32::consts::PI;
    if phase_in >= 0.0f32 {
      (phase_in + pi) % (2.0f32 * pi) - pi
    } else {
      (phase_in - pi) % (-2.0f32 * pi) + pi
    }
  }

  // pub fn set_wet_amount(&mut self, value: f32) {
  //   self.delay.wet_amount = value;
  // }

  // pub fn set_feedback_amount(&mut self, value: f32) {
  //   log!("wet-amount: {}, feedback: {}, length: {}", self.delay.wet_amount, self.delay.feedback_amount, self.delay.length);

  //   self.delay.feedback_amount = std::cmp::min(value, 0.95);
  // }

  // pub fn set_delay_length(&mut self, value: f32) {
  //   let mut len = unsafe { (value * self.sample_rate as f32).to_int_unchecked::<usize>() };
  //   len = std::cmp::max(len, 10);
  //   self.delay = DelayLine::new(len, self.delay.wet_amount, self.delay.feedback_amount);
  // }
}
