use wasm_bindgen::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

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

#[derive(Debug)]
struct DelayLine
{
  length: usize,
  pos: usize,
  array: Vec<f32>,
  wet_amount: f32,
  feedback_amount: f32,
}

impl DelayLine
{
  pub fn new(length: usize, wet_amount: f32, feedback_amount: f32) -> DelayLine
  {
    DelayLine {
      length,
      pos: 0,
      array: vec![0.0; length],
      wet_amount,
      feedback_amount
    }
  }

  pub fn process(&mut self, sample: f32) -> f32
  {
//   let before = sample;

    let dry_amount = 1.0 - self.wet_amount;
    let p = &mut self.array[self.pos];

    let wet_sample = sample + *p * self.feedback_amount;
    *p = wet_sample;
    self.pos = (self.pos + 1) % self.length;

    (sample * dry_amount) + (wet_sample * self.wet_amount)
//     log!("pos: {}, before: {}, after: {}", self.pos, before, x);
  }

  pub fn dump(&self) -> String {
    format!("{:?}", self)
  }
}

#[wasm_bindgen]
pub struct WasmProcessor {
  sample_rate: usize,
  phase: f32,
  delay: DelayLine,
  input_level: f32,
  output_level: f32,
}

#[wasm_bindgen]
impl WasmProcessor {
  pub fn new(
    sample_rate: usize,
    block_size: usize,
    initial_delay_length: f32,
    initial_wet_amount: f32,
    initial_feedback_amount: f32
  ) -> WasmProcessor {
    utils::set_panic_hook();

    unsafe { MIN_DECIBEL_GAIN = decibel_to_gain(MIN_DECIBEL); };

    let len = unsafe { (initial_delay_length * sample_rate as f32).to_int_unchecked::<usize>() };
    log!("initial-delay-length {}", initial_delay_length);
    log!("len {}", len);

    let d = WasmProcessor {
      sample_rate,
      phase: 0.0,
      delay: DelayLine::new(len, initial_wet_amount, initial_feedback_amount),
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
