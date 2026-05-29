//! PRNG generator trait and concrete engine types.

/// Trait implemented by all mohu PRNG engines.
pub trait Generator: Send + Sync {
    /// Seed the generator from a u64.
    fn seed(seed: u64) -> Self
    where
        Self: Sized;
    /// Fill a byte slice with random bytes.
    fn fill_bytes(&mut self, dest: &mut [u8]);
    /// Return a random u64.
    fn next_u64(&mut self) -> u64;
}

/// PCG-64-DXSM generator — fast, statistically strong, default engine.
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.inc);
    }

    fn output(state: u128) -> u64 {
        // DXSM output function
        let hi = (state >> 64) as u64;
        let lo = (state | 1) as u64;
        let hi = hi ^ (hi >> 32);
        let hi = hi.wrapping_mul(0xDA94_2042_E4DD_58B5);
        let hi = hi ^ (hi >> 48);
        hi.wrapping_mul(lo)
    }
}

impl Generator for Pcg64 {
    fn seed(seed: u64) -> Self {
        let mut g = Pcg64 {
            state: 0,
            inc: (seed as u128).wrapping_shl(1) | 1,
        };
        g.step();
        g.state = g.state.wrapping_add(seed as u128);
        g.step();
        g
    }

    fn next_u64(&mut self) -> u64 {
        let state = self.state;
        self.step();
        Self::output(state)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let val = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&val);
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let val = self.next_u64().to_le_bytes();
            rem.copy_from_slice(&val[..rem.len()]);
        }
    }
}

/// Philox 4×64 counter-based PRNG — stateless, GPU-friendly.
pub struct Philox4x64 {
    counter: [u64; 4],
    key: [u64; 2],
    buffer: [u64; 4],
    buffer_pos: usize,
}

impl Philox4x64 {
    fn round(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
        const M0: u64 = 0xD2E7470EE14C6C93;
        const M1: u64 = 0xCA5A826395121157;
        let (hi0, lo0) = mul128(M0, ctr[0]);
        let (hi1, lo1) = mul128(M1, ctr[2]);
        [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
    }

    fn generate(&mut self) {
        let mut x = self.counter;
        let mut k = self.key;
        for _ in 0..10 {
            x = Self::round(x, k);
            k[0] = k[0].wrapping_add(0x9E3779B97F4A7C15);
            k[1] = k[1].wrapping_add(0xBB67AE8584CCA73B);
        }
        self.buffer = x;
        self.buffer_pos = 0;
        // advance counter
        self.counter[0] = self.counter[0].wrapping_add(1);
        if self.counter[0] == 0 {
            self.counter[1] = self.counter[1].wrapping_add(1);
        }
    }
}

fn mul128(a: u64, b: u64) -> (u64, u64) {
    let product = (a as u128).wrapping_mul(b as u128);
    ((product >> 64) as u64, product as u64)
}

impl Generator for Philox4x64 {
    fn seed(seed: u64) -> Self {
        let mut g = Philox4x64 {
            counter: [0, 0, 0, 0],
            key: [seed, seed.wrapping_mul(0x9E3779B97F4A7C15)],
            buffer: [0; 4],
            buffer_pos: 4,
        };
        g.generate();
        g
    }

    fn next_u64(&mut self) -> u64 {
        if self.buffer_pos >= 4 {
            self.generate();
        }
        let val = self.buffer[self.buffer_pos];
        self.buffer_pos += 1;
        val
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let val = self.next_u64().to_le_bytes();
            rem.copy_from_slice(&val[..rem.len()]);
        }
    }
}
