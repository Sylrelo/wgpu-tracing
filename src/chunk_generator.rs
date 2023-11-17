use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use perlin2d::PerlinNoise2D;

// const CHUNK_X: usize = 36;
// const CHUNK_Y: usize = 256;
// const CHUNK_Z: usize = 36;
const CHUNK_X: usize = 64;
const CHUNK_Y: usize = 64;
const CHUNK_Z: usize = 64;
pub const CHUNK_TSIZE: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;
pub const CHUNK_MEM_OFFSET: usize = CHUNK_TSIZE; // 995326;
                                                 // pub const CHUNK_MEM_OFFSET: usize = 1000000; // 995326;
const CHUNK_RADIUS: i32 = 8;

#[allow(dead_code, unused_variables)]
pub struct Chunk {
    generated_chunks: HashMap<[i32; 4], usize>,

    pub chunks_mem: Vec<u32>,
    chunks_mem_free: Vec<usize>,

    chunk_to_upload: HashSet<usize>,

    pub root_chunks: Vec<[i32; 4]>,
    pub root_grid: Vec<[i32; 4]>,
}

#[allow(dead_code, unused_variables)]
impl Chunk {
    pub fn init() -> Self {
        Self {
            generated_chunks: HashMap::new(),

            chunks_mem: Vec::new(),
            chunks_mem_free: Vec::new(),

            chunk_to_upload: HashSet::new(),

            root_chunks: Vec::new(),
            root_grid: Vec::new(),
        }
    }

    pub fn new(&mut self, position: [i32; 4]) {
        let chunk_offset = self.get_free_chunk_memory_zone();

        self.generated_chunks
            .insert(position, chunk_offset / CHUNK_MEM_OFFSET);

        // println!("{:?}", position);

        for x in 0..CHUNK_X {
            for z in 0..CHUNK_Z {
                let index = (z * CHUNK_X * CHUNK_Y) + (24 * CHUNK_X) + x;
                self.chunks_mem[chunk_offset + index] = 3;
            }
        }

        for x in 0..CHUNK_X {
            for z in 0..CHUNK_Z {
                let pos = [
                    ((position[0] * CHUNK_X as i32 + x as i32) as f64).abs(),
                    ((position[2] * CHUNK_Z as i32 + z as i32) as f64).abs(),
                ];

                let pepe =
                    PerlinNoise2D::new(6, 10.0, 0.5, 1.1, 2.0, (100.0, 100.0), 40.0, 1346139461);

                let yp = pepe.get_noise(pos[0], pos[1]);

                let y = (yp as usize).min(256);

                for y in (0..y).rev() {
                    let index = (z * CHUNK_X * CHUNK_Y) + (y * CHUNK_X) + x;
                    self.chunks_mem[chunk_offset + index] = 1;
                }
            }
        }

        self.root_chunks.push([
            ((position[0]) * CHUNK_X as i32),
            0,
            ((position[2]) * CHUNK_Z as i32),
            chunk_offset as i32,
        ]);

        self.chunk_to_upload.insert(chunk_offset / CHUNK_MEM_OFFSET);
    }

    pub fn generate_around(&mut self, player_pos: [f32; 4]) {
        let time_to_generate_arount_start = Instant::now();
        let mut generated_count = 0;
        let player_pos_chunk = [
            (player_pos[0] / CHUNK_X as f32) as i32,
            0,
            (player_pos[2] / CHUNK_Z as f32) as i32,
        ];

        for x in player_pos_chunk[0] as i32..player_pos_chunk[0] as i32 + CHUNK_RADIUS {
            for z in player_pos_chunk[2] as i32..player_pos_chunk[2] as i32 + CHUNK_RADIUS {
                let pos = [x as i32, 0, z as i32, 0];

                if self.generated_chunks.contains_key(&pos) == true {
                    continue;
                }

                generated_count += 1;
                self.new(pos)
            }
        }

        if generated_count > 0 {
            println!(
                "+++ Generated Around : {}ms",
                time_to_generate_arount_start.elapsed().as_millis()
            );
            println!(
                "=> Generated chunks {}. New Total : {}",
                generated_count,
                self.generated_chunks.len()
            );
        }

        self.root_grid.resize(30 * 30, [0; 4]);

        for x in 0..30 {
            for z in 0..30 {
                let chunk_pos = [(x), 0, (z), 0];

                let chk = self.generated_chunks.get(&chunk_pos);
                self.root_grid[(x + z * 30) as usize] = [0, 0, 0, 0];

                if chk.is_none() {
                    // print!("       | ");
                } else {
                    // print!("{:5} | ", self.root_grid[x + z * 20]);
                    let val = chk.unwrap();
                    // print!("{:8} | ", val * CHUNK_TSIZE + 1);
                    // print!("{:2}  {:2} | ", chunk_pos[0], chunk_pos[2]);
                    self.root_grid[(x + z * 30) as usize] = [
                        (chunk_pos[0]) * CHUNK_X as i32,
                        0,
                        (chunk_pos[2]) * CHUNK_Z as i32,
                        (*val as i32) * CHUNK_MEM_OFFSET as i32 + 1,
                    ];
                }
            }
            // println!("");
        }
        println!("-> {}", self.generated_chunks.len());
        // println!("{:?} {:?}", player_pos, player_pos_chunk);

        // println!("Tot {} ({} ms)", a, start_timer.elapsed().as_millis());
        // let mut unloaded_chunks = 2;
        // let farthest_chunks = self.clean_farthest_chunk(player_pos_chunk, 9.);
        // for chunk in farthest_chunks {
        //     let gen_chunk = self.generated_chunks.get(&chunk);
        //     let chunk_offset_id = gen_chunk.unwrap().clone();

        //     for i in chunk_offset_id * CHUNK_TSIZE..(chunk_offset_id + 1) * CHUNK_TSIZE {
        //         self.chunks_mem[i] = 0;
        //     }

        //     self.chunks_mem_free.push(chunk_offset_id);
        //     self.generated_chunks.remove(&chunk);
        //     unloaded_chunks += 1;
        // }

        // if unloaded_chunks > 0 {
        //     println!(
        //         "Unloaded chunks : {}. New Total : {}",
        //         unloaded_chunks,
        //         self.generated_chunks.len()
        //     );
        // }

        // println!(
        //     "+++ Unload Chunks : {}ms",
        //     time_to_generate_arount_start.elapsed().as_millis()
        // );
    }

    pub fn get_free_chunk_memory_zone(&mut self) -> usize {
        if self.chunks_mem_free.is_empty() {
            let chunk_offset = self.chunks_mem.len();
            self.chunks_mem.resize(chunk_offset + CHUNK_MEM_OFFSET, 0);

            return chunk_offset;
        }
        let free_zone = self.chunks_mem_free.pop().unwrap() * CHUNK_MEM_OFFSET;

        return free_zone;
    }

    fn clean_farthest_chunk(&self, player_pos_chunk: [i32; 3], max_dist: f32) -> Vec<[i32; 4]> {
        let mut fartest_chunks: Vec<[i32; 4]> = Vec::new();

        for chunk in &self.generated_chunks {
            let dist = Self::get_chunk_distance(player_pos_chunk, chunk.0);

            if dist < max_dist {
                continue;
            }

            fartest_chunks.push(chunk.0.clone());
        }

        return fartest_chunks;
    }

    fn get_chunk_distance(player_pos_chunk: [i32; 3], chunk_pos: &[i32; 4]) -> f32 {
        let v: [f32; 2] = [
            player_pos_chunk[0] as f32 - chunk_pos[0] as f32,
            player_pos_chunk[2] as f32 - chunk_pos[2] as f32,
        ];

        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();

        return len;
    }
}
