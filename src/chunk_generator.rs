use std::collections::{HashMap, HashSet};

use noise::{core::simplex::simplex_2d, permutationtable::PermutationTable};

static mut PERMTABLE: Option<PermutationTable> = None;

const CHUNK_X: usize = 36;
const CHUNK_Y: usize = 256;
const CHUNK_Z: usize = 36;
const CHUNK_TSIZE: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;

#[allow(dead_code, unused_variables)]
pub struct Chunk {
    // position: [i32; 4],
    voxels: Vec<[u32; CHUNK_TSIZE]>,

    chunks: Vec<[i32; 4]>,

    generated_chunks: HashMap<[i32; 4], usize>,

    chunks_mem: Vec<u32>,
    chunks_mem_free: Vec<usize>,
}

#[allow(dead_code, unused_variables)]
impl Chunk {
    pub fn init() -> Self {
        Self {
            chunks: Vec::new(),
            voxels: Vec::new(),

            generated_chunks: HashMap::new(),
            chunks_mem: Vec::new(),
            chunks_mem_free: Vec::new(),
        }
    }

    pub fn new(&mut self, position: [i32; 4]) {
        unsafe {
            if PERMTABLE.is_none() {
                PERMTABLE = Some(PermutationTable::new(0));
            }
        }
        // let chunk_offset = self.chunks_mem.len();

        // self.chunks_mem.resize(chunk_offset + CHUNK_TSIZE, 0);

        let chunk_offset = self.get_free_chunk_memory_zone();

        self.generated_chunks
            .insert(position, chunk_offset / CHUNK_TSIZE);

        println!("Generating {:?}", position);
        println!(
            " - Offset {:?} | Index_offset {:?}",
            chunk_offset,
            chunk_offset / CHUNK_TSIZE
        );

        for x in 0..CHUNK_X {
            // for y in 0..256 {
            for z in 0..CHUNK_Z {
                unsafe {
                    let pery = simplex_2d(
                        [
                            (position[0] + x as i32) as f64,
                            (position[2] + z as i32) as f64,
                        ],
                        &PERMTABLE.unwrap(),
                    );
                    let y = (128.0 - (pery.0 * (CHUNK_Y as f64)))
                        .ceil()
                        .min(CHUNK_Y as f64)
                        .max(0.0) as usize;

                    // self.voxels[voxels_id][x + 36 * (y + 256 * z)] = 1;

                    self.chunks_mem[chunk_offset + (x + 36 * (y + 256 * z))] = 1;
                }
            }
            // }
        }
    }

    pub fn generate_around(&mut self, player_pos: [f32; 4]) {
        for x in player_pos[0] as i32 - 15..player_pos[0] as i32 + 15 {
            for z in player_pos[2] as i32 - 15..player_pos[2] as i32 + 15 {
                let pos = [x as i32, 0, z as i32, 0];

                if self.generated_chunks.contains_key(&pos) == true {
                    continue;
                }

                self.new(pos)
            }
        }

        println!("=> {}", self.chunks_mem_free.len());

        // self.generated_chunks.

        // println!("{:?}", self.voxels.len());

        let farthest_chunks = self.clean_farthest_chunk(player_pos, 19.);
        for chunk in farthest_chunks {
            let gen_chunk = self.generated_chunks.get(&chunk);
            let chunk_offset_id = gen_chunk.unwrap().clone();

            for i in chunk_offset_id * CHUNK_TSIZE..(chunk_offset_id + 1) * CHUNK_TSIZE {
                self.chunks_mem[i] = 0;
            }

            self.chunks_mem_free.push(chunk_offset_id);
            self.generated_chunks.remove(&chunk);
        }

        // for (index, value) in self.chunks_mem.iter().enumerate() {
        //     print!("{}", value);
        //     if index % CHUNK_TSIZE == 0 {
        //         println!("-");
        //     }
        // }

        println!("=> {}", self.chunks_mem_free.len());

        for free in &self.chunks_mem_free {
            println!(
                "{:4} | {:10} - {}",
                free,
                free * CHUNK_TSIZE,
                (free + 1) * CHUNK_TSIZE
            );
        }

        // self.clean_farthest_chunk(player_pos, 19.);
    }

    pub fn get_free_chunk_memory_zone(&mut self) -> usize {
        if self.chunks_mem_free.is_empty() {
            let chunk_offset = self.chunks_mem.len();
            self.chunks_mem.resize(chunk_offset + CHUNK_TSIZE, 0);
            return chunk_offset;
        }
        let free_zone = self.chunks_mem_free.pop().unwrap() * CHUNK_TSIZE;

        println!("Reusing available zone : {}", free_zone);
        return free_zone;
    }

    fn clean_farthest_chunk(&self, player_pos: [f32; 4], max_dist: f32) -> Vec<[i32; 4]> {
        let mut fartest_chunks: Vec<[i32; 4]> = Vec::new();

        for chunk in &self.generated_chunks {
            let dist = Self::get_chunk_distance(player_pos, chunk.0);

            if dist < max_dist {
                continue;
            }

            fartest_chunks.push(chunk.0.clone());
        }

        println!("Chunk too far count : {}", fartest_chunks.len());

        return fartest_chunks;
    }

    fn get_chunk_distance(player_pos: [f32; 4], chunk_pos: &[i32; 4]) -> f32 {
        let v: [f32; 4] = [
            player_pos[0] - chunk_pos[0] as f32,
            player_pos[1] - chunk_pos[1] as f32,
            player_pos[2] - chunk_pos[2] as f32,
            0.0,
        ];

        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

        return len;
    }
}
