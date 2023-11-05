use std::collections::{HashMap, HashSet};

use noise::{core::simplex::simplex_2d, permutationtable::PermutationTable};

static mut PERMTABLE: Option<PermutationTable> = None;

const CHUNK_X: usize = 36;
const CHUNK_Y: usize = 256;
const CHUNK_Z: usize = 36;
const CHUNK_TSIZE: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;
const CHUNK_RADIUS: i32 = 5;

// pub struct ChunksAabbGpu {
//     position: [f32; 4],
// }

#[allow(dead_code, unused_variables)]
pub struct Chunk {
    // position: [i32; 4],
    // voxels: Vec<[u32; CHUNK_TSIZE]>,

    // chunks: Vec<[i32; 4]>,

    generated_chunks: HashMap<[i32; 4], usize>,
    pub generated_chunks_gpu: Vec<[f32; 4]>,

    chunks_mem: Vec<u32>,
    chunks_mem_free: Vec<usize>,

    chunk_to_upload: HashSet<usize>,
}

#[allow(dead_code, unused_variables)]
impl Chunk {
    pub fn init() -> Self {
        Self {
            // chunks: Vec::new(),
            // voxels: Vec::new(),

            generated_chunks: HashMap::new(),
            generated_chunks_gpu: Vec::new(),

            chunks_mem: Vec::new(),
            chunks_mem_free: Vec::new(),

            chunk_to_upload: HashSet::new()
        }
    }

    pub fn new(&mut self, position: [i32; 4]) {
        unsafe {
            if PERMTABLE.is_none() {
                PERMTABLE = Some(PermutationTable::new(0));
            }
        }

        // self.generated_chunks.iter().
        // let chunk_offset = self.chunks_mem.len();

        // self.chunks_mem.resize(chunk_offset + CHUNK_TSIZE, 0);

        let chunk_offset = self.get_free_chunk_memory_zone();

        self.generated_chunks
            .insert(position, chunk_offset / CHUNK_TSIZE);

        // println!("Generating {:?}", position);
        // println!(
        //     " - Offset {:?} | Index_offset {:?}",
        //     chunk_offset,
        //     chunk_offset / CHUNK_TSIZE
        // );

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

        self.chunk_to_upload.insert(chunk_offset / CHUNK_TSIZE);

        // println!("Chunks to GPU-Update {}", self.chunk_to_upload.len());

    }

    pub fn generate_around(&mut self, player_pos: [f32; 4]) {
        let mut generated_count = 0;
        let player_pos_chunk = [
            (player_pos[0] / CHUNK_X as f32) as i32,
            0,
            (player_pos[2] / CHUNK_Z as f32) as i32,
        ];
        

        for x in player_pos_chunk[0] as i32 - CHUNK_RADIUS..player_pos_chunk[0] as i32 + CHUNK_RADIUS {
            for z in player_pos_chunk[2] as i32 - CHUNK_RADIUS..player_pos_chunk[2] as i32 + CHUNK_RADIUS {
                let pos = [x as i32, 0, z as i32, 0];

                if self.generated_chunks.contains_key(&pos) == true {
                    continue;
                }

                generated_count += 1;
                self.new(pos)
            }
        }

        if generated_count > 0 {
            println!("=> Generated chunks {}. New Total : {}",generated_count, self.generated_chunks.len());
        }


        // self.generated_chunks.

        // println!("{:?}", self.voxels.len());
        // return ;

        let mut unloaded_chunks = 0;

        let farthest_chunks = self.clean_farthest_chunk(player_pos_chunk, 10.);
        for chunk in farthest_chunks {
            let gen_chunk = self.generated_chunks.get(&chunk);
            let chunk_offset_id = gen_chunk.unwrap().clone();

            for i in chunk_offset_id * CHUNK_TSIZE..(chunk_offset_id + 1) * CHUNK_TSIZE {
                self.chunks_mem[i] = 0;
            }

            self.chunks_mem_free.push(chunk_offset_id);
            self.generated_chunks.remove(&chunk);
            unloaded_chunks += 1;
        }

        if unloaded_chunks > 0 {
            println!("Unloaded chunks : {}. New Total : {}", unloaded_chunks, self.generated_chunks.len());
        }

        self.generated_chunks_gpu.clear();
        for chunk in &self.generated_chunks {
            self.generated_chunks_gpu.push([
                chunk.0[0] as f32,
                0.0,
                chunk.0[2] as f32,
                1.0,
            ]);
        }
        println!("{:?}", self.generated_chunks_gpu.len());

        // for (index, value) in self.chunks_mem.iter().enumerate() {
        //     print!("{}", value);
        //     if index % CHUNK_TSIZE == 0 {
        //         println!("-");
        //     }
        // }

        // println!("=> {}", self.chunks_mem_free.len());

        // for free in &self.chunks_mem_free {
        //     println!(
        //         "{:4} | {:10} - {}",
        //         free,
        //         free * CHUNK_TSIZE,
        //         (free + 1) * CHUNK_TSIZE
        //     );
        // }

        // self.clean_farthest_chunk(player_pos, 19.);


    }

    pub fn get_free_chunk_memory_zone(&mut self) -> usize {
        if self.chunks_mem_free.is_empty() {
            let chunk_offset = self.chunks_mem.len();
            self.chunks_mem.resize(chunk_offset + CHUNK_TSIZE, 0);
            return chunk_offset;
        }
        let free_zone = self.chunks_mem_free.pop().unwrap() * CHUNK_TSIZE;

        // println!("Reusing available zone : {}", free_zone);
        return free_zone;
    }

    fn clean_farthest_chunk(&self, player_pos_chunk: [i32; 3], max_dist: f32) -> Vec<[i32; 4]> {
        let mut fartest_chunks: Vec<[i32; 4]> = Vec::new();

        for chunk in &self.generated_chunks {
            let dist = Self::get_chunk_distance(player_pos_chunk, chunk.0);

            // println!("{}", dist);

            if dist < max_dist {
                continue;
            }

            fartest_chunks.push(chunk.0.clone());
        }

        // println!("Chunk too far count : {}", fartest_chunks.len());

        return fartest_chunks;
    }

    fn get_chunk_distance(player_pos_chunk: [i32; 3], chunk_pos: &[i32; 4]) -> f32 {
        let v: [f32; 2] = [
            player_pos_chunk[0] as f32 - chunk_pos[0] as f32,
            player_pos_chunk[2] as f32 - chunk_pos[2] as f32,
        ];

        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();

        // println!("{:?} {:?} {}", player_pos, chunk_pos, len);
        return len;
    }
}
