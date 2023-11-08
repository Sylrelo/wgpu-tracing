use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use noise::{core::simplex::simplex_2d, permutationtable::PermutationTable};

static mut PERMTABLE: Option<PermutationTable> = None;

const CHUNK_X: usize = 36;
const CHUNK_Y: usize = 256;
const CHUNK_Z: usize = 36;
pub const CHUNK_TSIZE: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;
const CHUNK_RADIUS: i32 = 7;

#[allow(dead_code, unused_variables)]
pub struct Chunk {
    generated_chunks: HashMap<[i32; 4], usize>,

    pub chunks_mem: Vec<u32>,
    chunks_mem_free: Vec<usize>,

    chunk_to_upload: HashSet<usize>,

    pub root_chunks: Vec<[i32; 4]>,
    pub root_grid: Vec<[i32; 4]>,
    // uniform_grid: Vec<u32>,
    // generated_chunks_voxs: HashMap<[i32; 3], Vec<u32>>,
}

pub fn octavia_spencer(
    point: [f64; 2],
    iter_max: u32,
    persistence: f64,
    scale: f64,
    min: f64,
    max: f64,
) -> f64 {
    unsafe {
        if PERMTABLE.is_none() {
            PERMTABLE = Some(PermutationTable::new(436457824));
        }
    }
    let mut noise = 0.0f64;
    let mut amp = 1.0f64;
    let mut freq = scale;
    let mut max_amp = 0.0f64;

    unsafe {
        for _ in 0..iter_max {
            noise +=
                simplex_2d([(point[0] * freq), (point[1] * freq)], &PERMTABLE.unwrap()).0 * amp;
            max_amp += amp;
            amp *= persistence;
            freq *= 2.0;
        }
    }

    noise /= max_amp;

    noise = noise * (max - min) / 2.0 + (max + min) / 2.0;

    return noise;
}

#[allow(dead_code, unused_variables)]
impl Chunk {
    pub fn init() -> Self {
        Self {
            // chunks: Vec::new(),
            // voxels: Vec::new(),
            generated_chunks: HashMap::new(),

            // generated_chunks_gpu: Vec::new(),
            chunks_mem: Vec::new(),
            chunks_mem_free: Vec::new(),

            chunk_to_upload: HashSet::new(),
            // test_pos: [0.0, 0.0, 0.0],
            // last_pos: [0.0, 0.0, 0.0],
            // chunks_uniform_grod: Vec::new(),
            // uniform_grid: Vec::with_capacity(CHUNK_TSIZE * 1225),
            // generated_chunks_voxs: HashMap::new(),
            root_chunks: Vec::new(),
            root_grid: Vec::new(),
        }
    }

    pub fn new(&mut self, position: [i32; 4]) {
        let chunk_offset = self.get_free_chunk_memory_zone();

        self.generated_chunks
            .insert(position, chunk_offset / CHUNK_TSIZE);

        let mut yo: Vec<u32> = Vec::with_capacity(CHUNK_TSIZE);
        yo.resize(CHUNK_TSIZE, 0u32);

        for x in 0..CHUNK_X {
            for z in 0..CHUNK_Z {
                let pos = [
                    (((position[0]) * CHUNK_X as i32 + x as i32) as f64),
                    (((position[2]) * CHUNK_Z as i32 + z as i32) as f64),
                ];

                let y = octavia_spencer(pos, 16, 0.2, 0.006, 0.0, 255.0) as usize;

                for y in (0..y).rev() {
                    let index = (z * CHUNK_X * CHUNK_Y) + (y * CHUNK_X) + x;
                    // self.chunks_mem[chunk_offset + index] = 1;
                    yo[index] = 1;
                }
            }
        }

        self.root_chunks.push([
            ((position[0]) * CHUNK_X as i32),
            0,
            ((position[2]) * CHUNK_Z as i32),
            0,
        ]);

        // self.generated_chunks_voxs
        //     .insert([position[0], position[1], position[2]], yo);

        self.chunk_to_upload.insert(chunk_offset / CHUNK_TSIZE);
    }

    pub fn generate_around(&mut self, player_pos: [f32; 4]) {
        let time_to_generate_arount_start = Instant::now();
        let mut generated_count = 0;
        let player_pos_chunk = [
            (player_pos[0] / CHUNK_X as f32) as i32,
            0,
            (player_pos[2] / CHUNK_Z as f32) as i32,
        ];

        for x in
            player_pos_chunk[0] as i32 - CHUNK_RADIUS..player_pos_chunk[0] as i32 + CHUNK_RADIUS
        {
            for z in
                player_pos_chunk[2] as i32 - CHUNK_RADIUS..player_pos_chunk[2] as i32 + CHUNK_RADIUS
            {
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
                let chk = self.generated_chunks.get(&[15 - (x), 0, 15 - (z), 0]);
                if chk.is_none() {
                    print!("{:2} {:2} | ", x, z);
                } else {
                    // print!("{:5} | ", self.root_grid[x + z * 20]);
                    let val = chk.unwrap();
                    print!("{:5} | ", val);
                    self.root_grid[(x + z * 30) as usize] = [
                        (15 - (x)) * CHUNK_X as i32,
                        0,
                        (15 - (z)) * CHUNK_Z as i32,
                        *val as i32 + 1,
                    ];
                }
            }
            println!("");
        }
        println!("-> {}", self.root_chunks.len() * 16);
        println!("{:?} {:?}", player_pos, player_pos_chunk);

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
            self.chunks_mem.resize(chunk_offset + CHUNK_TSIZE, 0);

            // println!(
            //     "Empty, new size : {} -> {} ",
            //     chunk_offset,
            //     chunk_offset + CHUNK_TSIZE
            // );
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
