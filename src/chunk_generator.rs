use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use cgmath::num_traits::Float;
use noise::{core::simplex::simplex_2d, permutationtable::PermutationTable};

static mut PERMTABLE: Option<PermutationTable> = None;

const CHUNK_X: usize = 36;
const CHUNK_Y: usize = 256;
const CHUNK_Z: usize = 36;
pub const CHUNK_TSIZE: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;
const CHUNK_RADIUS: i32 = 8;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct ChunkGpuBVHNode {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub data: [u32; 4], // 0: entry_index, 1: exit_index, 2: chunk_id.
    pub _padding: [f32; 4],
}

#[allow(dead_code)]
impl ChunkGpuBVHNode {
    pub fn new(
        // position: [i32; 3],
        aabb: &bvh::aabb::AABB,
        entry_index: u32,
        exit_index: u32,
        chunk_id: u32,
    ) -> ChunkGpuBVHNode {
        ChunkGpuBVHNode {
            aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
            aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z, 0.0],
            // position: [(position[0]) as f32, 0.0, (position[2]) as f32, 0.0],
            data: [entry_index, exit_index, chunk_id, 0],
            // entry_index,
            // exit_index,
            // chunk_id,
            // _padding: 0,
            _padding: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

pub struct ChunkData {
    position: [i32; 3],
    bvh_index: usize,
}

impl bvh::aabb::Bounded for ChunkData {
    fn aabb(&self) -> bvh::aabb::AABB {
        let uwu = bvh::aabb::AABB::with_bounds(
            bvh::Point3::new(
                self.position[0] as f32 * CHUNK_X as f32,
                0.0,
                self.position[2] as f32 * CHUNK_Z as f32,
            ),
            bvh::Point3::new(
                CHUNK_X as f32 + (self.position[0] as f32 * CHUNK_X as f32),
                CHUNK_Y as f32,
                CHUNK_Z as f32 + (self.position[2] as f32 * CHUNK_Z as f32),
            ),
        );

        // println!("{:?} - min {:?} max {:?}", self.position, uwu.min, uwu.max);
        return uwu;
    }
}

impl bvh::bounding_hierarchy::BHShape for ChunkData {
    fn set_bh_node_index(&mut self, index: usize) {
        self.bvh_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.bvh_index
    }
}

#[allow(dead_code, unused_variables)]
pub struct Chunk {
    // position: [i32; 4],
    // voxels: Vec<[u32; CHUNK_TSIZE]>,

    // chunks: Vec<[i32; 4]>,
    generated_chunks: HashMap<[i32; 4], usize>,

    // pub generated_chunks_gpu: Vec<[f32; 4]>,
    pub generated_chunks_gpubvh: Vec<ChunkGpuBVHNode>,
    pub chunks_mem: Vec<u32>,
    chunks_mem_free: Vec<usize>,

    chunk_to_upload: HashSet<usize>,

    // test_pos: [f32; 3],
    // last_pos: [f32; 3],
    pub chunks_uniform_grod: Vec<[u32; 4]>,
}

#[allow(dead_code, unused_variables)]
impl Chunk {
    pub fn init() -> Self {
        Self {
            // chunks: Vec::new(),
            // voxels: Vec::new(),
            generated_chunks: HashMap::new(),

            // generated_chunks_gpu: Vec::new(),
            generated_chunks_gpubvh: Vec::new(),
            chunks_mem: Vec::new(),
            chunks_mem_free: Vec::new(),

            chunk_to_upload: HashSet::new(),

            // test_pos: [0.0, 0.0, 0.0],
            // last_pos: [0.0, 0.0, 0.0],
            chunks_uniform_grod: Vec::new(),
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

        let chunk_offset = self.get_free_chunk_memory_zone();
        // self.chunks_mem.resize(chunk_offset + CHUNK_TSIZE, 0);

        self.generated_chunks
            .insert(position, chunk_offset / CHUNK_TSIZE);

        // println!("{}", chunk_offset);
        // println!("Generating {:?}", position);
        // println!(
        //     " - Offset {:?} | Index_offset {:?}",
        //     chunk_offset,
        //     chunk_offset / CHUNK_TSIZE
        // );

        // println!("{}", chunk_offset);

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

                    // println!("Y {}", y);/

                    for y in (0..y).rev() {
                        let index = (z * CHUNK_X * CHUNK_Y) + (y * CHUNK_X) + x;
                        // let index = y * CHUNK_X * CHUNK_Y + z * CHUNK_X + x;

                        // println!("{} {}", y, index);
                        self.chunks_mem[chunk_offset + index] = 1;
                        // self.chunks_mem[chunk_offset + index] = 1;
                        // print!("{:?}", self.chunks_mem[chunk_offset + index]);
                    }

                    // self.voxels[voxels_id][x + 36 * (y + 256 * z)] = 1;

                    // self.chunks_mem[chunk_offset + (x + 36 * (y + 256 * z))] = 1;
                }
            }
            // }
        }
        // println!("");

        self.chunk_to_upload.insert(chunk_offset / CHUNK_TSIZE);

        // println!("Chunks to GPU-Update {}", self.chunk_to_upload.len());
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
        println!(
            "+++ Generated Around : {}ms",
            time_to_generate_arount_start.elapsed().as_millis()
        );

        // let pos = [0, 0, 0, 0];
        // if self.generated_chunks.contains_key(&pos) == false {
        //     self.new(pos);
        // }
        // let pos = [1, 0, 0, 0];
        // if self.generated_chunks.contains_key(&pos) == false {
        //     self.new(pos);
        // }
        // let pos = [-1, 0, 0, 0];
        // if self.generated_chunks.contains_key(&pos) == false {
        //     self.new(pos);
        // }

        println!("{}", self.chunks_mem.len());

        if generated_count > 0 {
            println!(
                "=> Generated chunks {}. New Total : {}",
                generated_count,
                self.generated_chunks.len()
            );
        }

        println!("{:?} {:?}", player_pos, player_pos_chunk);
        // if self.test_pos[0] == 0.0

        // if player_pos[0] != self.last_pos[0] {
        //     self.test_pos[0] += player_pos[0] - self.last_pos[0];

        //     if self.test_pos[0].abs() >= 1.0 {
        //         self.test_pos[0] = self.test_pos[0].fract()
        //     }

        //     self.last_pos[0] = player_pos[0];
        // }
        // if player_pos[2] != self.last_pos[2] {
        //     self.test_pos[2] += player_pos[2] - self.last_pos[2];

        //     if self.test_pos[2].abs() >= 1.0 {
        //         self.test_pos[2] = self.test_pos[2].fract()
        //     }

        //     self.last_pos[2] = player_pos[2];
        // }

        // for chunk in &self.generated_chunks {
        //     println!("{:?}", chunk);
        // }

        // println!(" {:?} - {:?} ", self.test_pos, player_pos);

        // for z in player_pos_chunk[2] - CHUNK_RADIUS..player_pos_chunk[2] + CHUNK_RADIUS {
        //     for x in player_pos_chunk[0] - CHUNK_RADIUS..player_pos_chunk[0] + CHUNK_RADIUS {
        //         print!(" [{:2} {:2}] ", x, z);
        //     }
        //     println!("");
        //     // println!("{} {}", x, player_pos_chunk[0] + CHUNK_RADIUS);

        //     // if x >= player_pos_chunk[0] + CHUNK_RADIUS {
        //     //     println!("");
        //     // }
        // }

        // self.generated_chunks.

        // println!("{:?}", self.voxels.len());
        // return ;

        let mut unloaded_chunks = 2;
        let farthest_chunks = self.clean_farthest_chunk(player_pos_chunk, 9.);
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
            println!(
                "Unloaded chunks : {}. New Total : {}",
                unloaded_chunks,
                self.generated_chunks.len()
            );
        }

        println!(
            "+++ Unload Chunks : {}ms",
            time_to_generate_arount_start.elapsed().as_millis()
        );
        // println!("");
        // println!("{:?}", player_pos);

        let mut chunks_as_vecforbvhtest: Vec<ChunkData> = Vec::new();

        // self.generated_chunks_gpu.clear();
        for chunk in &self.generated_chunks {
            // self.generated_chunks_gpu
            //     .push([chunk.0[0] as f32, 0.0, chunk.0[2] as f32, 1.0]);

            chunks_as_vecforbvhtest.push(ChunkData {
                bvh_index: 0,
                position: [chunk.0[0] as i32, 0, chunk.0[2] as i32],
            });

            // println!("=> {:?}", chunk.0);
        }

        println!(
            "+++ Prepare for BVH : {}ms",
            time_to_generate_arount_start.elapsed().as_millis()
        );
        // println!("{:?}", self.generated_chunks_gpu.len());

        // println!("=> {}", chunks_as_vecforbvhtest.len());

        let startbvhtime = Instant::now();
        let bvh = bvh::bvh::BVH::build(&mut chunks_as_vecforbvhtest);

        let custom_constructor = |aabb: &bvh::aabb::AABB, entry, exit, shape_index| {
            ChunkGpuBVHNode::new(aabb, entry, exit, shape_index)
        };
        self.generated_chunks_gpubvh.clear();

        self.generated_chunks_gpubvh = bvh.flatten_custom(&custom_constructor);
        println!(
            "Len {}; Size : {}",
            self.generated_chunks_gpubvh.len(),
            self.generated_chunks_gpubvh.len() * 32
        );

        println!(
            "+++ BVH Build : {} us",
            time_to_generate_arount_start.elapsed().as_micros()
        );

        println!(
            "Elapsed total {} ms",
            time_to_generate_arount_start.elapsed().as_millis()
        );

        // for (index, flat) in self.generated_chunks_gpubvh.iter().enumerate() {
        //     let size = ((flat.aabb_max[0] - flat.aabb_min[0])
        //         * (flat.aabb_max[0] - flat.aabb_min[0]))
        //         + ((flat.aabb_max[2] - flat.aabb_min[2]) * (flat.aabb_max[2] - flat.aabb_min[2]));

        //     println!(
        //         "{:>10} - [{:?} {:?}] - {:?} => {}",
        //         index,
        //         flat.aabb_min,
        //         flat.aabb_max,
        //         flat.data,
        //         size.sqrt()
        //     );
        // }
        // if flat.data[0] == 4294967295 {
        //     let cul = &chunks_as_vecforbvhtest[flat.data[2] as usize];
        //     println!("cul {:?}", cul.position);
        // }

        //     println!(
        //         "Polygon(({}, {}, {}), ({}, {}, {}), ({}, {}, {}), ({}, {}, {}))",
        //         flat.aabb_min[0],
        //         flat.aabb_min[1],
        //         flat.aabb_min[2],

        //         flat.aabb_min[0],
        //         flat.aabb_min[1],
        //         flat.aabb_max[2],

        //         flat.aabb_max[0],
        //         flat.aabb_min[1],
        //         flat.aabb_max[2],
        //         flat.aabb_max[0],
        //         flat.aabb_min[1],
        //         flat.aabb_min[2],
        //     );
        // }

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

            println!(
                "Empty, new size : {} -> {} ",
                chunk_offset,
                chunk_offset + CHUNK_TSIZE
            );
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
