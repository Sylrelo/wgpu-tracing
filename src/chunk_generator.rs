use std::{
    collections::{HashMap, HashSet},
    process::{self, exit},
    time::Instant,
};

use bvh::{
    self,
    aabb::{Bounded, AABB},
    bounding_hierarchy::BHShape,
    Point3,
};

use perlin2d::PerlinNoise2D;

const CHUNK_X: usize = 36;
const CHUNK_Y: usize = 256;
const CHUNK_Z: usize = 36;
pub const CHUNK_TSIZE: usize = CHUNK_X * CHUNK_Y * CHUNK_Z;
pub const CHUNK_MEM_OFFSET: usize = 1000000; // 995326;
const CHUNK_RADIUS: i32 = 6;

pub struct VoxelGenerated {
    chunk_position: [i32; 3],
    position: [u32; 3],
    voxel_type: u32,
    node_index: usize,
}

pub struct ChunkGenerated {
    position: [f32; 3],
    offset: u32,
    node_index: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuBvhNode {
    pub min: [f32; 4],
    pub max: [f32; 4],
    pub entry: u32,
    pub exit: u32,
    pub offset: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct VoxelBvhNode {
    pub entry: u32,
    pub exit: u32,
    pub aabbmin_n_type: u32,
    pub aabbmax: u32,
}

// 9 + 9 + 6 + 6 = 30
// 20 + 6 = 26
// 20 + 6 = 26
// 26 + 26 + 30 = 82

impl VoxelBvhNode {
    pub fn new(aabb: &AABB, entry: u32, exit: u32, voxel_material: u32) -> Self {
        let xmin = aabb.min.x as u32;
        let ymin = aabb.min.y as u32;
        let zmin = aabb.min.z as u32;

        let xmax = aabb.max.x as u32;
        let ymax = aabb.max.y as u32;
        let zmax = aabb.max.z as u32;

        let mut aabb_xxyy = xmin << 26 | xmax << 20 | ymin << 11 | ymax << 2;
        let mut aabb_zz = zmin << 26 | zmax << 20;

        let entry_shift = if entry == 4294967295 {
            0 << 12 | 1 << 11 | voxel_material << 2
        } else {
            entry << 12 | 0 << 11
        };

        if entry == 4294967295 {
            aabb_xxyy = 0;
            aabb_zz = 0;
        }

        Self {
            entry: entry_shift,
            exit: exit << 12,
            aabbmin_n_type: aabb_xxyy,
            aabbmax: aabb_zz,
        }
    }
}

impl GpuBvhNode {
    pub fn new(aabb: &AABB, entry: u32, exit: u32, offset: u32) -> Self {
        Self {
            min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
            max: [aabb.max.x, aabb.max.y, aabb.max.z, 0.0],
            entry,
            exit,
            offset,
            _padding: 0,
        }
    }
}

impl Bounded for ChunkGenerated {
    fn aabb(&self) -> AABB {
        AABB::with_bounds(
            Point3::new(self.position[0], self.position[1], self.position[2]),
            Point3::new(
                self.position[0] + CHUNK_X as f32,
                self.position[1] + CHUNK_Y as f32,
                self.position[2] + CHUNK_Z as f32,
            ),
        )
    }
}

impl BHShape for ChunkGenerated {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl Bounded for VoxelGenerated {
    fn aabb(&self) -> AABB {
        // let position = [
        //     self.chunk_position[0] + self.position[0] as i32,
        //     self.chunk_position[1] + self.position[1] as i32,
        //     self.chunk_position[2] + self.position[2] as i32,
        // ];

        let ab = AABB::with_bounds(
            Point3::new(
                self.position[0] as f32,
                self.position[1] as f32,
                self.position[2] as f32,
            ),
            Point3::new(
                self.position[0] as f32 + 1.0,
                self.position[1] as f32 + 1.0,
                self.position[2] as f32 + 1.0,
            ),
        );

        // println!("==> {:?}", ab);

        return ab;
    }
}

impl BHShape for VoxelGenerated {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

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
    bvh_generated_chunks: Vec<ChunkGenerated>,
    pub bvh_chunks: Vec<GpuBvhNode>,
    pub bvh_chunk_voxels: Vec<VoxelBvhNode>,
    // pub bvh_chunk_voxels_bitwise: Vec<VoxelBvhNode>,
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

            bvh_generated_chunks: Vec::new(),
            bvh_chunks: Vec::new(),
            bvh_chunk_voxels: Vec::new(),
            // bvh_chunk_voxels_bitwise: Vec::new()
        }
    }

    pub fn new(&mut self, position: [i32; 4]) {
        let chunk_offset = self.get_free_chunk_memory_zone();

        self.generated_chunks
            .insert(position, chunk_offset / CHUNK_MEM_OFFSET);

        // let mut yo: Vec<u32> = Vec::with_capacity(CHUNK_TSIZE);
        // yo.resize(CHUNK_TSIZE, 0u32);

        println!("{:?}", position);

        let mut voxels_gen: Vec<VoxelGenerated> = Vec::new();

        voxels_gen.push(VoxelGenerated {
            chunk_position: [
                ((position[0]) * CHUNK_X as i32),
                0,
                ((position[2]) * CHUNK_Z as i32),
            ],
            // position: [pos[0] as u32, y as u32, pos[1] as u32],
            position: [18, 40, 18],
            voxel_type: 2,
            node_index: 0,
        });

        voxels_gen.push(VoxelGenerated {
            chunk_position: [
                ((position[0]) * CHUNK_X as i32),
                0,
                ((position[2]) * CHUNK_Z as i32),
            ],
            // position: [pos[0] as u32, y as u32, pos[1] as u32],
            position: [20, 33, 20],
            voxel_type: 3,
            node_index: 0,
        });

        for x in 0..CHUNK_X {
            for z in 0..CHUNK_Z {
                voxels_gen.push(VoxelGenerated {
                    chunk_position: [
                        ((position[0]) * CHUNK_X as i32),
                        0,
                        ((position[2]) * CHUNK_Z as i32),
                    ],
                    position: [x as u32, 24, z as u32],
                    voxel_type: 3,
                    node_index: 0,
                });
            }
        }
        for x in 0..CHUNK_X {
            for z in 0..CHUNK_Z {
                let pos = [
                    ((position[0] * CHUNK_X as i32 + x as i32) as f64).abs(),
                    ((position[2] * CHUNK_Z as i32 + z as i32) as f64).abs(),
                ];

                // let y = octavia_spencer(pos, 2, 0.005, 0.005, 0.0, 255.0) as usize / 2;
                // unsafe {

                let pepe =
                    PerlinNoise2D::new(6, 10.0, 0.5, 1.1, 2.0, (100.0, 100.0), 40.0, 1346139461);

                let yp = pepe.get_noise(pos[0], pos[1]);

                let y = (yp as usize).min(256);

                // for y in (0..y).rev() {
                //     let index = (z * CHUNK_X * CHUNK_Y) + (y * CHUNK_X) + x;
                //     self.chunks_mem[chunk_offset + index] = 1;
                //     voxels_gen.push(VoxelGenerated {
                //         position: [pos[0] as u32, y as u32, pos[1] as u32],
                //         voxel_type: 1,
                //         node_index: 0,
                //     })
                // }
                let index = (z * CHUNK_X * CHUNK_Y) + (y * CHUNK_X) + x;
                self.chunks_mem[chunk_offset + index] = 1;

                for y in (y - 2..y).rev() {
                    if y == 26 || y == 33 {
                        continue;
                    }
                    voxels_gen.push(VoxelGenerated {
                        chunk_position: [
                            ((position[0]) * CHUNK_X as i32),
                            0,
                            ((position[2]) * CHUNK_Z as i32),
                        ],
                        // position: [pos[0] as u32, y as u32, pos[1] as u32],
                        position: [x as u32, y as u32, z as u32],
                        voxel_type: 1,
                        node_index: 0,
                    })
                }

                // 331776 / 995326 = 0.33333400313063255
                // 40985 / 122953 = 0.33333875545940317
                // 1296 / 3886 = 0.33350488934637157

                // }
                // let y = ((position[0].abs() * position[0].abs()
                //     + position[2].abs() * position[2].abs()) as i32
                //     + 1 * 16) as usize;

                // let y = if position[0] == 0 && position[2] == 0 {
                //     50
                // } else if position[0] == -1 && position[2] == 0 {
                //     40
                // } else if position[0] == 0 && position[2] == -1 {
                //     20
                // } else {
                //     1
                // };
            }
        }

        self.root_chunks.push([
            ((position[0]) * CHUNK_X as i32),
            0,
            ((position[2]) * CHUNK_Z as i32),
            chunk_offset as i32,
        ]);

        let start_time = Instant::now();
        let bvh_voxel = bvh::bvh::BVH::build(&mut voxels_gen);

        let custom_constructor = |aabb: &AABB, entry, exit, shape_index: u32| {
            let offset = if shape_index != 4294967295 {
                voxels_gen[shape_index as usize].voxel_type
            } else {
                0
            };
            // println!("=> {} {}", entry, exit);
            VoxelBvhNode::new(aabb, entry, exit, offset)
        };
        let bvh_voxel_flatten = bvh_voxel.flatten_custom(&custom_constructor);

        // bvh_voxel.pretty_print();
        // for v in bvh_voxel_flatten {
        //     println!("{:8} {:8}", v.entry, v.exit);
        // }

        // exit(0);
        println!(
            "Offset: {} | Vox count : {} / {} - {} ms",
            chunk_offset,
            voxels_gen.len(),
            bvh_voxel_flatten.len(),
            start_time.elapsed().as_millis()
        );

        self.bvh_generated_chunks.push(ChunkGenerated {
            position: [
                (position[0] * CHUNK_X as i32) as f32,
                (position[1] * CHUNK_Y as i32) as f32,
                (position[2] * CHUNK_Z as i32) as f32,
            ],
            offset: (chunk_offset as u32),
            node_index: 0,
        });

        for (index, voxel) in bvh_voxel_flatten.iter().enumerate() {
            self.bvh_chunk_voxels[chunk_offset + index] = voxel.clone();
        }

        // self.bvh_chunk_voxels.cop
        // self.generated_chunks_voxs
        //     .insert([position[0], position[1], position[2]], yo);

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

        let bvh_chunk = bvh::bvh::BVH::build(&mut self.bvh_generated_chunks);

        let custom_constructor = |aabb: &AABB, entry, exit, shape_index| {
            let offset = if shape_index != 4294967295 {
                self.bvh_generated_chunks[shape_index as usize].offset
            } else {
                0
            };

            GpuBvhNode::new(aabb, entry, exit, offset)
        };

        self.bvh_chunks = bvh_chunk.flatten_custom(&custom_constructor);

        // for (index, node) in self.bvh_chunk_voxels.iter().enumerate() {
        //     if index > 16 {
        //         break;
        //     }
        //     // if node.exit == 0 {
        //     //     continue;
        //     // }

        //     let position_index_min = node.aabbmin_n_type >> 13 & 524287;
        //     let position_index_max = node.aabbmax >> 13 & 524287;

        //     let posmax = [
        //         (position_index_max % (CHUNK_X as u32)),
        //         ((position_index_max / (CHUNK_X as u32)) % (CHUNK_Y as u32)),
        //         (position_index_max / ((CHUNK_X as u32) * (CHUNK_Y as u32))),
        //     ];
        //     let posmin = [
        //         (position_index_min % (CHUNK_X as u32)),
        //         ((position_index_min / (CHUNK_X as u32)) % (CHUNK_Y as u32)),
        //         (position_index_min / ((CHUNK_X as u32) * (CHUNK_Y as u32))),
        //     ];

        //     println!(
        //         "{:5} - ({:11}) ({:11}) - {:1} {:8} {:8}  : {:?} {:?}",
        //         index,
        //         node.entry,
        //         node.exit,
        //         (node.entry >> 11) & 1,
        //         (node.entry >> 12) & 1048575,
        //         (node.exit >> 12) & 1048575,
        //         posmin,
        //         posmax,
        //     );
        // }

        // for (index, node) in self.bvh_chunks.iter().enumerate() {
        //     println!(
        //         "{:5} - {:11} {:11} | {:11} | {:?} {:?}",
        //         index, node.entry, node.exit, node.offset, node.min, node.max
        //     );
        // }
        println!("{}", self.generated_chunks.len());
        println!("{}", self.bvh_chunks.len());
        println!("{}", self.bvh_chunk_voxels.len());
        println!(
            "{}",
            self.bvh_chunk_voxels.len() * std::mem::size_of::<GpuBvhNode>()
        );
        println!("{}", self.bvh_chunk_voxels.len() * 16);

        // println!(
        //     "{} {} ",
        //     self.bvh_generated_chunks.len(),
        //     self.bvh_chunks.len() * std::mem::size_of::<GpuBvhNode>()
        // );

        // bvh_chunk.pretty_print();

        // exit(0);

        // self.generated_chunks_gpubvh.clear();
        // self.root_grid.resize(30 * 30, [0; 4]);

        // for x in 0..30 {
        //     for z in 0..30 {
        //         let chunk_pos = [15 - (x), 0, 15 - (z), 0];

        //         let chk = self.generated_chunks.get(&chunk_pos);
        //         self.root_grid[(x + z * 30) as usize] = [0, 0, 0, 0];

        //         if chk.is_none() {
        //             print!("       | ");
        //         } else {
        //             // print!("{:5} | ", self.root_grid[x + z * 20]);
        //             let val = chk.unwrap();
        //             // print!("{:8} | ", val * CHUNK_TSIZE + 1);
        //             print!("{:2}  {:2} | ", chunk_pos[0], chunk_pos[2]);
        //             self.root_grid[(x + z * 30) as usize] = [
        //                 (15 - (x)) * CHUNK_X as i32,
        //                 0,
        //                 (15 - (z)) * CHUNK_Z as i32,
        //                 (*val as i32) * CHUNK_TSIZE as i32 + 1,
        //             ];
        //         }
        //     }
        //     println!("");
        // }
        // println!("-> {}", self.root_chunks.len() * 16);
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

            let chunk_offset = self.bvh_chunk_voxels.len();
            self.bvh_chunk_voxels.resize(
                chunk_offset + CHUNK_MEM_OFFSET,
                VoxelBvhNode {
                    entry: 0,
                    exit: 0,
                    aabbmin_n_type: 0,
                    aabbmax: 0,
                },
                // GpuBvhNode {
                //     min: [0.0, 0.0, 0.0, 0.0],
                //     max: [0.0, 0.0, 0.0, 0.0],
                //     entry: 0,
                //     exit: 0,
                //     offset: 0,
                //     _padding: 0,
                // },
            );

            // println!(
            //     "Empty, new size : {} -> {} ",
            //     chunk_offset,
            //     chunk_offset + CHUNK_TSIZE
            // );
            return chunk_offset;
        }
        let free_zone = self.chunks_mem_free.pop().unwrap() * CHUNK_MEM_OFFSET;

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
