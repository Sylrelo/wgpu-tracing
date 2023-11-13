// STRUCTURES ================================================

struct Settings {
    position: vec4<f32>,
    chunk_content_count: u32,
    root_chunk_count: u32,
}

struct Ray {
    orig: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
}

struct GpuBvhNode {
    min: vec4<f32>,
    max: vec4<f32>,
    entry: u32,
    exit: u32,
    offset: u32,
    _padding: u32,
}

struct VoxelBvhNode {
    entry: u32,
    exit: u32,
    aabbmin_n_type: u32,
    aabbmax: u32,
}


// DATA ======================================================

@group(0) @binding(0)
var<uniform> settings: Settings;

@group(1) @binding(0)
var<storage> chunk_content: array<u32>;

@group(1) @binding(1)
var<storage> bvh_root_chunks: array<GpuBvhNode>;

@group(1) @binding(2)
var<storage> bvh_voxels_chunk: array<VoxelBvhNode>;

@group(1) @binding(3)
var<storage> root_chunks: array<vec4<i32>>;

@group(2) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(1)
var normal_output: texture_storage_2d<rgba8snorm, write>;

@group(2) @binding(2)
var color_output2: texture_storage_2d<rgba8unorm, write>;

// CONSTANTS =================================================

const M_PI = 3.1415926535897932384626433832795;
const M_TWOPI = 6.28318530718;
const F32_MAX = 3.402823E+38;
const CHUNK_XMAX = 36;
const CHUNK_YMAX = 256;
const CHUNK_ZMAX = 36;
// const CHUNK_TSIZE = CHUNK_XMAX * CHUNK_YMAX * CHUNK_ZMAX;
const CHUNK_TSIZE = 331776;

// UTILITIES ==================================================

// PATHTRACING UTILITIES =====================================

fn wang_hash(seed: ptr<function, u32>) -> u32 {
    (*seed) = (*seed ^ 61u) ^ (*seed >> 16u);
    (*seed) *= 9u;
    (*seed) = *seed ^ ((*seed) >> 4u);
    (*seed) *= u32(0x27d4eb2d);
    (*seed) = *seed ^ ((*seed) >> 15u);

    return *seed;
}
 
fn random_float_01(seed: ptr<function, u32>) -> f32 {
    return f32(wang_hash(seed)) / 4294967296.0;
}
 
fn random_unit_vector(seed: ptr<function, u32>) -> vec3<f32> {
    let z = random_float_01(seed) * 2.0 - 1.0;
    let a = random_float_01(seed) * M_TWOPI;
    let r = sqrt(1.0f - z * z);
    let x = r * cos(a);
    let y = r * sin(a);

    return (vec3(x, y, z));
}

fn rand3_on_sphere(seed: ptr<function, u32>) -> vec3<f32> {
    let t = M_PI * random_float_01(seed);
    let z = random_float_01(seed);
    let r = sqrt((z + 1.0) * (1.0 - z));
    let x = cos(t) * r;
    let y = sin(t) * r;

    return vec3(x, y, z);
}

fn rand2_in_circle(seed: ptr<function, u32>) -> vec2<f32> {
    let t = M_PI * random_float_01(seed);
    let r = sqrt((random_float_01(seed) + 1.0) / 2.0);

    return r * vec2(cos(t), sin(t));
}

fn ortho(v: vec3<f32>) -> vec3<f32> {
    if abs(v.x) > abs(v.z) {
        return vec3(-v.y, v.x, 0.0);
    } else {
        return  vec3(0.0, -v.z, v.y);
    }
}

fn sample_cone(seed: ptr<function, u32>, dir_in: vec3<f32>, extent: f32) -> vec3<f32> {
    let dir = normalize(dir_in);
    let o1 = normalize(ortho(dir));
    let o2 = normalize(cross(dir, o1));
    var r = vec2(random_float_01(seed), random_float_01(seed));
    r.x = r.x * 2.0 * M_PI;
    r.y = 1.0 - r.y * extent;
    let oneminus = sqrt(1.0 - r.y * r.y);
    return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}


fn getCosineWeightedSample(seed: ptr<function, u32>, dir: vec3<f32>) -> vec3<f32> {
    let o1 = normalize(ortho(dir));
    let o2 = normalize(cross(dir, o1));
    var r = vec2(random_float_01(seed), random_float_01(seed));
    r.x = r.x * 2.0 * M_PI;
    r.y = pow(r.y, 0.5);
    let oneminus = sqrt(1.0 - r.y * r.y);

    return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}

// ===========================================================

fn vec_rot_x(in: vec3<f32>, rad: f32) -> vec3<f32> {

    return vec3(
        in.x,
        in.y * cos(rad) + in.z * -sin(rad),
        in.y * sin(rad) + in.z * cos(rad),
    );
}

fn vec_rot_y(in: vec3<f32>, rad: f32) -> vec3<f32> {
    var	n = vec3<f32>(0.0);


    n.x = in.x * cos(rad) + in.z * sin(rad);
    n.y = in.y;
    n.z = in.x * -sin(rad) + in.z * cos(rad);
    return (n);
}

fn vec_rot_z(in: vec3<f32>, rad: f32) -> vec3<f32> {
    var	n = vec3<f32>(0.0);

    n.x = in.x * cos(rad) + in.y * -sin(rad);
    n.y = in.x * sin(rad) + in.y * cos(rad);
    n.z = in.z;

    return (n);
}

// TEST ======================================================

struct BvhVoxelHit {
    material: u32,
    dist: f32,
    normal: vec3<f32>,
    point: vec3<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
}


fn intersect_aabb(ray: Ray, min: vec3<f32>, max: vec3<f32>) -> f32 {

    let is_inside_v = step(min, ray.orig) - step(max, ray.orig);
    let is_inside_t = is_inside_v.x * is_inside_v.y * is_inside_v.z;
    if is_inside_t > 0.0 {
        return is_inside_t;
    }

    let t0s = (min - ray.orig) * ray.inv_dir;
    let t1s = (max - ray.orig) * ray.inv_dir;

    let tsmaller = min(t0s, t1s);
    let tbigger = max(t0s, t1s);

    let tmin = max(tsmaller[0], max(tsmaller[1], tsmaller[2]));
    let tmax = min(tbigger[0], min(tbigger[1], tbigger[2]));

    // let t = min(tmin, tmax);

    if tmin < tmax {
        return tmin;
    }

    // if t > 0.0 {
    //     return t;
    // }
    return 0.0;
}

fn aabb_intersect(ray_in: Ray, min: vec3<f32>, max: vec3<f32>) {
}

fn normal_cube(
    ray_position: vec3<f32>,
    pos: vec3<f32>,
    min: vec3<f32>,
    max: vec3<f32>
) -> vec3<f32> {
    let epsilon = 0.00001;


    // return vec3(0.0);
    // let box_center = (max - min) * 0.5 + min;
    // let popos = (ray_position) - box_center;
    // let aled = popos / max(max(abs(popos.x), abs(popos.y)), abs(popos.z));
    // let box_normal = clamp(aled, vec3(0.0), vec3(1.0));
    // let norm = normalize(floor(box_normal * 1.000001));

    // return norm;
    // let c = (min + max) * 0.5;
    // let p = ray_position - c;
    // let d = (min - max) * 0.5;

    // return normalize(vec3<f32>(floor(p / abs(d) * epsilon)));


    // let center = 0.5 * (min + max);
    // let pc = ray_position - center;
    // let size = 0.5 * (max - min);

    // return normalize(sign(centerToPoint) * step(vec3(-epsilon), abs(centerToPoint) - halfSize));

    // var normal = vec3(0.0);
    // normal += vec3(sign(pc.x), 0.0, 0.0) * step(abs(abs(pc.x) - size.x), epsilon);
    // normal += vec3(0.0, sign(pc.y), 0.0) * step(abs(abs(pc.y) - size.y), epsilon);
    // normal += vec3(0.0, 0.0, sign(pc.z)) * step(abs(abs(pc.z) - size.z), epsilon);

    // return normalize(normal);

    let d = ray_position - (min + 0.5);
    let dabs = abs(d);
    if dabs.x > dabs.y {
        if dabs.x > dabs.z {
            return vec3(sign(d.x), 0.0, 0.0);
        } else {
            return vec3(0.0, 0.0, sign(d.z));
        }
    } else {
        if dabs.y > dabs.z {
            return vec3(0.0, sign(d.y), 0.0);
        } else {
            return vec3(0.0, 0.0, sign(d.z));
        }
    }


    // let bmin = min;
    // let bmax = max;

    // let cx = abs(ray_position.x - bmin.x);
    // let fx = abs(ray_position.x - bmax.x);

    // let cy = abs(ray_position.y - bmin.y);
    // let fy = abs(ray_position.y - bmax.y);

    // let cz = abs(ray_position.z - bmin.z);
    // let fz = abs(ray_position.z - bmax.z);



    // if ray_position.x < bmin.x + epsilon {
    //     return vec3(-1.0, 0.0, 0.0);
    // } else if ray_position.x > bmax.x - epsilon {
    //     return vec3(1.0, 0.0, 0.0);
    // } else if ray_position.y < bmin.y + epsilon {
    //     return vec3(0.0, -1.0, 0.0);
    // } else if ray_position.y > bmax.y - epsilon {
    //     return vec3(0.0, 1.0, 0.0);
    // } else if ray_position.z < bmin.z + epsilon {
    //     return vec3(0.0, 0.0, -1.0);
    // } else if ray_position.z > bmax.z - epsilon {
    //     return vec3(0.0, 0.0, 1.0);
    // }

    //  if cx <= epsilon {
    //     return vec3(-1.0, 0.0, 0.0);
    // } else if fx <= epsilon {
    //     return vec3(1.0, 0.0, 0.0);
    // } else if cy <= epsilon {
    //     return vec3(0.0, -1.0, 0.0);
    // } else if fy <= epsilon {
    //     return vec3(0.0, 1.0, 0.0);
    // } else if cz <= epsilon {
    //     return vec3(0.0, 0.0, -1.0);
    // } else if fz <= epsilon {
    //     return vec3(0.0, 0.0, 1.0);
    // }

    // return vec3(0.0, 0.0, 0.0);
}

// RAYTRACING =================================================

fn precalc_ray(ray: ptr<function, Ray>) {
    // (*ray).sign_x = u32((*ray).dir.x < 0.0);
    // (*ray).sign_y = u32((*ray).dir.y < 0.0);
    // (*ray).sign_z = u32((*ray).dir.z < 0.0);
    (*ray).inv_dir = 1.0 / (*ray).dir;
}

const MEM_SIZE: u32 = 1000000u;

fn traverse_voxels(ray_in: Ray, chunk_position: vec3<f32>, offset: u32) -> BvhVoxelHit {

    var hit: BvhVoxelHit = BvhVoxelHit(0u, F32_MAX, vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    var node_idx = offset;
    var prev_t = F32_MAX;
    // var t_dist = F32_MAX;

    var aabb_min = vec3(0.0);
    var aabb_max = vec3(0.0);

    while node_idx < (offset + MEM_SIZE) && node_idx < arrayLength(&bvh_voxels_chunk) { // offset + max_size
        let node = bvh_voxels_chunk[node_idx];

        if node.exit == 0u {
            break;
        }

        if ((node.entry >> 11u) & 1u) == 1u {
            node_idx = offset + ((node.exit >> 12u) & 1048575u);

            if prev_t < hit.dist {
                hit.dist = prev_t;
                hit.material = ((node.entry >> 2u) & 511u);
                hit.aabb_min = aabb_min;
                hit.aabb_max = aabb_max;
            }
            continue;
        }

        let posmin = vec3(
            f32((node.aabbmin_n_type >> 26u) & 63u),
            f32((node.aabbmin_n_type >> 11u) & 511u),
            f32((node.aabbmax >> 26u) & 63u)
        );

        let posmax = vec3(
            f32((node.aabbmin_n_type >> 20u) & 63u),
            f32((node.aabbmin_n_type >> 2u) & 511u),
            f32((node.aabbmax >> 20u) & 63u)
        );

        let t = intersect_aabb(
            ray_in,
            chunk_position + posmin,
            chunk_position + posmax,
        );

        if t > 0.0 {
            node_idx = offset + ((node.entry >> 12u) & 1048575u);
            aabb_min = chunk_position + posmin;
            aabb_max = chunk_position + posmax;
            prev_t = t;
        } else {
            node_idx = offset + ((node.exit >> 12u) & 1048575u);
        }
    }

    return hit;
}


// TEST AGAIN
struct DataDda {
    map: vec3<i32>,
    max: vec3<f32>,
    step_amount: vec3<i32>,
    delta: vec3<f32>,
    side: i32,
    hit_data: u32,
    t: f32,
    mask: vec3<bool>,
}


fn dda_prepare_scratch(
    ray_in: Ray,
    grid_min: vec3<f32>,
    grid_max: vec3<f32>,
    aabb_tmin: f32,
) -> DataDda {
    var dda: DataDda;
    let resolution = vec3(36, 256, 36);
    let cell_dimensions = vec3(1.0, 1.0, 1.0);

    let ray_orig_cell = (ray_in.orig + ray_in.dir * aabb_tmin) - grid_min;
    dda.map = clamp(vec3<i32>(floor(ray_orig_cell / cell_dimensions)), vec3(0), resolution);


    if ray_in.dir.x < 0.0 {
        dda.step_amount.x = -1;
        dda.delta.x = -cell_dimensions.x * ray_in.inv_dir.x;
        dda.max.x = aabb_tmin + (f32(dda.map.x) * cell_dimensions.x - ray_orig_cell.x) * ray_in.inv_dir.x;
    } else if ray_in.dir.x > 0.0 {
        dda.step_amount.x = 1;
        dda.delta.x = cell_dimensions.x * ray_in.inv_dir.x;
        dda.max.x = aabb_tmin + (f32(dda.map.x + 1) * cell_dimensions.x - ray_orig_cell.x) * ray_in.inv_dir.x;
    }

    if ray_in.dir.y < 0.0 {
        dda.step_amount.y = -1;
        dda.delta.y = -cell_dimensions.y * ray_in.inv_dir.y;
        dda.max.y = aabb_tmin + (f32(dda.map.y) * cell_dimensions.y - ray_orig_cell.y) * ray_in.inv_dir.y;
    } else if ray_in.dir.y > 0.0 {
        dda.step_amount.y = 1;
        dda.delta.y = cell_dimensions.y * ray_in.inv_dir.y;
        dda.max.y = aabb_tmin + (f32(dda.map.y + 1) * cell_dimensions.y - ray_orig_cell.y) * ray_in.inv_dir.y;
    }

    if ray_in.dir.z < 0.0 {
        dda.step_amount.z = -1;
        dda.delta.z = -cell_dimensions.z * ray_in.inv_dir.z;
        dda.max.z = aabb_tmin + (f32(dda.map.z) * cell_dimensions.z - ray_orig_cell.z) * ray_in.inv_dir.z;
    } else if ray_in.dir.z > 0.0 {
        dda.step_amount.z = 1;
        dda.delta.z = cell_dimensions.z * ray_in.inv_dir.z;
        dda.max.z = aabb_tmin + (f32(dda.map.z + 1) * cell_dimensions.z - ray_orig_cell.z) * ray_in.inv_dir.z;
    }

    // for (var i = 0; i < 3; i++) {
    //     let ray_orig_cell = (ray_in.orig[i] + ray_in.dir[i] * aabb_tmin) - grid_min[i];
    //     dda.map[i] = clamp(i32(floor(ray_orig_cell / cell_dimensions[i])), 0, resolution[i]);

    //     if ray_in.dir[i] < 0.0 {
    //         dda.step_amount[i] = -1;
    //         dda.delta[i] = -cell_dimensions[i] * ray_in.inv_dir[i];
    //         dda.max[i] = aabb_tmin + (f32(dda.map[i]) * cell_dimensions[i] - ray_orig_cell) * ray_in.inv_dir[i];
    //     } else {
    //         dda.step_amount[i] = 1;
    //         dda.delta[i] = cell_dimensions[i] * ray_in.inv_dir[i];
    //         dda.max[i] = aabb_tmin + (f32(dda.map[i] + 1) * cell_dimensions[i] - ray_orig_cell) * ray_in.inv_dir[i];
    //     }
    // }

    return dda;
}




fn dda_prepare(ray: Ray, cell_size: vec3<f32>, min_bound: vec3<f32>) -> DataDda {
    var dda: DataDda;

    dda.map = vec3<i32>((ray.orig - min_bound) / cell_size);
    dda.delta = vec3(abs(ray.inv_dir) * cell_size);
    dda.step_amount = vec3(0);
    dda.max = vec3(0.0);

    if ray.dir.x < 0.0 {
        dda.step_amount.x = -1;
        dda.max.x = (min_bound.x + (f32(dda.map.x) * cell_size.x) - ray.orig.x) * ray.inv_dir.x ;
    } else if ray.dir.x > 0.0 {
        dda.step_amount.x = 1;
        dda.max.x = (min_bound.x + (f32(dda.map.x + 1) * cell_size.x) - ray.orig.x) * ray.inv_dir.x;
    }

    if ray.dir.y < 0.0 {
        dda.step_amount.y = -1;
        dda.max.y = (min_bound.y + (f32(dda.map.y) * cell_size.y) - ray.orig.y) * ray.inv_dir.y;
    } else if ray.dir.y > 0.0 {
        dda.step_amount.y = 1;
        dda.max.y = (min_bound.y + (f32(dda.map.y + 1) * cell_size.y) - ray.orig.y) * ray.inv_dir.y;
    }

    if ray.dir.z < 0.0 {
        dda.step_amount.z = -1;
        dda.max.z = (min_bound.z + (f32(dda.map.z) * cell_size.z) - ray.orig.z) * ray.inv_dir.z;
    } else if ray.dir.z > 0.0 {
        dda.step_amount.z = 1;
        dda.max.z = (min_bound.z + (f32(dda.map.z + 1) * cell_size.z) - ray.orig.z) * ray.inv_dir.z;
    }

    return dda;
}

fn dda_steps(ray: Ray, dda: ptr<function, DataDda>) {

 
    // if (*dda).max.x < (*dda).max.y && (*dda).max.x < (*dda).max.z {
    // //     (*dda).map.x += (*dda).step_amount.x;
    // //     (*dda).max.x += (*dda).delta.x;
    //     (*dda).side = 0;
    // //     // (*dda).t = (f32((*dda).map.x) - ray.orig.x + f32(1 - (*dda).step_amount.x) * 0.5) * ray.inv_dir.x;
    // } else if (*dda).max.y < (*dda).max.z {
    // //     (*dda).map.y += (*dda).step_amount.y;
    // //     (*dda).max.y += (*dda).delta.y;
    //     (*dda).side = 2;
    // //     // (*dda).t = (f32((*dda).map.y) - ray.orig.y + f32(1 - (*dda).step_amount.y) * 0.5) * ray.inv_dir.y;
    // } else {
    // //     (*dda).map.z += (*dda).step_amount.z;
    // //     (*dda).max.z += (*dda).delta.z;
    //     (*dda).side = 1;
    // //     // (*dda).t = (f32((*dda).map.z) - ray.orig.z + f32(1 - (*dda).step_amount.z) * 0.5) * ray.inv_dir.z;
    // }


    let mask = ((*dda).max.xyz <= min((*dda).max.yzx, (*dda).max.zxy));

    (*dda).map += vec3<i32>(mask) * (*dda).step_amount;
    (*dda).max += vec3<f32>(mask) * (*dda).delta;

    let tmp_side = vec3<i32>(mask) * vec3(1, 3, 2);
    (*dda).side = max(tmp_side.x, max(tmp_side.y, tmp_side.z)) - 1;

    (*dda).mask = mask;
}


fn get_normal(side: i32, delta: vec3<i32>) -> vec3<f32> {

    if side == 0 && delta.x > 0 {
        return vec3(-1.0, 0.0, 0.0);
    } else if side == 0 && delta.x < 0 {
        return vec3(1.0, 0.0, 0.0);
    }

    if side == 1 && delta.z < 0 {
        return vec3(0.0, 0.0, 1.0);
    } else if side == 1 && delta.z > 0 {
        return vec3(0.0, 0.0, -1.0);
    }

    if delta.y > 0 {
        return vec3(0.0, -1.0, 0.0);
    } else {
        return vec3(0.0, 1.0, 0.0);
    }
}

    // ray_in: Ray,
    // grid_min: vec3<f32>,
    // grid_max: vec3<f32>,
    // grid_resolution: vec3<f32>,
    // aabb_tmin: f32,

fn traver_voxel_ug(
    ray_in: Ray,
    chunk_offset: u32,
    min_bound: vec3<f32>,
    max_bound: vec3<f32>,
    dist: f32,
) -> DataDda {

    var dda: DataDda = dda_prepare_scratch(
        ray_in,
        min_bound,
        max_bound,
        dist
    );
    // var dda: DataDda = dda_prepare(ray_in, vec3(1.0), min_bound);
    var iter = 0u;

    let len = settings.chunk_content_count;

    if chunk_offset >= len || len <= 0u {
        return dda;
    }

    while iter < 80u && dda.hit_data == 0u {
        iter++;
        dda_steps(ray_in, &dda);

        let index = i32(chunk_offset) + ((dda.map.z * CHUNK_XMAX * CHUNK_YMAX) + (dda.map.y * CHUNK_XMAX) + dda.map.x);

        if dda.map.x < 0 || dda.map.x >= CHUNK_XMAX || dda.map.y < 0 || dda.map.y >= CHUNK_YMAX || dda.map.z < 0 || dda.map.z >= CHUNK_ZMAX || index >= i32(len) || index < 0 {
            continue;
        }

        dda.hit_data = chunk_content[index];
    }



    if dda.hit_data != 0u {
        let hit_t = intersect_aabb(
            ray_in,
            min_bound + vec3(f32(dda.map.x), f32(dda.map.y), f32(dda.map.z)),
            min_bound + vec3(f32(dda.map.x) + 1.0, f32(dda.map.y) + 1.0, f32(dda.map.z) + 1.0),
        );
        dda.t = hit_t;
    }
    return dda;
}

// 

fn bvh_traverse_chunks(ray_in: Ray) -> BvhVoxelHit {
    var hit: BvhVoxelHit = BvhVoxelHit(0u, F32_MAX, vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

    var node_idx = 0u;
    var prev_t = F32_MAX;
    var prev_node: GpuBvhNode;

    var last_t = F32_MAX;

    // if true {
    //     var prev_t = F32_MAX;
    //     var tmp_hit: DataDda;

    //     for (var i = 0u; i < 36u; i++) {
    //         let curr = root_chunks[i];

    //         let t = intersect_aabb(
    //             ray_in,
    //             vec3<f32>(curr.xyz),
    //             vec3<f32>(curr.xyz) + vec3(36.0, 256.0, 36.0),
    //         );

    //         if t > 0.0  {
    //             // hit.dist = t;
    //             // hit.material = 1u;
    //             // prev_t = t;


    //             tmp_hit = traver_voxel_ug(
    //                 ray_in,
    //                 u32(curr.w),
    //                 vec3<f32>(curr.xyz),
    //                 vec3<f32>(curr.xyz) + vec3(36.0, 256.0, 36.0),
    //                 t
    //             );
    //             if tmp_hit.hit_data != 0u && tmp_hit.t > 0.0 && tmp_hit.t < hit.dist {
    //                 hit.dist = tmp_hit.t;
    //                 hit.material = tmp_hit.hit_data;
    //                 // hit.normal = get_normal(tmp_hit.side, tmp_hit.step_amount);
    //                 hit.normal = vec3<f32>(tmp_hit.mask) * -sign(vec3<f32>(tmp_hit.step_amount));
    //             }
    //         }
    //     }

    //     return hit;
    // }

    while node_idx < settings.root_chunk_count {
        let node = bvh_root_chunks[node_idx];

        if node.entry == 4294967295u {
            node_idx = node.exit;


            if prev_t < last_t {
                last_t = prev_t;
            }

            // var vox_ray = ray_in;
            // vox_ray.orig -= prev_node.min.xyz;

            // let test_t = intersect_aabb(
            //     vox_ray,
            //     vec3(0.0),
            //     vec3(36.0, 256.0, 36.0),
            // );

            // vox_ray.orig += ray_in.dir * prev_t;

            let tmp_hit = traver_voxel_ug(
                ray_in,
                node.offset,
                prev_node.min.xyz,
                prev_node.max.xyz,
                prev_t
            );
            // let tmp_hit = traver_voxel_ug(
            //     vox_ray,
            //     prev_node.min.xyz,
            //     node.offset,
            //     prev_t
            // );

            if tmp_hit.hit_data != 0u && tmp_hit.t < hit.dist {
                hit.dist = tmp_hit.t;
                hit.material = tmp_hit.hit_data;
                hit.normal = vec3<f32>(tmp_hit.mask) * -sign(vec3<f32>(tmp_hit.step_amount));
                // hit.normal = tmp_hit.normal_test;
                // hit.normal = get_normal(tmp_hit.side, tmp_hit.step_amount);
            }

            // last_t = prev_t;

            // let tmp_hit = traverse_voxels(ray_in, prev_node.min.xyz, node.offset);
            // if tmp_hit.dist < hit.dist {
            //     hit = tmp_hit;
            // }
            continue;
        }

        let t = intersect_aabb(
            ray_in,
            node.min.xyz,
            node.max.xyz,
        );

        if t > 0.0 {
            node_idx = node.entry;
            prev_node = node;
            prev_t = t;
        } else {
            node_idx = node.exit;
        }
    }


    // hit.dist = F32_MAX;
    // hit.material = 0u;

    // var vox_ray = ray_in;
    //         // vox_ray.orig -= prev_node.min.xyz;
    // let min_bound = vec3(0.0, 0.0, 0.0) * vec3(36.0, 256.0, 36.0);

    // vox_ray.orig += min_bound;
    // vox_ray.orig = vox_ray.orig + ray_in.dir * 9.5;
    // let tmp_hit = traver_voxel_ug(
    //     vox_ray,
    //     vec3(0.0),
    //     0u,
    // );

    // if tmp_hit != F32_MAX {
    //     hit.dist = tmp_hit;

    //     hit.material = 1u;
    // }

    return hit;
}

fn fresnel(cosEN: f32, in: vec3<f32>) -> vec3<f32> {
    let e = 1.0 - cosEN;
    var e5 = e * e;
    e5 *= e5 * e;
    return (1.0 - e5) * in + e5;
}

fn sample_sunlight(seed: ptr<function, u32>, hit_point: vec3<f32>, hit_normal: vec3<f32>) -> vec3<f32> {
    let sun_position = vec3(200.0, 600.0, -500.0);
    let light_vec = sun_position - hit_point;
    let light_dir = normalize(light_vec);
    let dst = length(light_vec);

    let theta = asin(150.0 / dst);

    var shadow_ray: Ray;
    shadow_ray.dir = sample_cone(seed, light_dir, theta);
    shadow_ray.orig = hit_point + hit_normal * 0.001;
    precalc_ray(&shadow_ray);

    let inv_prob = 2.0 * (1.0 - cos(theta)) * 50.0;
    let light_val = clamp(dot(hit_normal, light_dir), 0.0, 1.0);

    let shadow_hit = bvh_traverse_chunks(shadow_ray);

    if shadow_hit.material == 0u {
        return vec3(1.0) * light_val * inv_prob;
    }

    return vec3(0.0);
}

fn brdf(
    seed: ptr<function, u32>,
    normal: vec3<f32>,
    ray_dir: vec3<f32>,
    material: u32,
    specular_bounce: ptr<function, bool>,
) -> vec3<f32> {
    *specular_bounce = false;

    if material == 3u {
        *specular_bounce = true;
        // return (reflect(ray_dir, normal));
        return normalize(reflect(ray_dir, -normal) + getCosineWeightedSample(seed, normal) * 0.1);
    } else {
        return getCosineWeightedSample(seed, normal);
    }
}

fn pathtrace(ray_in: Ray, seed: ptr<function, u32>, screen_pos: vec2<i32>) -> vec3<f32> {
    var throughput: vec3<f32> = vec3(1.0, 1.0, 1.0);
    var ray = ray_in;
    var specular_bounce = true;
    var color: vec3<f32> = vec3(0.0, 0.0, 0.0);

    for (var i = 0; i < 3; i++) {
        var voxel_hit = bvh_traverse_chunks(ray);

        if voxel_hit.material == 0u || voxel_hit.dist == F32_MAX {
            color += (vec3(161.0 / 255.0, 247.0 / 255.0, 1.0) * 0.0) * throughput;
            break;
        }

        voxel_hit.point = ray_in.orig + ray_in.dir * voxel_hit.dist;
        // voxel_hit.normal = normal_cube(voxel_hit.point, vec3(0.0), voxel_hit.aabb_min, voxel_hit.aabb_max);

        if i == 0 {
            textureStore(normal_output, screen_pos, voxel_hit.normal.xyzz);
        }

        var vox_color = voxel_hit.normal * 0.5 + 0.5;

        // if i == 1 {
        //     return vox_color;
        // }
        // throughput *= vox_color;

        // vox_color = vec3(0.1, 0.3, 0.6);

        ray.orig = voxel_hit.point + voxel_hit.normal * 0.0001;

        if voxel_hit.material == 2u {
            color += throughput * vec3(1.0, 1.0, 1.0) ;
            break;
        }

        ray.dir = brdf(seed, voxel_hit.normal, ray.dir, voxel_hit.material, &specular_bounce);

        if !specular_bounce || dot(ray.dir, voxel_hit.normal) < 0.0 {
            throughput *= vox_color;
        }

        if !specular_bounce {
            color += throughput * sample_sunlight(seed, voxel_hit.point, voxel_hit.normal);
        }
        // if voxel_hit.material == 1u {
        //     throughput *= vox_color;
        //     // ray.dir = normalize(random_unit_vector(seed) + voxel_hit.normal);
        //     ray.dir = getCosineWeightedSample(seed, voxel_hit.normal);

        //     color += throughput * sample_sunlight(seed, voxel_hit.point, voxel_hit.normal);
        // }

        // if voxel_hit.material == 3u {
        //     throughput *= vec3(1.0, 1.0, 1.0);
        //     ray.dir = normalize(reflect(ray.dir, voxel_hit.normal) + getCosineWeightedSample(seed, voxel_hit.normal) * 0.02);
        // }


        precalc_ray(&ray);
    }

    return color;
    // return throughput;
}

fn raytrace(ray_in: Ray) -> vec3<f32> {
    var hit = bvh_traverse_chunks(ray_in);

    // hit = traverse_voxels(ray_in, vec3(0.0), 0u);

    if hit.material != 0u {
        hit.point = ray_in.orig + ray_in.dir * hit.dist;

        // hit.normal = normal_cube(hit.point, vec3(0.0), hit.aabb_min, hit.aabb_max);
        // return vec3(hit.dist / 300.0, 0.4, 0.4);
        return vec3(hit.normal * 0.5 + 0.5);
        // return vec3(hit.dist / 300.0);
        // return vec3(hit.dist / 300.0);
    }

    // if prev_t != F32_MAX {
    //     return vec3(iter / 1500.0, 0.0, 0.0);
    // }

    return vec3(
        0.02,
        0.00,
        0.00
    );
}


// ===========================================================
@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_size: vec2<u32> = vec2<u32>(textureDimensions(color_output));
    let screen_pos: vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let aspect_ratio = f32(screen_size.x) / f32(screen_size.y);

    let tatan = tan(1.5708 / 2.0);
    let ndc_pixel = vec2(
        (f32(screen_pos.x) + 0.5) / f32(screen_size.x),
        (f32(screen_pos.y) + 0.5) / f32(screen_size.y),
    );
    let ndc_pos = vec2<f32>(
        (2.0 * ndc_pixel.x - 1.0 * tatan) * aspect_ratio,
        1.0 - 2.0 * ndc_pixel.y * tatan
    );
    var ray_direction = normalize(vec3(ndc_pos.xy, -1.0));

    var ray: Ray = Ray(
        settings.position.xyz,
        ray_direction,
        1.0 / ray_direction
    );

    var final_color = vec3<f32>(0.0, 0.0, 0.0);
    var seed: u32 = (u32(screen_pos.x) * (1973u) + u32(screen_pos.y) * (9277u) * (26699u)) | (1u);

    let MAX_SAMPLES = 1;

    ray.dir = vec_rot_x(ray.dir, -0.65);
    // ray.dir = vec_rot_y(ray.dir, -1.9);
    precalc_ray(&ray);

    for (var i = 0; i < MAX_SAMPLES; i++) {
        // seed = (1973u * 9277u + u32(i) * 26699u) | (1u);
        seed = (u32(screen_pos.x) * 1973u + u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);
        // seed = (u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);
        // wang_hash(&seed);


        // let foc_target = ray.orig + ray.dir * 160.0;
        // let defocus = 0.0002 * rand2_in_circle(&seed);

        // ray.orig += vec3(defocus.xy, 0.0);
        // ray.dir = normalize(foc_target - ray.orig);

        final_color += pathtrace(ray, &seed, screen_pos);
    }
    final_color = (final_color / f32(MAX_SAMPLES));

    let gamma = 1.6;
    let exposure = 1.0;

    var tone_mapping = vec3(1.0) - exp(-final_color * gamma);
    tone_mapping = pow(tone_mapping, vec3(1.0 / exposure));

    textureStore(color_output, screen_pos, vec4(tone_mapping.xyz, 1.0));
    textureStore(color_output2, screen_pos, vec4(tone_mapping.xyz, 1.0));
    // textureStore(color_output, screen_pos, vec4(raytrace(ray).xyz, 1.0));
}