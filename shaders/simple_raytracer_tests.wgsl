// STRUCTURES ================================================

struct Settings {
    position: vec4<f32>,
    chunk_content_count: u32,
    bvh_node_count: u32,
}

struct Ray {
    orig: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
}

struct HitData {
    has_hit: bool,
    dist: f32,
    normal: vec3<f32>,
}

struct ChunkBvhNode {
    min: vec4<f32>,
    max: vec4<f32>,
    data: vec4<u32>,
    _padding: vec4<u32>,
}

// DATA ======================================================

@group(0) @binding(0)
var<uniform> settings: Settings;

@group(1) @binding(0)
var<storage> chunk_content: array<u32>;

@group(1) @binding(1)
var<storage> chunks: array<vec4<f32>>;

@group(2) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

// CONSTANTS =================================================

const M_PI = 3.14159265358;
const M_TWOPI = 6.28318530718;
const F32_MAX = 3.402823E+38;
const CHUNK_XMAX = 36;
const CHUNK_YMAX = 256;
const CHUNK_ZMAX = 36;
const CHUNK_TSIZE = CHUNK_XMAX * CHUNK_YMAX * CHUNK_ZMAX;

// UTILITY ===================================================


// ===========================================================

// TEST ======================================================

struct DataDda {
    map: vec3<i32>,
    max: vec3<f32>,
    step_amount: vec3<i32>,
    delta: vec3<f32>,
    side: i32,
    hit_data: u32,
    t: f32,
}

struct VoxelHit {
    dist: f32,
    normal: vec3<f32>,
    voxel: u32,
}

fn dda_prepare(ray: Ray) -> DataDda {
    var dda: DataDda;

    dda.map = vec3<i32>(ray.orig);
    dda.delta = vec3(abs(ray.inv_dir));
    dda.step_amount = vec3(0);
    dda.max = vec3(0.0);

    if ray.dir.x < 0.0 {
        dda.step_amount.x = -1;
        dda.max.x = (ray.orig.x - f32(dda.map.x)) * dda.delta.x;
    } else if ray.dir.x > 0.0 {
        dda.step_amount.x = 1;
        dda.max.x = (f32(dda.map.x) + 1.0 - ray.orig.x) * dda.delta.x;
    }

    if ray.dir.y < 0.0 {
        dda.step_amount.y = -1;
        dda.max.y = (ray.orig.y - f32(dda.map.y)) * dda.delta.y;
    } else if ray.dir.y > 0.0 {
        dda.step_amount.y = 1;
        dda.max.y = (f32(dda.map.y) + 1.0 - ray.orig.y) * dda.delta.y;
    }

    if ray.dir.z < 0.0 {
        dda.step_amount.z = -1;
        dda.max.z = (ray.orig.z - f32(dda.map.z)) * dda.delta.z;
    } else if ray.dir.z > 0.0 {
        dda.step_amount.z = 1;
        dda.max.z = (f32(dda.map.z) + 1.0 - ray.orig.z) * dda.delta.z;
    }

    return dda;
}

fn dda_steps(ray: Ray, dda: ptr<function, DataDda>) {
    if (*dda).max.x < (*dda).max.y && (*dda).max.x < (*dda).max.z {
        (*dda).map.x += (*dda).step_amount.x;
        (*dda).max.x += (*dda).delta.x;
        (*dda).side = 0;
        (*dda).t = (f32((*dda).map.x) - ray.orig.x + f32(1 - (*dda).step_amount.x) * 0.5) * ray.inv_dir.x;
    } else if (*dda).max.y < (*dda).max.z {
        (*dda).map.y += (*dda).step_amount.y;
        (*dda).max.y += (*dda).delta.y;
        (*dda).side = 2;
        (*dda).t = (f32((*dda).map.y) - ray.orig.y + f32(1 - (*dda).step_amount.y) * 0.5) * ray.inv_dir.y;
    } else {
        (*dda).map.z += (*dda).step_amount.z;
        (*dda).max.z += (*dda).delta.z;
        (*dda).side = 1;
        (*dda).t = (f32((*dda).map.z) - ray.orig.z + f32(1 - (*dda).step_amount.z) * 0.5) * ray.inv_dir.z;
    }
}

fn dda_voxels(ray: Ray, chunk_offset: u32) -> VoxelHit {
    var voxel_hit: VoxelHit = VoxelHit(F32_MAX, vec3(0.0), 0u);
    var dda: DataDda = dda_prepare(ray);

    var iter = 0u;

    let len = settings.chunk_content_count;
    if chunk_offset >= len || len <= 0u {
        // voxel_hit.voxel = 666u;
        return voxel_hit;
    }

    while iter < 150u && voxel_hit.voxel == 0u {
        iter++;
        dda_steps(ray, &dda);

        let index = i32(chunk_offset) + ((dda.map.z * CHUNK_XMAX * CHUNK_YMAX) + (dda.map.y * CHUNK_XMAX) + dda.map.x);

        if dda.map.x < 0 || dda.map.x >= CHUNK_XMAX || dda.map.y < 0 || dda.map.y >= CHUNK_YMAX || dda.map.z < 0 || dda.map.z >= CHUNK_ZMAX || index >= i32(len) || index < 0 {
            // voxel_hit.voxel = 777u;
            continue;
        }

        // voxel_hit.voxel = chunk_content[
        //     i32(chunk_offset)+
        //     (dda.map.y * CHUNK_XMAX * CHUNK_YMAX + dda.map.z * CHUNK_XMAX + dda.map.x)
        // ];

        voxel_hit.voxel = chunk_content[index];
        voxel_hit.dist = dda.t;
    }

    return voxel_hit;
}

fn intersect_aabb(ray: Ray, min: vec3<f32>, max: vec3<f32>) -> f32 {
    let bmin = min;
    let bmax = max;

    let tx1: f32 = (bmin.x - ray.orig.x) * ray.inv_dir.x;
    let tx2: f32 = (bmax.x - ray.orig.x) * ray.inv_dir.x;

    var tmin: f32 = min(tx1, tx2);
    var tmax: f32 = max(tx1, tx2);

    let ty1: f32 = (bmin.y - ray.orig.y) * ray.inv_dir.y;
    let ty2: f32 = (bmax.y - ray.orig.y) * ray.inv_dir.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    let tz1: f32 = (bmin.z - ray.orig.z) * ray.inv_dir.z;
    let tz2: f32 = (bmax.z - ray.orig.z) * ray.inv_dir.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    let t = max(tmin, tmax);

    if tmax < 0.0 {
        return 0.0;
    }
    if tmin > tmax {
        return 0.0;
    }

    // if tmin < 0.0 {
    //     return tmax;
    // } else {
    //     return tmin;
    // }

    // if tmax >= tmin {
    //     return tmin;
    // }

    return t;


    // let t0s = (min - ray.orig) * ray.inv_dir;
    // let t1s = (max - ray.orig) * ray.inv_dir;

    // let tsmaller = min(t0s, t1s);
    // let tbigger = max(t0s, t1s);

    // let tmin = max(tsmaller[0], max(tsmaller[1], tsmaller[2]));
    // let tmax = min(tbigger[0], min(tbigger[1], tbigger[2]));

    // if tmin < tmax {
    //     return tmin;
    // }


    // return 0.0;
}

struct BvhHitData {
    hit: bool,
    dist: f32,
    offset_index: u32,
    position: vec3<f32>,
}

fn sdf_box(ray_pos: vec3<f32>) -> f32 {
    let b = vec3(32.0, 1.0, 32.0);
    let p = abs(ray_pos) - b;

    let q = abs(p) - b;
    return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_box_sides(ray_pos: vec3<f32>) -> f32 {

    let e = 0.5;
    let b = vec3(36.0, 256.0, 36.0);
    let p = abs(ray_pos) - b;
    let q = abs(p + e) - e;

    return min(min(
        length(max(vec3(p.x, q.y, q.z), vec3(0.0))) + min(max(p.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, p.y, q.z), vec3(0.0))) + min(max(q.x, max(p.y, q.z)), 0.0)
    ),
        length(max(vec3(q.x, q.y, p.z), vec3(0.0))) + min(max(q.x, max(q.y, p.z)), 0.0));
}

fn raytrace(ray_in: Ray) -> vec3<f32> {
    return vec3(
        0.00,
        0.00,
        0.00
    );

//   return vec3(0.05, 0.05, 0.10);
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

    textureStore(color_output, screen_pos, vec4(raytrace(ray).xyz, 1.0));
}