// STRUCTURES ================================================

struct Settings {
    position: vec4<f32>,
    chunk_count: u32,
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

@group(1) @binding(2)
var<storage> chunks_bvh: array<ChunkBvhNode>;

@group(2) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

// CONSTANTS =================================================

const M_PI = 3.14159265358;
const M_TWOPI = 6.28318530718;
const F32_MAX = 3.402823E+38;

// UTILITY ===================================================


// ===========================================================

// TEST ======================================================

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

    if tmax >= tmin {
        return tmin;
    }

    return 0.0;


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

fn traverse_chunks_bvh(ray: Ray) -> BvhHitData {
    var dst = F32_MAX;
    var hit: BvhHitData = BvhHitData(false, F32_MAX, 0u, vec3(0.0));
    var current_index = 0u;
    var i = 0;

    while current_index < settings.bvh_node_count {
        i++;
        if i > 5000 {
            break;
        }

        let node = chunks_bvh[current_index];

        if node.data[0] == 4294967295u {
            // hit.has_hit = true;
            // let ray_position = ray.orig + ray.dir * hit.dist;

            // hit.normal = normal_cube(ray_position, vec3(0.0), lastmin, lastmax);

            // if last_t < hit.t {
            //     hit.t = last_t;
            // }

            // current_index = node.exit_index;
            // continue;
            hit.hit = true;
            // hit.position = node.min.xyz;
            current_index = node.data[1];
            hit.position = node.min.xyz;
            hit.offset_index = node.data[2];

            break;
        }
        // if node.entry_index == 4294967295u && node.shape_index < arrayLength(&voxels) - 1u {
        //     let current_voxel = voxels[node.shape_index];
        //     let t = intersect_cube(
        //         ray,
        //         current_voxel.min.xyz,
        //         current_voxel.max.xyz,
        //         current_voxel.pos.xyz
        //     );

        //     if t > 0.0 && t < dst {
        //         hit.has_hit = true;
        //         hit.t = t;
        //         hit.tri = i32(node.shape_index);
        //         dst = t;
        //     }
        //     current_index = node.exit_index;
        //     continue;
        // }

        let aabb_test_t = intersect_aabb(
            ray,
            node.min.xyz,
            node.max.xyz,
        );

        if aabb_test_t > 0.0 {
            current_index = node.data[0];

            if aabb_test_t < hit.dist {
                hit.dist = aabb_test_t;
            }
            // hit.dist = aabb_test_t;
            // hit.t = aabb_test_t;
            // lastmax = node.aabb_max.xyz;
            // lastmin = node.aabb_min.xyz;
            // last_t = aabb_test_t;
        } else {
            current_index = node.data[1];
        }
    }

    return hit;
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

    let bvh_chunk_hit = traverse_chunks_bvh(ray_in);

    if bvh_chunk_hit.hit == false {
        return vec3(0.5, 0.00, 0.00);
    }


    var total_dist = 0.0; //bvh_chunk_hit.dist;

    for (var steps = 0; steps < 64; steps++) {
        let pos = ray_in.orig + ray_in.dir * total_dist;

        var chk_dst = F32_MAX;
        // for (var chunk_id = 0u; chunk_id < settings.chunk_count; chunk_id++) {

        //     let chunk_pos = chunks[chunk_id].xyz * vec3(36.0, 1.0, 36.0);
        //     let t = sdf_box_sides(chunk_pos - pos);

        //     if t > 0.0 && t < chk_dst {
        //         chk_dst = t;
        //     }
        // }

        let chunk_pos = chunks[bvh_chunk_hit.offset_index].xyz * vec3(36.0, 1.0, 36.0);
        chk_dst = sdf_box_sides(chunk_pos - pos);
        // let t = sdf_box_sides(chunk_pos - pos);

        if chk_dst >= F32_MAX {
            break;
        }

        if chk_dst < 0.1 {
            return vec3(0.2, 0.2, 0.5);
        }

        if total_dist > 500.0 {
            return vec3(0.05, 0.0, 0.05);
            // break;
        }

        total_dist += chk_dst;
    }

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