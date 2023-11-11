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

@group(2) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

// CONSTANTS =================================================

const M_PI = 3.1415926535897932384626433832795;
const M_TWOPI = 6.28318530718;
const F32_MAX = 3.402823E+38;
const CHUNK_XMAX = 36;
const CHUNK_YMAX = 256;
const CHUNK_ZMAX = 36;
const CHUNK_TSIZE = CHUNK_XMAX * CHUNK_YMAX * CHUNK_ZMAX;

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

// ===========================================================

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

fn normal_cube(
    ray_position: vec3<f32>,
    pos: vec3<f32>,
    min: vec3<f32>,
    max: vec3<f32>
) -> vec3<f32> {
    let epsilon = 0.00001;

    // let c = (min + max) * 0.5;
    // let p = ray_position - c;
    // let d = (min - max) * 0.5;

    // return normalize(vec3<f32>(floor(p / abs(d) * epsilon)));

    let bmin = min - pos;
    let bmax = max - pos;

    let cx = abs(ray_position.x - bmin.x);
    let fx = abs(ray_position.x - bmax.x);
    let cy = abs(ray_position.y - bmin.y);
    let fy = abs(ray_position.y - bmax.y);
    let cz = abs(ray_position.z - bmin.z);
    let fz = abs(ray_position.z - bmax.z);

    if cx < epsilon {
        return vec3(-1.0, 0.0, 0.0);
    } else if fx < epsilon {
        return vec3(1.0, 0.0, 0.0);
    } else if cy < epsilon {
        return vec3(0.0, -1.0, 0.0);
    } else if fy < epsilon {
        return vec3(0.0, 1.0, 0.0);
    } else if cz < epsilon {
        return vec3(0.0, 0.0, -1.0);
    } else if fz < epsilon {
        return vec3(0.0, 0.0, 1.0);
    }

    return vec3(0.0, 0.0, 0.0);
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
            f32((node.aabbmin_n_type >> 11u) & 63u),
            f32((node.aabbmax >> 26u) & 63u)
        );

        let posmax = vec3(
            f32((node.aabbmin_n_type >> 20u) & 63u),
            f32((node.aabbmin_n_type >> 2u) & 63u),
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

fn bvh_traverse_chunks(ray_in: Ray) -> BvhVoxelHit {
    var hit: BvhVoxelHit = BvhVoxelHit(0u, F32_MAX, vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

    var node_idx = 0u;
    var prev_t = F32_MAX;
    var prev_node: GpuBvhNode;

    // var iter = 0;

    while node_idx < settings.root_chunk_count {
        let node = bvh_root_chunks[node_idx];

        // iter += 1;

        if node.entry == 4294967295u {
            node_idx = node.exit;

            let tmp_hit = traverse_voxels(ray_in, prev_node.min.xyz, node.offset);

            if tmp_hit.dist < hit.dist {
                hit = tmp_hit;
            }

            // if prev_t < hit.dist {
            //     hit.dist = prev_t;
            //     // hit.dist = f32(iter);
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
            prev_t = t;
        } else {
            node_idx = node.exit;
        }
        prev_node = node;
    }

    // hit.dist = f32(iter);
    return hit;
}

fn pathtrace(ray_in: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var throughput: vec3<f32> = vec3(1.0, 1.0, 1.0);
    var color: vec3<f32> = vec3(0.0, 0.0, 0.0);
    var ray = ray_in;

    for (var i = 0; i < 4; i++) {
        var voxel_hit = bvh_traverse_chunks(ray);

        if voxel_hit.material == 0u {
            color += (vec3(161.0 / 255.0, 247.0 / 255.0, 1.0) * 0.35) * throughput;
            break ;
        }

        voxel_hit.point = ray_in.orig + ray_in.dir * voxel_hit.dist;
        voxel_hit.normal = normal_cube(voxel_hit.point, vec3(0.0), voxel_hit.aabb_min, voxel_hit.aabb_max);

        // let ray_position = ray.orig + ray.dir * voxel_hit.dist;
        
        // if true {
        //     return vec3(voxel_hit.dist / 100.0);
        // }

        // let vox_color = vec3(0.4, 0.3, 0.9);
        let vox_color = voxel_hit.normal * 0.5 + 0.5;

        ray.orig = voxel_hit.point + voxel_hit.normal * 0.0001;


        // temporary direct light sampling
        let sun_position = vec3(300.0, 255.0, -100.0);
        let sun_direction = normalize(sun_position - voxel_hit.point);
        let n_light = max(min(dot(voxel_hit.normal, sun_direction), 1.0), 0.0);
        var light_color = (throughput * vox_color * vec3(1.0, 1.0, 1.0) * n_light);

        if random_float_01(seed) > 0.4 {

            ray.dir = normalize(sun_direction);
            precalc_ray(&ray);
            let shadow_ray = bvh_traverse_chunks(ray);

            if shadow_ray.material != 0u {
                light_color *= 0.0;
            }
        }

        color += vec3(0.0) * throughput + light_color;
        color += vec3(0.0) * throughput;
        throughput *= vox_color;

        ray.dir = normalize(random_unit_vector(seed) + voxel_hit.normal);
        precalc_ray(&ray);
    }

    return color;
}

fn raytrace(ray_in: Ray) -> vec3<f32> {
    var hit = bvh_traverse_chunks(ray_in);

    // hit = traverse_voxels(ray_in, vec3(0.0), 0u);

    if hit.material != 0u {
        hit.point = ray_in.orig + ray_in.dir * hit.dist;

        hit.normal = normal_cube(hit.point, vec3(0.0), hit.aabb_min, hit.aabb_max);

        return vec3(hit.normal * 0.5 + 0.5);
        // return vec3(hit.dist / 300.0);
    }

    // if prev_t != F32_MAX {
    //     return vec3(iter / 1500.0, 0.0, 0.0);
    // }

    return vec3(
        0.00,
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

    for (var i = 0; i < MAX_SAMPLES; i++) {
        // seed = (1973u * 9277u + u32(i) * 26699u) | (1u);
        seed = (u32(screen_pos.x) * 1973u + u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);
        // seed = (u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);
        // wang_hash(&seed);
        final_color += pathtrace(ray, &seed);
    }
    final_color = (final_color / f32(MAX_SAMPLES));

    let gamma = 1.6;
    let exposure = 1.0;

    var tone_mapping = vec3(1.0) - exp(-final_color * gamma);
    tone_mapping = pow(tone_mapping, vec3(1.0 / exposure));

    textureStore(color_output, screen_pos, vec4(tone_mapping.xyz, 1.0));
    // textureStore(color_output, screen_pos, vec4(raytrace(ray).xyz, 1.0));
}