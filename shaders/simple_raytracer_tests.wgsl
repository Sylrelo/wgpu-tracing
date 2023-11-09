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
var<storage> root_chunks: array<vec4<i32>>;

@group(1) @binding(2)
var<storage> root_grid: array<vec4<i32>>;

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

fn dda_voxels(ray: Ray, min_bound: vec3<f32>, chunk_offset: u32) -> VoxelHit {
    var voxel_hit: VoxelHit = VoxelHit(F32_MAX, vec3(0.0), 0u);
    var dda: DataDda = dda_prepare(ray, vec3(1.0), min_bound);

    var iter = 0u;

    let len = settings.chunk_content_count;
    if chunk_offset >= len || len <= 0u {
        return voxel_hit;
    }

    while iter < 90u && voxel_hit.voxel == 0u {
        iter++;
        dda_steps(ray, &dda);

        let index = i32(chunk_offset) + ((dda.map.z * CHUNK_XMAX * CHUNK_YMAX) + (dda.map.y * CHUNK_XMAX) + dda.map.x);

        if dda.map.x < 0 || dda.map.x >= CHUNK_XMAX || dda.map.y < 0 || dda.map.y >= CHUNK_YMAX || dda.map.z < 0 || dda.map.z >= CHUNK_ZMAX || index >= i32(len) || index < 0 {
            continue;
        }

        voxel_hit.voxel = chunk_content[index];
        // voxel_hit.dist = 100.0 - f32(iter) ;
        voxel_hit.dist = dda.t;
        voxel_hit.normal = get_normal(dda.side, dda.step_amount);
    }

    return voxel_hit;
}

fn intersect_aabb(ray: Ray, min: vec3<f32>, max: vec3<f32>) -> f32 {
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

    return 0.0;
}

fn ug_traverse_root(ray_in: Ray) -> VoxelHit {
    var hit: VoxelHit;
    var chunk_ray = ray_in;

    chunk_ray.orig.x += 540.0;
    chunk_ray.orig.z += 540.0;

    var dda: DataDda = dda_prepare(chunk_ray, vec3(36.0, 256.0, 36.0), vec3(0.0));

    var max_iter = 0u;

    while max_iter < 45u && hit.voxel == 0u {
        max_iter += 1u;
        dda_steps(chunk_ray, &dda);

        if dda.map.x < 0 || dda.map.x >= 30 || dda.map.z < 0 || dda.map.z >= 30 || dda.map.y != 0 {
            continue;
        }
        // hit.dist = 30.0 - f32(max_iter);

        let chunk = root_grid[u32(f32(dda.map.x)) + u32(f32(dda.map.z)) * 30u];

        if chunk.w != 0 {
            var ray_voxel = ray_in;
            ray_voxel.orig += vec3<f32>(chunk.xyz);

            let t = intersect_aabb(
                ray_voxel,
                vec3<f32>(0.0),
                vec3(36.0, 256.0, 36.0),
            );
            ray_voxel.orig = ray_voxel.orig + ray_in.dir * t;

            if t > 0.0 {
                hit = dda_voxels(ray_voxel, vec3<f32>(0.0, 0.0, 0.0), u32(chunk.w - 1));
                hit.dist += t;
            }
        }
    }

    return hit;
}


// RAYTRACING =================================================

fn precalc_ray(ray: ptr<function, Ray>) {
    // (*ray).sign_x = u32((*ray).dir.x < 0.0);
    // (*ray).sign_y = u32((*ray).dir.y < 0.0);
    // (*ray).sign_z = u32((*ray).dir.z < 0.0);
    (*ray).inv_dir = 1.0 / (*ray).dir;
}

fn pathtrace(ray_in: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var throughput: vec3<f32> = vec3(1.0, 1.0, 1.0);
    var color: vec3<f32> = vec3(0.0, 0.0, 0.0);
    var ray = ray_in;

    for (var i = 0; i < 6; i++) {
        let voxel_hit = ug_traverse_root(ray);

        if voxel_hit.voxel == 0u {
            color += vec3(0.0, 0.0, 0.0) * throughput;
            break ;
        }

        let ray_position = ray.orig + ray.dir * voxel_hit.dist;
        
        // if true {
        //     return vec3(voxel_hit.dist / 100.0);
        // }

        let vox_color = vec3(0.4, 0.3, 0.9);
        // let vox_color = voxel_hit.normal * 0.5 + 0.5;

        ray.orig = ray_position + voxel_hit.normal * 0.001;


        // temporary direct light sampling

        let sun_position = vec3(-200.0, 290.0, -55.0);
        let sun_direction = normalize(sun_position - ray_position);
        let n_light = max(min(dot(voxel_hit.normal, sun_direction), 1.0), 0.0);

        ray.dir = normalize(sun_direction);
        precalc_ray(&ray);
        let shadow_ray = ug_traverse_root(ray);

        color += vec3(0.0) * throughput + (throughput * vox_color * vec3(1.0, 1.0, 1.0) * n_light * f32(shadow_ray.voxel == 0u));

        // color += vec3(0.0) * throughput;
        throughput *= vox_color;

        ray.dir = normalize(random_unit_vector(seed) + voxel_hit.normal);
        precalc_ray(&ray);
    }

    return color;
}


fn raytrace(ray_in: Ray) -> vec3<f32> {
    var ray = ray_in;

    let vox_hit = ug_traverse_root(ray);
    let t = vox_hit.dist / 550.0;

    if vox_hit.voxel != 0u {
        return vec3(vox_hit.normal * t);
        // return vec3(vox_hit.dist / 550.0);
    } else {
        return vec3((vox_hit.dist / 500.0), 0.0, 0.0);
        // return vec3(0.01);
    }

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

    for (var i = 0; i < 2; i++) {
        seed = (1973u * 9277u + u32(i) * 26699u) | (1u);
        seed = (u32(screen_pos.x) * 1973u + u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);
        // seed = (u32(screen_pos.x) * 1973u + u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);
        // wang_hash(&seed);
        final_color += pathtrace(ray, &seed);
    }
    final_color = (final_color / f32(2));

    let gamma = 1.6;
    let exposure = 1.0;

    var tone_mapping = vec3(1.0) - exp(-final_color * gamma);
    tone_mapping = pow(tone_mapping, vec3(1.0 / exposure));

    textureStore(color_output, screen_pos, vec4(tone_mapping.xyz, 1.0));
    // textureStore(color_output, screen_pos, vec4(raytrace(ray).xyz, 1.0));
}