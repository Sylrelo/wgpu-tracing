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

// DATA ======================================================

@group(0) @binding(0)
var<uniform> settings: Settings;

@group(1) @binding(0)
var<storage> chunk_content: array<u32>;

@group(1) @binding(1)
var<storage> root_grid_chunks: array<vec4<i32>>;

// @group(2) @binding(0)
// var color_output: texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(0)
var normal_output: texture_storage_2d<rgba8snorm, write>;

@group(2) @binding(1)
var color_output: texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(2)
var depth_output: texture_storage_2d<rgba32float, read_write>;

@group(2) @binding(3)
var velocity_texture: texture_storage_2d<rgba32float, write>;

// CONSTANTS =================================================

const M_PI = 3.1415926535897932384626433832795;
const M_TWOPI = 6.28318530718;
const F32_MAX = 3.402823E+38;
const CHUNK_XMAX = 64;
const CHUNK_YMAX = 64;
const CHUNK_ZMAX = 64;
// const CHUNK_TSIZE = CHUNK_XMAX * CHUNK_YMAX * CHUNK_ZMAX;
// const CHUNK_TSIZE = 331776;
const MEM_SIZE = 262144u;

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

    if tmin < tmax {
        return tmin;
    }

    return 0.0;
}

// RAYTRACING =================================================

fn precalc_ray(ray: ptr<function, Ray>) {
    // (*ray).sign_x = u32((*ray).dir.x < 0.0);
    // (*ray).sign_y = u32((*ray).dir.y < 0.0);
    // (*ray).sign_z = u32((*ray).dir.z < 0.0);
    (*ray).inv_dir = 1.0 / (*ray).dir;
}

fn dda_prepare_scratch(
    ray_in: Ray,
    grid_min: vec3<f32>,
    grid_max: vec3<f32>,
    cell_dimensions: vec3<f32>,
    aabb_tmin: f32,
) -> DataDda {
    var dda: DataDda;
    let resolution = vec3(CHUNK_XMAX, CHUNK_YMAX, CHUNK_XMAX);

    let ray_orig_cell = (ray_in.orig + ray_in.dir * aabb_tmin) - grid_min;
    dda.map = clamp(vec3<i32>(floor(ray_orig_cell / cell_dimensions)), vec3(0), resolution);


    if ray_in.dir.x < 0.0 {
        dda.step_amount.x = -1;
        dda.delta.x = -cell_dimensions.x * ray_in.inv_dir.x;
        dda.max.x = aabb_tmin + (f32(dda.map.x) * cell_dimensions.x - ray_orig_cell.x) * ray_in.inv_dir.x;
    } else {
        dda.step_amount.x = 1;
        dda.delta.x = cell_dimensions.x * ray_in.inv_dir.x;
        dda.max.x = aabb_tmin + (f32(dda.map.x + 1) * cell_dimensions.x - ray_orig_cell.x) * ray_in.inv_dir.x;
    }

    if ray_in.dir.y < 0.0 {
        dda.step_amount.y = -1;
        dda.delta.y = -cell_dimensions.y * ray_in.inv_dir.y;
        dda.max.y = aabb_tmin + (f32(dda.map.y) * cell_dimensions.y - ray_orig_cell.y) * ray_in.inv_dir.y;
    } else {
        dda.step_amount.y = 1;
        dda.delta.y = cell_dimensions.y * ray_in.inv_dir.y;
        dda.max.y = aabb_tmin + (f32(dda.map.y + 1) * cell_dimensions.y - ray_orig_cell.y) * ray_in.inv_dir.y;
    }

    if ray_in.dir.z < 0.0 {
        dda.step_amount.z = -1;
        dda.delta.z = -cell_dimensions.z * ray_in.inv_dir.z;
        dda.max.z = aabb_tmin + (f32(dda.map.z) * cell_dimensions.z - ray_orig_cell.z) * ray_in.inv_dir.z;
    } else {
        dda.step_amount.z = 1;
        dda.delta.z = cell_dimensions.z * ray_in.inv_dir.z;
        dda.max.z = aabb_tmin + (f32(dda.map.z + 1) * cell_dimensions.z - ray_orig_cell.z) * ray_in.inv_dir.z;
    }
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
    let mask = ((*dda).max.xyz <= min((*dda).max.yzx, (*dda).max.zxy));

    (*dda).map += vec3<i32>(mask) * (*dda).step_amount;
    (*dda).max += vec3<f32>(mask) * (*dda).delta;

    let tmp_side = vec3<i32>(mask) * vec3(1, 3, 2);
    (*dda).side = max(tmp_side.x, max(tmp_side.y, tmp_side.z)) - 1;

    (*dda).mask = mask;
}

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
        vec3(1.0),
        dist,
    );
    // var dda: DataDda = dda_prepare(ray_in, vec3(1.0), min_bound);
    var iter = 0u;

    let len = settings.chunk_content_count;

    if chunk_offset >= len || len <= 0u {
        return dda;
    }

    while iter < 140u && dda.hit_data == 0u {
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


fn traverse_root_grid(
    ray_in: Ray,
) -> DataDda {
    var ray_chunks = ray_in;

    var dda: DataDda = dda_prepare_scratch(
        ray_chunks,
        vec3(0.0),
        vec3(0.0),
        vec3<f32>(vec3(CHUNK_XMAX, CHUNK_YMAX, CHUNK_XMAX)),
        0.0
    );

    var iter = 0u;

    var voxel_hit: DataDda;
    voxel_hit.t = F32_MAX;

    var grid_hit = vec4(0, 0, 0, 0);

    var last_t = F32_MAX;

    let ray_orig_as_chunkp = vec3<i32>(ray_in.orig) / (vec3(CHUNK_XMAX, CHUNK_YMAX, CHUNK_ZMAX));
    let index_current_position = ((ray_orig_as_chunkp.z * 30) + ray_orig_as_chunkp.x);


    while iter < 10u && voxel_hit.hit_data == 0u {
        iter++;
        dda_steps(ray_in, &dda);

        let index = ((dda.map.z * 30) + dda.map.x);

        if dda.map.x < 0 || dda.map.x >= 30 || dda.map.y != 0 || dda.map.z < 0 || dda.map.z >= 30 {
            continue;
        }

        grid_hit = root_grid_chunks[index];

        if grid_hit.w != 0 {

            let t = intersect_aabb(
                ray_in,
                vec3<f32>(grid_hit.xyz),
                vec3<f32>(grid_hit.xyz) + vec3<f32>(vec3(CHUNK_XMAX, CHUNK_YMAX, CHUNK_XMAX)),
            );
            last_t = t;

            let tmp_voxel_hit = traver_voxel_ug(
                ray_in,
                u32(grid_hit.w - 1),
                vec3<f32>(grid_hit.xyz),
                vec3<f32>(grid_hit.xyz) + vec3<f32>(vec3(CHUNK_XMAX, CHUNK_YMAX, CHUNK_XMAX)),
                t
            );
            if tmp_voxel_hit.hit_data != 0u && tmp_voxel_hit.t < voxel_hit.t {
                voxel_hit = tmp_voxel_hit;
            }
        }
    }

    // Clusterfuck to handle the case of being IN the cell.
    if index_current_position < 0 || u32(index_current_position) >= arrayLength(&root_grid_chunks) {
        return voxel_hit;
    }

    grid_hit = root_grid_chunks[index_current_position];

    if grid_hit.w == 0 {
        return voxel_hit;
    }

    let t = intersect_aabb(
        ray_in,
        vec3<f32>(grid_hit.xyz),
        vec3<f32>(grid_hit.xyz) + vec3<f32>(vec3(CHUNK_XMAX, CHUNK_YMAX, CHUNK_XMAX)),
    );

    if t > 0.0 && t < last_t {
        let tmp_voxel_hit = traver_voxel_ug(
            ray_in,
            u32(grid_hit.w - 1),
            vec3<f32>(grid_hit.xyz),
            vec3<f32>(grid_hit.xyz) + vec3<f32>(vec3(CHUNK_XMAX, CHUNK_YMAX, CHUNK_XMAX)),
            0.0
        );
        if tmp_voxel_hit.hit_data != 0u && tmp_voxel_hit.t < voxel_hit.t {
            voxel_hit = tmp_voxel_hit;
        }
    }

    return voxel_hit;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    shadow_ray.dir = sample_cone(seed, light_dir, theta * 0.5);
    shadow_ray.orig = hit_point + hit_normal * 0.0001;
    precalc_ray(&shadow_ray);

    let inv_prob = 2.0 * (1.0 - cos(theta)) * 50.0;
    let light_val = clamp(dot(hit_normal, light_dir), 0.0, 1.0);

    // let shadow_hit = bvh_traverse_chunks(shadow_ray);
    let shadow_hit = traverse_root_grid(shadow_ray);

    if shadow_hit.hit_data == 0u {
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
        return (reflect(ray_dir, normal));
        // return normalize(reflect(ray_dir, -normal) + getCosineWeightedSample(seed, normal) * 0.05);
    } else {
        // return random_unit_vector(seed);
        return getCosineWeightedSample(seed, normal);
    }
}

fn pathtrace(ray_in: Ray, seed: ptr<function, u32>, screen_pos: vec2<i32>) -> vec3<f32> {
    var throughput: vec3<f32> = vec3(1.0, 1.0, 1.0);
    var ray = ray_in;
    var specular_bounce = true;
    var color: vec3<f32> = vec3(0.0, 0.0, 0.0);

    var is_prev_reflection = false;
    let prev_position = textureLoad(depth_output, screen_pos);

    for (var i = 0; i < 3; i++) {
        // var voxel_hit = bvh_traverse_chunks(ray);
        let voxel_hit = traverse_root_grid(ray);

        if voxel_hit.hit_data == 0u || voxel_hit.t == F32_MAX {
            color += (vec3(161.0 / 255.0, 247.0 / 255.0, 1.0) * 0.0) * throughput;
            break;
        }

        let point = ray.orig + ray.dir * voxel_hit.t;
        // let normal = get_normal(voxel_hit.side, voxel_hit.step_amount);
        let normal = vec3<f32>(voxel_hit.mask) * -sign(vec3<f32>(voxel_hit.step_amount));

        if i == 0 || (is_prev_reflection && i == 1) {
            textureStore(normal_output, screen_pos, normal.xyzz);




            let current_position = vec4((point * 0.001), 1.0);
            textureStore(depth_output, screen_pos, current_position);

            textureStore(velocity_texture, screen_pos, vec4((current_position.xyz - prev_position.xyz) * 1000.0, 1.0));
            // textureStore(depth_output, screen_pos, vec4(vec3(voxel_hit.t / 500.0), 1.0));
        }

        var vox_color = normal * 0.5 + 0.5;

        // if i == 1 {
        //     return vox_color;
        // }
        // throughput *= vox_color;

        vox_color = vec3(0.6, 0.3, 0.6);
        // vox_color = vec3(1.0);

        ray.orig = point + normal * 0.0001;

        // if voxel_hit.hit_data == 2u {
        //     color += throughput * vec3(1.0, 1.0, 1.0) ;
        //     break;
        // }

        ray.dir = brdf(seed, normal, ray.dir, voxel_hit.hit_data, &specular_bounce);

        if !specular_bounce || dot(ray.dir, normal) < 0.0 {
            throughput *= vox_color;
        }

        if !specular_bounce {
            // color += throughput;
            color += throughput * sample_sunlight(seed, point, normal);
        }
        // if voxel_hit.material == 1u {
        //     throughput *= vox_color;
        //     // ray.dir = normalize(random_unit_vector(seed) + voxel_hit.normal);
        //     ray.dir = getCosineWeightedSample(seed, voxel_hit.normal);

        //     color += throughput * sample_sunlight(seed, voxel_hit.point, voxel_hit.normal);
        // }

        if voxel_hit.hit_data == 3u {
            is_prev_reflection = true;
            color += throughput * vec3(0.0, 0.5, 1.0) * 0.05;
        //     ray.dir = normalize(reflect(ray.dir, voxel_hit.normal) + getCosineWeightedSample(seed, voxel_hit.normal) * 0.02);
        }


        precalc_ray(&ray);
    }

    return color;
    // return throughput;
}


fn raytrace(ray_in: Ray, screen_pos: vec2<f32>) -> vec3<f32> {
    var final_color = vec3(0.0);
    var seed = 0u;
    let MAX_SAMPLES = 1;

    for (var i = 0; i < MAX_SAMPLES; i++) {
        var ray = ray_in;

        ray.orig.x += f32((MAX_SAMPLES / 2) - i) * 0.01;
        ray.orig.z += f32((MAX_SAMPLES / 2) - i) * 0.01;

        var color = vec3(0.0, 0.0, 0.0);
        var ratio = 1.0;
        seed = (u32(screen_pos.x) * 1973u + u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);

        let hit = traverse_root_grid(ray);

        if hit.hit_data == 0u { 
                continue;
        }

        let normal = vec3<f32>(hit.mask) * -sign(vec3<f32>(hit.step_amount));
        let hit_point = ray.orig + ray.dir * hit.t;
        let light_color = sample_sunlight(&seed, hit_point, normal);
        color += ((normal * 0.5 + 0.5) * light_color) ;

        final_color += color;
    }

    return final_color / f32(MAX_SAMPLES);
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

    ray.dir = vec_rot_x(ray.dir, -0.45);
    // ray.dir = vec_rot_y(ray.dir, -1.9);
    precalc_ray(&ray);

    // textureStore(normal_output, screen_pos, vec4(0.0));
    // textureStore(depth_output, screen_pos, vec4(0.0));

    for (var i = 0; i < MAX_SAMPLES; i++) {
        // seed = (1973u * 9277u + u32(i) * 26699u) | (1u);
        seed = (u32(screen_pos.x) * 1973u + u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);

        // let foc_target = ray.orig + ray.dir * 2.3;
        // let defocus = 0.05 * rand2_in_circle(&seed);

        // ray.orig += vec3(defocus.xy, 0.0);
        // ray.dir = normalize(foc_target - ray.orig);
        // precalc_ray(&ray);

        final_color += pathtrace(ray, &seed, screen_pos);
    }
    final_color = (final_color / f32(MAX_SAMPLES));

    let gamma = 1.6;
    let exposure = 1.0;

    var tone_mapping = vec3(1.0) - exp(-final_color * gamma);
    tone_mapping = pow(tone_mapping, vec3(1.0 / exposure));

    textureStore(color_output, screen_pos, vec4(tone_mapping.xyz, 1.0));
    // textureStore(color_output2, screen_pos, vec4(tone_mapping.xyz, 1.0));
    // textureStore(color_output, screen_pos, vec4(raytrace(ray, ndc_pixel).xyz, 1.0));
}