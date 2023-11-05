// STRUCTURES ================================================

struct Settings {
    position: vec4<f32>,
    chunk_count: u32,
    _padding: u32,
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
const F64_MAX = 3.402823E+38;

// UTILITY ===================================================


// ===========================================================

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

    var total_dist = 0.0;

    for (var steps = 0; steps < 64; steps++) {
        // var

        let pos = ray_in.orig + ray_in.dir * total_dist;

        var chk_dst = F64_MAX;
        for (var chunk_id = 0u; chunk_id < settings.chunk_count; chunk_id++) {

            let chunk_pos = chunks[chunk_id].xyz * vec3(36.0, 1.0, 36.0);
            let t = sdf_box_sides(chunk_pos - pos);

            if t > 0.0 && t < chk_dst {
                chk_dst = t;
            }
        }

        if chk_dst >= F64_MAX {
            break;
        }

        // chk_dst = sdf_box_sides(pos - vec3(0.0, 0.0, 10.0));

        if chk_dst < 0.1 {
            return vec3(0.2, 0.2, 0.5);
        }


        if total_dist > 500.0 {
            return vec3(0.05, 0.0, 0.0);
            // break;
        }

        total_dist += chk_dst;
    }
//     let len = arrayLength(&chunks);
//     var real_len = 0u;

//     for (var i = 0u; i < len; i++) {
//         if chunks[i][3] > 0.0 {
//             real_len += 1u;
//         }
//     }

    return vec3(
        0.05,
        0.05,
        0.05
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