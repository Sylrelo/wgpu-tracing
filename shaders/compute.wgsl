struct ComputeUniform {
    view_proj: mat4x4<f32>,
    test: vec4<f32>,
}

struct Triangle {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
}

@group(0) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

//@group(1) @binding(0)
//var<uniform> unidata: ComputeUniform;

@group(1) @binding(0)
var<storage> triangles: array<Triangle>;

fn intersect_triangle(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>) -> f32
{
    let edge0 = p1 - p0;
    let edge1 = p2 - p0;

    let h = cross(ray_direction, edge1);
    let a = dot(edge0, h);

    if (abs(a) < 0.00001) {
        return 0.0;
    }

    let f = 1.0 / a;
    let s = ray_origin - p0;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return 0.0;
    }

    let q = cross(s, edge0);
    let v = f * dot(ray_direction, q);

    if (v < 0.0 || u + v > 1.0) {
        return 0.0;
    }

    let t = f * dot(edge1, q);

    if (t > 0.00001) {
        return t;
    }
    return 0.0;
}

@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_size: vec2<u32> = textureDimensions(color_output);
    let screen_pos : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let aspect_ratio = f32(screen_size.x) / f32(screen_size.y);

    let tatan = tan(1.5708 / 2.0);
    let ndc_pixel = vec2(
        (f32(screen_pos.x) + 0.5) / f32(screen_size.x),
        (f32(screen_pos.y) + 0.5) / f32(screen_size.y),
    );
    let ndc_pos = vec2<f32>(
        (2.0 * f32(ndc_pixel.x) - 1.0 * tatan) * aspect_ratio,
        1.0 - 2.0 * f32(ndc_pixel.y) * tatan
    );

    let ray_origin = vec3(0.0, 0.0, 10.0);
    let ray_direction = normalize(vec3(ndc_pos.xy, -1.0) - ray_origin);

    var total_dist = 0.0;
    var final_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    let tot = i32(arrayLength(&triangles));
    var t_dst = 100000;
    for (var i = 0; i < tot; i++) {
        let tri = triangles[i];

        let t = intersect_triangle(
            ray_origin,
            ray_direction,
            tri.p0.xyz,
            tri.p1.xyz,
            tri.p2.xyz
        );

        if (t > 0.0) {
            final_color = vec4(0.5, 0.2, 0.0, 1.0);
        }
    }


//    if (intersect_triangle(ray_origin, ray_direction, vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0)) > 0.0) {
//        final_color = vec4(0.6, 0.6, 0.3, 1.0);
//    }

    // test cacahuzete
    textureStore(color_output, screen_pos, final_color);
}