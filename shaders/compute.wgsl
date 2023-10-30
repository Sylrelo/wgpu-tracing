struct ComputeUniform {
    view_proj: mat4x4<f32>,
    test: vec4<f32>,
}

struct Triangle {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
}

struct Voxel {
    min: vec4<f32>,
    max: vec4<f32>,
    pos: vec4<f32>,
}

@group(0) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

//@group(1) @binding(0)
//var<uniform> unidata: ComputeUniform;

@group(1) @binding(0)
var<storage> triangles: array<Triangle>;

@group(1) @binding(1)
var<storage> voxels: array<Voxel>;


fn intersect_cube(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    min: vec3<f32>,
    max: vec3<f32>,
    pos: vec3<f32>
) -> f32
{
    let bmin = min - pos;
    let bmax = max - pos;

    let tx1: f32 = (bmin.x - ray_origin.x) / ray_direction.x;
    let tx2: f32 = (bmax.x - ray_origin.x) / ray_direction.x;

    var tmin: f32 = min(tx1, tx2);
    var tmax: f32 = max(tx1, tx2);

    let ty1: f32 = (bmin.y - ray_origin.y) / ray_direction.y;
    let ty2: f32 = (bmax.y - ray_origin.y) / ray_direction.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    let tz1: f32 = (bmin.z - ray_origin.z) / ray_direction.z;
    let tz2: f32 = (bmax.z - ray_origin.z) / ray_direction.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    if (tmax >= tmin) {
        return tmin;
    }

    return 0.0;
}

fn intersect_triangle(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>) -> f32
{

//
//      let e1 = p0 - p2;
//      let e2 = p1 - p2;
//
//      let t = o - v2;
//      let p = cross(d, e2);
//      let q = cross(t, e1);
//      let a = vec3(dot(q,e2), dot(p,t), dot(q,d)) / dot(p,e1);
//
//      return a;
//      return vec4(a, 1.0 - a.y - a.z);


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

struct TriangleHit {
    tri: i32,
    has_hit: bool,
}

fn get_closest(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> TriangleHit
{
    var dist = 1000000.0;
    var index = 0;
    var has_hit = false;
    let tot = i32(arrayLength(&voxels));

    for (var i = 0; i < tot; i++) {
        let current_triangle = voxels[i];

        let t = intersect_cube(
            ray_origin,
            ray_direction,
            current_triangle.min.xyz,
            current_triangle.max.xyz,
            current_triangle.pos.xyz
        );
        // let t = intersect_triangle(
        //     ray_origin,
        //     ray_direction,
        //     current_triangle.p0.xyz,
        //     current_triangle.p1.xyz,
        //     current_triangle.p2.xyz
        // );

        if (t > 0.0 && t < dist) {
            index = i;
            dist = t;
            has_hit = true;
        }
    }

    return TriangleHit(index, has_hit);
}

fn sdf_box(ray: vec3<f32>, b: vec3<f32>) -> f32
{
      var q = abs(ray) - b;
      return length(max(q, vec3<f32>(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn get_closest_raymarching(ray: vec3<f32>) -> f32
{
    var dist = 1000000.0;
    var has_hit = false;
    let tot = i32(arrayLength(&triangles));

    for (var i = 0; i < tot; i++)
    {
//        let current_triangle = triangles[i];
        let t = sdf_box(ray  + (triangles[i].p2.xyz * 2.5), vec3<f32>(0.5, 0.5, 0.5));

        if (t < dist) {
            dist = t;
        }
    }

    return dist;
}

fn raymarch(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> f32
{
    var total_traveled_distance = 0.0;

    for (var i = 0; i < 256; i++) {
        var current_position = ray_origin + total_traveled_distance * ray_direction;
        var closest = get_closest_raymarching(current_position);

        if (closest < 0.01) {
            return 1.0;
        }

        if (total_traveled_distance > 100.0) {
            break ;
        }

        total_traveled_distance += closest;
    }

    return 0.0;
}

@compute
@workgroup_size(16, 9)
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

    let ray_origin = vec3(0.0, 0.0, 7.5);
    let ray_direction = normalize(vec3(ndc_pos.xy, -1.0));

    var final_color = vec4<f32>(0.025, 0.025, 0.025, 1.0);

//    let dist = raymarch(ray_origin, ray_direction);;
//    if (dist > 0.0) {
//        final_color = vec4(0.1, 0.3, 0.6, 1.0);
//    }

    let hit = get_closest(ray_origin, ray_direction);
    if (hit.has_hit == true) {
        final_color = vec4(0.1, 0.6, 0.6, 1.0);
    }

    textureStore(color_output, screen_pos, final_color);
}