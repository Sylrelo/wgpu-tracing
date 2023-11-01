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
    node_index: u32,
    _padding: u32,
}

struct BvhNodeGpu {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    entry_index: u32,
    exit_index: u32,
    shape_index: u32,
    _padding: u32,
}


struct TriangleHit {
    tri: i32,
    has_hit: bool,
    t: f32,
}

struct Ray {
    orig: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
    sign_x: u32,
    sign_y: u32,
    sign_z: u32
}

struct VoxelWorldTest {
    voxel: vec4<f32>,
}

struct Camera {
    position: vec4<f32>
}


@group(0) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

//@group(1) @binding(0)
//var<uniform> unidata: ComputeUniform;

@group(1) @binding(0)
var<storage> triangles: array<Triangle>;

@group(1) @binding(1)
var<storage> voxels: array<Voxel>;

@group(1) @binding(2)
var<storage> bvh: array<BvhNodeGpu>;

@group(1) @binding(3)
var<storage> voxelworld: array<VoxelWorldTest>;

@group(2) @binding(0)
var<uniform> cameraUniform: Camera;

////////////////////////////////////////////////////////////////////////////////////

struct VoxelTraversalHit {
    has_hit: bool,
    t: f32,
    normal: vec3<f32>,
    color: vec3<f32>,
}

fn get_dist(ray: Ray, side: i32, map: vec3<i32>, stepAmount: vec3<i32>) -> f32 {
    var t = 0.0;

    if side == 0 {
        t = (f32(map.x) - ray.orig.x + f32(1 - stepAmount.x) / 2.0) / ray.dir.x;
    } else if side == 1 {
        t = (f32(map.z) - ray.orig.z + f32(1 - stepAmount.z) / 2.0) / ray.dir.z;
    } else {
        t = (f32(map.y) - ray.orig.y + f32(1 - stepAmount.y) / 2.0) / ray.dir.y;
    }

    return t;
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

fn cast_dda_branchless() 
{
}

fn cast_dda(ray: Ray, maxSteps: i32) -> VoxelTraversalHit {
    var map = vec3<i32>(ray.orig);

    var stepAmount = vec3(0);
    let delta = vec3(abs(ray.inv_dir));
    var max = vec3(0.0);
    var voxel = vec4(0.0);
    var side = 0;

    if ray.dir.x < 0.0 {
        stepAmount.x = -1;
        max.x = (ray.orig.x - f32(map.x)) * delta.x;
    } else if ray.dir.x > 0.0 {
        stepAmount.x = 1;
        max.x = (f32(map.x) + 1.0 - ray.orig.x) * delta.x;
    }

    if ray.dir.y < 0.0 {
        stepAmount.y = -1;
        max.y = (ray.orig.y - f32(map.y)) * delta.y;
    } else if ray.dir.y > 0.0 {
        stepAmount.y = 1;
        max.y = (f32(map.y) + 1.0 - ray.orig.y) * delta.y;
    }

    if ray.dir.z < 0.0 {
        stepAmount.z = -1;
        max.z = (ray.orig.z - f32(map.z)) * delta.z;
    } else if ray.dir.z > 0.0 {
        stepAmount.z = 1;
        max.z = (f32(map.z) + 1.0 - ray.orig.z) * delta.z;
    }

    var iter = 0;

    while voxel.w < 1.0 {
        iter += 1;

        if iter >= maxSteps {
            break;
        }

        if max.x < max.y && max.x < max.z {
            map.x += stepAmount.x;
            max.x += delta.x;
            side = 0;
        } else if max.y < max.z {
            map.y += stepAmount.y;
            max.y += delta.y;
            side = 2;
        } else {
            map.z += stepAmount.z;
            max.z += delta.z;
            side = 1;
        }

        // if max.x < max.y {
        //     if max.x < max.z {
        //         map.x += stepAmount.x;
        //         // if map.x >= 20 || map.x < 0 {return vec4(0.0);}
        //         max.x += delta.x;
        //         side = 0;
        //     } else {
        //         map.z += stepAmount.z;
        //         // if map.z >= 20 || map.z < 0 {return vec4(0.0);}
        //         max.z += delta.z;
        //         side = 1;
        //     }
        // } else {
        //     if max.y < max.z {
        //         map.y += stepAmount.y;
        //         // if map.y >= 20 || map.y < 0 {return vec4(0.0);}
        //         max.y += delta.y;
        //         side = 2;
        //     } else {
        //         map.z += stepAmount.z;
        //         // if map.z >= 20 || map.z < 0 {return vec4(0.0);}
        //         max.z += delta.z;
        //         side = 1;
        //     }
        // }

        if map.z < 0 || map.z >= 100 || map.x < 0 || map.x >= 100 || map.y < 0 || map.y >= 100 {
            continue;
        }
        voxel = voxelworld[map.y * 100 * 100 + map.z * 100 + map.x].voxel;
        // voxelworld[map.y * MAP_WIDTH * MAP_HEIGHT + map.z * MAP_WIDTH + map.x];
    }


    return VoxelTraversalHit(voxel.w > 0.9, get_dist(ray, side, map, stepAmount), get_normal(side, stepAmount), voxel.xyz);
}

////////////////////////////////////////////////////////////////////////////////////

fn intersect_aabb(
    ray: Ray,
    min: vec3<f32>,
    max: vec3<f32>,
) -> bool {
    let t0s = (min - ray.orig) * ray.inv_dir;
    let t1s = (max - ray.orig) * ray.inv_dir;

    let tsmaller = min(t0s, t1s);
    let tbigger = max(t0s, t1s);

    let tmin = max(tsmaller[0], max(tsmaller[1], tsmaller[2]));
    let tmax = min(tbigger[0], min(tbigger[1], tbigger[2]));

    return (tmin < tmax);
}

fn intersect_cube(
    ray: Ray,
    min: vec3<f32>,
    max: vec3<f32>,
    pos: vec3<f32>
) -> f32 {
    let bmin = min - pos;
    let bmax = max - pos;

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

    return -1.0;
}

fn normal_cube(
    ray_position: vec3<f32>,
    pos: vec3<f32>,
    min: vec3<f32>,
    max: vec3<f32>
) -> vec3<f32> {
    let epsilon = 0.000001;

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

fn intersect_triangle(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>
) -> f32 {

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

    if abs(a) < 0.00001 {
        return 0.0;
    }

    let f = 1.0 / a;
    let s = ray_origin - p0;
    let u = f * dot(s, h);

    if u < 0.0 || u > 1.0 {
        return 0.0;
    }

    let q = cross(s, edge0);
    let v = f * dot(ray_direction, q);

    if v < 0.0 || u + v > 1.0 {
        return 0.0;
    }

    let t = f * dot(edge1, q);

    if t > 0.001 {
        return t;
    }
    return 0.0;
}

fn traverse_bvh(ray: Ray) -> TriangleHit {
    var dst = 10000000.0;
    var hit: TriangleHit = TriangleHit(0, false, dst);
    var current_index = 0u;
    var i = 0;

    let len = arrayLength(&bvh) ;

    while current_index < len {
        i++;
        if i > 10000 {
            break;
        }

        let node = bvh[current_index];

        if node.entry_index == 4294967295u && node.shape_index < arrayLength(&voxels) - 1u {
            let current_voxel = voxels[node.shape_index];
            let t = intersect_cube(
                ray,
                current_voxel.min.xyz,
                current_voxel.max.xyz,
                current_voxel.pos.xyz
            );

            if t > 0.0 && t < dst {
                hit.has_hit = true;
                hit.t = t;
                hit.tri = i32(node.shape_index);
                dst = t;
            }
            current_index = node.exit_index;
            continue;
        }

        let aabb_test_t = intersect_aabb(
            ray,
            node.aabb_min.xyz,
            node.aabb_max.xyz
        );

        if aabb_test_t {
            current_index = node.entry_index;
        } else {
            current_index = node.exit_index;
        }
    }
    return hit;
}

fn get_closest(ray: Ray) -> TriangleHit {
    var dist = 1000000.0;
    var index = 0;
    var has_hit = false;
    let tot = i32(arrayLength(&voxels));

    for (var i = 0; i < tot; i++) {
        let current_triangle = voxels[i];

        let t = intersect_cube(
            ray,
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

        if t > 0.0 && t < dist {
            index = i;
            dist = t;
            has_hit = true;
        }
    }

    return TriangleHit(index, has_hit, dist);
}

fn sdf_box(ray: vec3<f32>, b: vec3<f32>) -> f32 {
    var q = abs(ray) - b;
    return length(max(q, vec3<f32>(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn get_closest_raymarching(ray: vec3<f32>) -> f32 {
    var dist = 1000000.0;
    var has_hit = false;
    let tot = i32(arrayLength(&triangles));

    for (var i = 0; i < tot; i++) {
//        let current_triangle = triangles[i];
        let t = sdf_box(ray + (triangles[i].p2.xyz * 2.5), vec3<f32>(0.5, 0.5, 0.5));

        if t < dist {
            dist = t;
        }
    }

    return dist;
}

fn raymarch(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> f32 {
    var total_traveled_distance = 0.0;

    for (var i = 0; i < 256; i++) {
        var current_position = ray_origin + total_traveled_distance * ray_direction;
        var closest = get_closest_raymarching(current_position);

        if closest < 0.01 {
            return 1.0;
        }

        if total_traveled_distance > 100.0 {
            break ;
        }

        total_traveled_distance += closest;
    }

    return 0.0;
}

const MAX_SAMPLES = 2;
// var<storage> seed: vec2<f32> = vec2<f32>(0.0, 0.0);

const M_PI = 3.1415926535897932384626433832795;
const M_PI_TWO = 6.28318530718;

//

fn wang_hash(seed: ptr<function, u32>) -> u32 {
    (*seed) = (*seed ^ 61u) ^ (*seed >> 16u);
    (*seed) *= 9u;
    (*seed) = *seed ^ ((*seed) >> 4u);
    (*seed) *= u32(0x27d4eb2d);
    (*seed) = *seed ^ ((*seed) >> 15u);

    return *seed;
}
 
fn RandomFloat01(seed: ptr<function, u32>) -> f32 {
    return f32(wang_hash(seed)) / 4294967296.0;
}
 
fn RandomUnitVector(seed: ptr<function, u32>) -> vec3<f32> {
    let z = RandomFloat01(seed) * 2.0 - 1.0;
    let a = RandomFloat01(seed) * M_PI_TWO;
    let r = sqrt(1.0f - z * z);
    let x = r * cos(a);
    let y = r * sin(a);

    return (vec3(x, y, z));
}


//

// fn rand2n(seed: ptr<function, vec2<f32>>) -> vec2<f32> {
//     (*seed) += vec2(-1.0, -1.0);
//     return vec2(fract(sin(dot((*seed).xy, vec2(12.9898, 78.233))) * 43758.5453), fract(cos(dot((*seed).xy, vec2(4.898, 7.23))) * 23421.631));
// }

// fn ortho(v: vec3<f32>) -> vec3<f32> {
//     if abs(v.x) > abs(v.z) {
//         return vec3(-v.y, v.x, 0.0);
//     }
//     return vec3(0.0, -v.z, v.y);
// }

// fn getSampleBiased(seed: ptr<function, u32>, dir_in: vec3<f32>, power: f32) -> vec3<f32> {
//     let dir = normalize(dir_in);

//     let o1 = normalize(ortho(dir));
//     let o2 = normalize(cross(dir, o1));
//     var r = rand2n(seed);

//     r.x = r.x * 2. * M_PI;
//     r.y = pow(r.y, 1.0 / (power + 1.0));
//     let oneminus = sqrt(1.0 - r.y * r.y);
//     return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
// }

fn raytrace(ray: Ray) -> vec3<f32> {
    // var hit: TriangleHit = TriangleHit(0, false, 0.0);
    let voxelHit = cast_dda(ray, 300);

    if voxelHit.has_hit {
        return (voxelHit.normal * 0.5) + 0.5;
    }
    return vec3(0.0, 0.0, 0.0);
}

fn pathtrace(ray_in: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var throughput: vec3<f32> = vec3(1.0, 1.0, 1.0);
    var color: vec3<f32> = vec3(0.0, 0.0, 0.0);
    var ray = ray_in;

    var maxSteps = 300;
    for (var i = 0; i < 8; i++) {
        // var hit: TriangleHit = TriangleHit(0, false, 0.0);

        // let hit = traverse_bvh(ray);
        if (i > 1) {
            maxSteps = 40;
        }

        let voxelHit = cast_dda(ray, maxSteps);

        // let hit = get_closest(ray);
        // let has_vox = cast_dda(ray);

        // if has_vox {
        //     hit.has_hit = true;
        // }

        if voxelHit.has_hit == false {
            color += vec3(1.00, 1.00, 1.00) * throughput;
            break ;
        }
        let ray_position = ray.orig + ray.dir * voxelHit.t;
        // let cube = voxels[hit.tri];
        // let normal = normal_cube(ray_position, cube.pos.xyz, cube.min.xyz, cube.max.xyz);
        let normal = voxelHit.normal;
        // if hit.tri < 105 {
            // color += vec3(1.0, 1.0, 1.0) * throughput;
        // }

        color += vec3(0.0, 0.0, 0.0) * throughput;
        throughput *= normal * 0.5 + 0.5;
        // throughput *= vec3(0.2, 0.3, 1.0);

        ray.orig = ray_position + normal * 0.0001;
        ray.dir = normalize(RandomUnitVector(seed) + normal);
        precalc_ray(&ray);
        // ray.inv_dir = 1.0 / ray.dir;
    }

    return color;
}

fn precalc_ray(ray: ptr<function, Ray>) {
    (*ray).sign_x = u32((*ray).dir.x < 0.0);
    (*ray).sign_y = u32((*ray).dir.y < 0.0);
    (*ray).sign_z = u32((*ray).dir.z < 0.0);
    (*ray).inv_dir = 1.0 / (*ray).dir;
}

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

    let ray_origin = cameraUniform.position.xyz;
    // let ray_origin = vec3(25.0, 25.0, 65.5);
    let ray_direction = normalize(vec3(ndc_pos.xy, -1.0));
    var ray: Ray = Ray(ray_origin, ray_direction, 1.0 / ray_direction, 0u, 0u, 0u);
    precalc_ray(&ray);

    var final_color = vec4<f32>(0.025, 0.025, 0.025, 1.0);

//    let dist = raymarch(ray_origin, ray_direction);;
//    if (dist > 0.0) {
//        final_color = vec4(0.1, 0.3, 0.6, 1.0);
//    }

    // var seed: vec2<f32> = vec2(0.0, 0.0);
    var seed: u32 = (u32(screen_pos.x) * (1973u) + u32(screen_pos.y) * (9277u) * (26699u)) | (1u);
    // var seed: u32 = uint rngState = u32(u32(fragCoord.x) * u32(1973) + u32(fragCoord.y) * u32(9277) + u32(iFrame) * u32(26699)) | u32(1);

    var path_tracing_color = vec3(0.0, 0.0, 0.0);
    for (var i = 0; i < MAX_SAMPLES; i++) {
        seed = (u32(screen_pos.x) * 1973u + u32(screen_pos.y) * 9277u + u32(i) * 26699u) | (1u);
        // wang_hash(&seed);
        path_tracing_color += pathtrace(ray, &seed);
    }
    path_tracing_color = path_tracing_color / f32(MAX_SAMPLES);

    // path_tracing_color = (path_tracing_color + raytrace(ray).xyz) / 2.0;

    textureStore(color_output, screen_pos, vec4(path_tracing_color.xyz, 1.0));

    // textureStore(color_output, screen_pos, vec4(raytrace(ray).xyz, 1.0));

}