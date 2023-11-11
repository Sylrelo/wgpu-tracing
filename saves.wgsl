
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

// fn ug_traverse_root(ray_in: Ray) -> VoxelHit {
//     var hit: VoxelHit;
//     var chunk_ray = ray_in;

//     chunk_ray.orig.x += 540.0;
//     chunk_ray.orig.z += 540.0;

//     var dda: DataDda = dda_prepare(chunk_ray, vec3(36.0, 256.0, 36.0), vec3(0.0));

//     var max_iter = 0u;

//     while max_iter < 45u && hit.voxel == 0u {
//         max_iter += 1u;
//         dda_steps(chunk_ray, &dda);

//         if dda.map.x < 0 || dda.map.x >= 30 || dda.map.z < 0 || dda.map.z >= 30 || dda.map.y != 0 {
//             continue;
//         }
//         // hit.dist = 30.0 - f32(max_iter);

//         let chunk = root_grid[u32(f32(dda.map.x)) + u32(f32(dda.map.z)) * 30u];

//         if chunk.w != 0 {
//             var ray_voxel = ray_in;
//             ray_voxel.orig += vec3<f32>(chunk.xyz);

//             let t = intersect_aabb(
//                 ray_voxel,
//                 vec3<f32>(0.0),
//                 vec3(36.0, 256.0, 36.0),
//             );


//             let t2 = intersect_aabb(
//                 ray_in,
//                 vec3<f32>(chunk.xyz),
//                 vec3<f32>(chunk.xyz) + vec3(36.0, 256.0, 36.0),
//             );

//             ray_voxel.orig = ray_voxel.orig + ray_in.dir * t;

//             if t > 0.0 {
//                 hit = dda_voxels(ray_voxel, vec3<f32>(0.0), u32(chunk.w - 1));
//                 hit.dist += t;
//             }
//         }
//     }

//     return hit;
// }


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

    while iter < 120u && voxel_hit.voxel == 0u {
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