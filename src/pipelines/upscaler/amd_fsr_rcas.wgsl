@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

//////////////////////////////////////////////////////////

const FSR_RCAS_LIMIT = (0.25 - (1.0 / 16.0));

//////////////////////////////////////////////////////////

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let input_size = textureDimensions(texture);


    textureStore(texture, screen_pos, textureLoad(texture, screen_pos));
}
