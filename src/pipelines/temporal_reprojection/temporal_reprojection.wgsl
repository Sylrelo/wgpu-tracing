@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

//////////////////////////////////////////////////////////

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let input_size = textureDimensions(texture);

    let old_col = textureLoad(texture, screen_pos);
    let uv = vec2<f32>(screen_pos) / vec2<f32>(input_size);

    textureStore(texture, screen_pos, vec4(old_col.xyz, 1.0));
}
