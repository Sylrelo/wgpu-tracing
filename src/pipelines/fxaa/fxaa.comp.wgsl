@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

//////////////////////////////////////////////////////////

const FXAA_SPAN_MAX = 4.0;
const FXAA_REDUCE_MUL =(1.0 / FXAA_SPAN_MAX);
const FXAA_REDUCE_MIN = (1.0 / 128.0);
const FXAA_SUBPIX_SHIFT = (1.0 / 4.0);

//////////////////////////////////////////////////////////

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let input_size = textureDimensions(texture);


    // textureStore(texture, screen_pos, textureLoad(texture, screen_pos));

    let old_col = textureLoad(texture, screen_pos);

    let uv = vec2<f32>(screen_pos) / vec2<f32>(input_size);
    let inverse = 1.0 / vec2<f32>(input_size);


    let uvs = vec4<f32>(
        uv,
        inverse
    );

    let col = FxaaPixelShader(uvs, 1.0 / vec2<f32>(input_size));

    textureStore(texture, screen_pos, vec4(col.xyz, 1.0));
}

//////////////////////////////////////////////////////////

fn load_tex(p: vec2<f32>) -> vec3<f32> {
    return textureLoad(texture, vec2<i32>(p * vec2(1280.0 * 0.3, 720.0 * 0.3))).xyz;
    // return textureLoad(texture, vec2<i32>(p * vec2(1280.0 * 0.3, 720.0 * 0.3))).xyz;
}

fn FxaaPixelShader(uv: vec4<f32>, rcpFrame: vec2<f32>) -> vec3<f32> {

    let rgbNW = load_tex(uv.xy + vec2(-1.0, -1.0) * rcpFrame.xy);
    let rgbNE = load_tex(uv.xy + vec2(1.0, -1.0) * rcpFrame.xy);
    let rgbSW = load_tex(uv.xy + vec2(-1.0, 1.0) * rcpFrame.xy);
    let rgbSE = load_tex(uv.xy + vec2(1.0, 1.0) * rcpFrame.xy);
    let rgbM = load_tex(uv.xy * rcpFrame.xy);

    let luma = vec3(0.299, 0.587, 0.114);
    let lumaNW = dot(rgbNW, luma);
    let lumaNE = dot(rgbNE, luma);
    let lumaSW = dot(rgbSW, luma);
    let lumaSE = dot(rgbSE, luma);
    let lumaM = dot(rgbM, luma);

    let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    var dir = vec2(0.0, 0.0);
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    let dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),
        FXAA_REDUCE_MIN
    );
    let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
        max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
        dir * rcpDirMin)) * rcpFrame.xy;

    let rgbA = (1.0 / 2.0) * (load_tex(uv.xy + dir * (1.0 / 3.0 - 0.5)) + load_tex(uv.xy + dir * (2.0 / 3.0 - 0.5)));
    let rgbB = rgbA * (1.0 / 2.0) + (1.0 / 4.0) * (load_tex(uv.xy + dir * (0.0 / 3.0 - 0.5)) + load_tex(uv.xy + dir * (3.0 / 3.0 - 0.5)));

    let lumaB = dot(rgbB, luma);

    if (lumaB < lumaMin) || (lumaB > lumaMax) {
        return rgbA;
    }

    return rgbB;
}

