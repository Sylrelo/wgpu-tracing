@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

//////////////////////////////////////////////////////////

const FSR_RCAS_LIMIT = (0.25 - (1.0 / 16.0));
const SHARPNESS = 0.3;

//////////////////////////////////////////////////////////

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let input_size = textureDimensions(texture);


    var con = 1.0;
    let color = FsrRcasF(vec2<f32>(screen_pos), con);

    // textureStore(texture, screen_pos, vec4(1.0));

    // textureStore(texture, screen_pos, textureLoad(texture, screen_pos));
    textureStore(texture, screen_pos, vec4(color, 1.0));
}

//////////////////////////////////////////////////////////

fn FsrRcasLoadF(pos: vec2<f32>) -> vec3<f32> {
    // return texture(iChannel0,p/iResolution.xy);
    return textureLoad(texture, vec2<i32>(vec2<f32>(pos))).rgb;
}


fn FsrRcasF(
    ip: vec2<f32>,
    con: f32
) -> vec3<f32> {

    let sp = vec2(ip);
    let b = FsrRcasLoadF(sp + vec2(0.0, -1.0)).rgb;
    let d = FsrRcasLoadF(sp + vec2(-1.0, 0.0)).rgb;
    let e = FsrRcasLoadF(sp).rgb;
    let f = FsrRcasLoadF(sp + vec2(1.0, 0.0)).rgb;
    let h = FsrRcasLoadF(sp + vec2(0.0, 1.0)).rgb;

    let bL = b.g + 0.5 * (b.b + b.r);
    let dL = d.g + 0.5 * (d.b + d.r);
    let eL = e.g + 0.5 * (e.b + e.r);
    let fL = f.g + 0.5 * (f.b + f.r);
    let hL = h.g + 0.5 * (h.b + h.r);

    var nz = 0.25 * (bL + dL + fL + hL) - eL;

    nz = clamp(
        abs(nz) / (max(max(bL, dL), max(eL, max(fL, hL))) - min(min(bL, dL), min(eL, min(fL, hL)))),
        0.0, 1.0
    );

    nz = 1.0 - 0.5 * nz;
    let mn4 = min(b, min(f, h));
    let mx4 = max(b, max(f, h));
    let peakC = vec2(1., -4.);

    let hitMin = mn4 / (4. * mx4);
    let hitMax = (peakC.x - mx4) / (4. * mn4 + peakC.y);
    let lobeRGB = max(-hitMin, hitMax);
    var lobe = max(
        -FSR_RCAS_LIMIT,
        min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)
    ) * con;
    
    // #ifdef FSR_RCAS_DENOISE
    // lobe *= nz;
    // #endif
    return (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
} 
