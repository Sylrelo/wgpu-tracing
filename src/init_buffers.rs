use crate::{
    camera::CameraUniform,
    pipelines::buffers_util::{create_storage_buffer, create_uniform_buffer, SynBuffer},
    Context,
};

pub struct Buffers {
    pub rt_chunk_grid: SynBuffer,
    pub rt_chunk_content: SynBuffer,
    pub rt_unidata: SynBuffer,

    pub uni_camera: SynBuffer,
}

impl Buffers {
    pub fn new(ctx: &Context) -> Self {
        let rt_chunk_grid = create_storage_buffer(ctx, "RT Chunk Grid", 30 * 30 * 16);
        let rt_chunk_content = create_storage_buffer(ctx, "RT Chunk Content", 1147483640);
        let rt_unidata = create_uniform_buffer(ctx, "RT Data", 32);

        let uni_camera = create_uniform_buffer(
            ctx,
            "UNI Camera",
            std::mem::size_of::<CameraUniform>() as u64,
        );

        Self {
            rt_chunk_grid,
            rt_chunk_content,
            rt_unidata,

            uni_camera,
        }
    }
}
