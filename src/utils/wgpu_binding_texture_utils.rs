use wgpu::{BindingType, Sampler, SamplerBindingType, TextureView, TextureViewDimension};
use crate::utils::wgpu_binding_utils::BindingGeneratorBuilder;

impl<'a> BindingGeneratorBuilder<'a> {

    /// Create a default binding with a 2D texture and sampler.
    ///
    /// Sample type is Float and Filterable.
    pub fn with_texture_and_sampler(mut self, texture_view: &'a TextureView, sampler: &'a Sampler) -> BindingGeneratorBuilder<'a> {
        self.context.binding_type = BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float {
                filterable: true
            },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        };
        self.context.set_texture_view(texture_view);
        self.context_done();

        self.context.binding_type = BindingType::Sampler(
            SamplerBindingType::Filtering
        );
        self.context.set_sampler(sampler);
        self.context_done();

        self
    }
}