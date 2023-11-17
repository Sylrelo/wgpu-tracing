#![allow(dead_code)]
use crate::utils::wgpu_binding_utils::BindingGeneratorBuilder;
use wgpu::{
    BindingType, Sampler, SamplerBindingType, ShaderStages, StorageTextureAccess, TextureFormat,
    TextureView, TextureViewDimension,
};

impl<'a> BindingGeneratorBuilder<'a> {
    /// Create a default binding with a 2D texture and sampler for a Fragment shader.
    ///
    /// Sample type is Float and Filterable.
    pub fn with_texture_and_sampler(
        self,
        texture_view: &'a TextureView,
        sampler: &'a Sampler,
    ) -> BindingGeneratorBuilder<'a> {
        self.with_texture_and_sampler_stage(ShaderStages::FRAGMENT, texture_view, sampler)
    }

    pub fn with_texture_only(
        mut self,
        visibility: ShaderStages,
        texture_view: &'a TextureView,
    ) -> BindingGeneratorBuilder<'a> {
        self.context.binding_type = BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        };
        self.context.set_texture_view(texture_view);
        self.context.visibility = visibility;
        self.context_done();

        self
    }

    pub fn with_texture_and_sampler_stage(
        mut self,
        visibility: ShaderStages,
        texture_view: &'a TextureView,
        sampler: &'a Sampler,
    ) -> BindingGeneratorBuilder<'a> {
        self.context.binding_type = BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        };
        self.context.set_texture_view(texture_view);
        self.context.visibility = visibility;
        self.context_done();

        self.context.binding_type = BindingType::Sampler(SamplerBindingType::Filtering);
        self.context.set_sampler(sampler);
        self.context.visibility = visibility;
        self.context_done();

        self
    }

    pub fn with_storage_texture(
        mut self,
        texture_view: &'a TextureView,
        format: TextureFormat,
        access: StorageTextureAccess,
    ) -> BindingGeneratorBuilder<'a> {
        self.context.binding_type = BindingType::StorageTexture {
            access,
            format,
            view_dimension: TextureViewDimension::D2,
        };
        self.context.set_texture_view(texture_view);
        self
    }

    // @deprecated pub fn with_default_storage_texture(
    //     mut self,
    //     texture_view: &'a TextureView,
    // ) -> BindingGeneratorBuilder<'a> {
    //     self.context.binding_type = BindingType::StorageTexture {
    //         access: StorageTextureAccess::WriteOnly,
    //         format: TextureFormat::Rgba8Unorm,
    //         view_dimension: TextureViewDimension::D2,
    //     };
    //     self.context.set_texture_view(texture_view);
    //     self
    // }

    // pub fn with_storage_texture(
    //     mut self,
    //     texture_view: &'a TextureView,
    //     format: TextureFormat,
    // ) -> BindingGeneratorBuilder<'a> {
    //     self.context.binding_type = BindingType::StorageTexture {
    //         access: StorageTextureAccess::WriteOnly,
    //         format,
    //         view_dimension: TextureViewDimension::D2,
    //     };
    //     self.context.set_texture_view(texture_view);
    //     self
    // }
}
