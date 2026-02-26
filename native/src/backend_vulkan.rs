use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use ash::vk;
use std::cell::RefCell;
use std::time::Instant;

#[cfg(target_os = "android")]
const ANDROID_LOG_INFO: i32 = 4;

#[cfg(target_os = "android")]
extern "C" {
    fn __android_log_write(
        prio: i32,
        tag: *const std::os::raw::c_char,
        text: *const std::os::raw::c_char,
    ) -> i32;
}

fn log_vulkan_timing(message: &str) {
    #[cfg(target_os = "android")]
    {
        if let (Ok(tag), Ok(text)) = (
            std::ffi::CString::new("plonky3"),
            std::ffi::CString::new(message),
        ) {
            unsafe {
                let _ = __android_log_write(ANDROID_LOG_INFO, tag.as_ptr(), text.as_ptr());
            }
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        eprintln!("{message}");
    }
}

struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    timestamp_period_ns: f32,
    timestamp_supported: bool,
}

struct VulkanPipelineState {
    shader_module: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct VulkanIoState {
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    data_buffer: vk::Buffer,
    data_memory: vk::DeviceMemory,
    data_capacity: usize,
    staging_upload_buffer: vk::Buffer,
    staging_upload_memory: vk::DeviceMemory,
    staging_upload_ptr: *mut u32,
    staging_upload_capacity: usize,
    staging_readback_buffer: vk::Buffer,
    staging_readback_memory: vk::DeviceMemory,
    staging_readback_ptr: *mut u32,
    staging_readback_capacity: usize,
    twiddle_buffer: vk::Buffer,
    twiddle_memory: vk::DeviceMemory,
    twiddle_ptr: *mut u32,
    twiddle_uploaded_log_n: Option<u32>,
    twiddle_capacity: usize,
}

struct VulkanRuntime {
    ctx: VulkanContext,
    pipeline: Option<VulkanPipelineState>,
    io: Option<VulkanIoState>,
    command_buffer: Option<vk::CommandBuffer>,
    submit_fence: Option<vk::Fence>,
    timestamp_query_pool: Option<vk::QueryPool>,
    timestamp_log_emitted: bool,
}

thread_local! {
    static VULKAN_RUNTIME_CACHE: RefCell<Option<VulkanRuntime>> = const { RefCell::new(None) };
}

fn with_vulkan_runtime<T>(f: impl FnOnce(&mut VulkanRuntime) -> Result<T, String>) -> Result<T, String> {
    VULKAN_RUNTIME_CACHE.with(|cached| {
        if cached.borrow().is_none() {
            let ctx = VulkanContext::create()?;
            *cached.borrow_mut() = Some(VulkanRuntime {
                ctx,
                pipeline: None,
                io: None,
                command_buffer: None,
                submit_fence: None,
                timestamp_query_pool: None,
                timestamp_log_emitted: false,
            });
        }
        let mut guard = cached.borrow_mut();
        let runtime = guard
            .as_mut()
            .ok_or_else(|| "vulkan context cache unexpectedly empty".to_string())?;
        f(runtime)
    })
}

impl VulkanRuntime {
    fn ensure_exec_state(&mut self) -> Result<(), String> {
        if self.command_buffer.is_none() {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.ctx.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = unsafe {
                self.ctx
                    .device
                    .allocate_command_buffers(&alloc_info)
                    .map_err(|e| format!("vk allocate command buffer: {e}"))?
                    .remove(0)
            };
            self.command_buffer = Some(command_buffer);
        }
        if self.submit_fence.is_none() {
            let fence_info = vk::FenceCreateInfo::default();
            let fence = unsafe {
                self.ctx
                    .device
                    .create_fence(&fence_info, None)
                    .map_err(|e| format!("vk create fence: {e}"))?
            };
            self.submit_fence = Some(fence);
        }
        Ok(())
    }

    fn ensure_pipeline(&mut self) -> Result<(), String> {
        if self.pipeline.is_some() {
            return Ok(());
        }
        let spv = fft_stage_spv();
        let words = unsafe { core::slice::from_raw_parts(spv.as_ptr() as *const u32, spv.len() / 4) };
        let shader_info = vk::ShaderModuleCreateInfo::default().code(words);
        let shader_module = unsafe {
            self.ctx
                .device
                .create_shader_module(&shader_info, None)
                .map_err(|e| format!("vk create shader module: {e}"))?
        };
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            self.ctx
                .device
                .create_descriptor_set_layout(&set_layout_info, None)
                .map_err(|e| format!("vk create descriptor set layout: {e}"))?
        };
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(core::mem::size_of::<FftStageParams>() as u32);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(core::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(core::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe {
            self.ctx
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| format!("vk create pipeline layout: {e}"))?
        };
        let entry_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            self.ctx
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), core::slice::from_ref(&pipeline_info), None)
                .map_err(|(_, e)| format!("vk create compute pipeline: {e}"))?
                .remove(0)
        };
        self.pipeline = Some(VulkanPipelineState {
            shader_module,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
        });
        Ok(())
    }

    fn ensure_timestamps(&mut self) -> Result<(), String> {
        if !self.ctx.timestamp_supported {
            if !self.timestamp_log_emitted {
                log_vulkan_timing("vulkan timing: timestamps unsupported on this queue/device");
                self.timestamp_log_emitted = true;
            }
            return Ok(());
        }
        if self.timestamp_query_pool.is_some() {
            return Ok(());
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(4);
        let pool = unsafe {
            self.ctx
                .device
                .create_query_pool(&info, None)
                .map_err(|e| format!("vk create timestamp query pool: {e}"))?
        };
        self.timestamp_query_pool = Some(pool);
        log_vulkan_timing("vulkan timing: timestamp query pool created");
        Ok(())
    }

    fn destroy_io(&mut self) {
        if let Some(io) = self.io.take() {
            unsafe {
                self.ctx.device.unmap_memory(io.staging_upload_memory);
                self.ctx.device.unmap_memory(io.staging_readback_memory);
                self.ctx.device.unmap_memory(io.twiddle_memory);
                self.ctx.device.free_memory(io.data_memory, None);
                self.ctx.device.free_memory(io.staging_upload_memory, None);
                self.ctx.device.free_memory(io.staging_readback_memory, None);
                self.ctx.device.free_memory(io.twiddle_memory, None);
                self.ctx.device.destroy_buffer(io.data_buffer, None);
                self.ctx
                    .device
                    .destroy_buffer(io.staging_upload_buffer, None);
                self.ctx
                    .device
                    .destroy_buffer(io.staging_readback_buffer, None);
                self.ctx.device.destroy_buffer(io.twiddle_buffer, None);
                self.ctx
                    .device
                    .destroy_descriptor_pool(io.descriptor_pool, None);
            }
        }
    }

    fn ensure_io(&mut self, data_size: usize, twiddle_size: usize) -> Result<(), String> {
        if let Some(io) = self.io.as_ref() {
            if io.data_capacity >= data_size
                && io.staging_upload_capacity >= data_size
                && io.staging_readback_capacity >= data_size
                && io.twiddle_capacity >= twiddle_size
            {
                return Ok(());
            }
        }

        self.destroy_io();
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| "pipeline must be initialized before IO".to_string())?;

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe {
            self.ctx
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(|e| format!("vk create descriptor pool: {e}"))?
        };
        let set_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(core::slice::from_ref(&pipeline.descriptor_set_layout));
        let descriptor_set = unsafe {
            self.ctx
                .device
                .allocate_descriptor_sets(&set_alloc_info)
                .map_err(|e| format!("vk allocate descriptor set: {e}"))?
                .remove(0)
        };

        let data_buffer_info = vk::BufferCreateInfo::default()
            .size(data_size as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_upload_buffer_info = vk::BufferCreateInfo::default()
            .size(data_size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_readback_buffer_info = vk::BufferCreateInfo::default()
            .size(data_size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let twiddle_buffer_info = vk::BufferCreateInfo::default()
            .size(twiddle_size as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let data_buffer = unsafe {
            self.ctx
                .device
                .create_buffer(&data_buffer_info, None)
                .map_err(|e| format!("vk create data buffer: {e}"))?
        };
        let staging_upload_buffer = unsafe {
            self.ctx
                .device
                .create_buffer(&staging_upload_buffer_info, None)
                .map_err(|e| format!("vk create staging upload buffer: {e}"))?
        };
        let staging_readback_buffer = unsafe {
            self.ctx
                .device
                .create_buffer(&staging_readback_buffer_info, None)
                .map_err(|e| format!("vk create staging readback buffer: {e}"))?
        };
        let twiddle_buffer = unsafe {
            self.ctx
                .device
                .create_buffer(&twiddle_buffer_info, None)
                .map_err(|e| format!("vk create twiddle buffer: {e}"))?
        };

        let data_reqs = unsafe { self.ctx.device.get_buffer_memory_requirements(data_buffer) };
        let staging_upload_reqs =
            unsafe { self.ctx.device.get_buffer_memory_requirements(staging_upload_buffer) };
        let staging_readback_reqs =
            unsafe { self.ctx.device.get_buffer_memory_requirements(staging_readback_buffer) };
        let twiddle_reqs = unsafe { self.ctx.device.get_buffer_memory_requirements(twiddle_buffer) };
        let mem_props = unsafe {
            self.ctx
                .instance
                .get_physical_device_memory_properties(self.ctx.physical_device)
        };
        let device_local_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let host_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let find_type = |reqs: vk::MemoryRequirements, required: vk::MemoryPropertyFlags| -> Result<u32, String> {
            (0..mem_props.memory_type_count)
                .find(|i| {
                    let suitable = (reqs.memory_type_bits & (1 << i)) != 0;
                    suitable
                        && mem_props.memory_types[*i as usize]
                            .property_flags
                            .contains(required)
                })
                .ok_or_else(|| format!("no suitable memory type for flags: {required:?}"))
        };
        let find_type_prefer_device_local = |reqs: vk::MemoryRequirements| -> Result<u32, String> {
            (0..mem_props.memory_type_count)
                .find(|i| {
                    let suitable = (reqs.memory_type_bits & (1 << i)) != 0;
                    suitable
                        && mem_props.memory_types[*i as usize]
                            .property_flags
                            .contains(device_local_flags)
                })
                .or_else(|| {
                    (0..mem_props.memory_type_count).find(|i| {
                        let suitable = (reqs.memory_type_bits & (1 << i)) != 0;
                        suitable
                            && mem_props.memory_types[*i as usize]
                                .property_flags
                                .contains(host_flags)
                    })
                })
                .ok_or_else(|| "no suitable memory type for data buffer".to_string())
        };
        let data_mem_type = find_type_prefer_device_local(data_reqs)?;
        let staging_upload_mem_type = find_type(staging_upload_reqs, host_flags)?;
        let staging_readback_mem_type = find_type(staging_readback_reqs, host_flags)?;
        let twiddle_mem_type = find_type(twiddle_reqs, host_flags)?;

        let data_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(data_reqs.size)
            .memory_type_index(data_mem_type);
        let staging_upload_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_upload_reqs.size)
            .memory_type_index(staging_upload_mem_type);
        let staging_readback_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_readback_reqs.size)
            .memory_type_index(staging_readback_mem_type);
        let twiddle_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(twiddle_reqs.size)
            .memory_type_index(twiddle_mem_type);
        let data_memory = unsafe {
            self.ctx
                .device
                .allocate_memory(&data_alloc, None)
                .map_err(|e| format!("vk alloc data memory: {e}"))?
        };
        let staging_upload_memory = unsafe {
            self.ctx
                .device
                .allocate_memory(&staging_upload_alloc, None)
                .map_err(|e| format!("vk alloc staging upload memory: {e}"))?
        };
        let staging_readback_memory = unsafe {
            self.ctx
                .device
                .allocate_memory(&staging_readback_alloc, None)
                .map_err(|e| format!("vk alloc staging readback memory: {e}"))?
        };
        let twiddle_memory = unsafe {
            self.ctx
                .device
                .allocate_memory(&twiddle_alloc, None)
                .map_err(|e| format!("vk alloc twiddle memory: {e}"))?
        };
        unsafe {
            self.ctx
                .device
                .bind_buffer_memory(data_buffer, data_memory, 0)
                .map_err(|e| format!("vk bind data memory: {e}"))?;
            self.ctx
                .device
                .bind_buffer_memory(staging_upload_buffer, staging_upload_memory, 0)
                .map_err(|e| format!("vk bind staging upload memory: {e}"))?;
            self.ctx
                .device
                .bind_buffer_memory(staging_readback_buffer, staging_readback_memory, 0)
                .map_err(|e| format!("vk bind staging readback memory: {e}"))?;
            self.ctx
                .device
                .bind_buffer_memory(twiddle_buffer, twiddle_memory, 0)
                .map_err(|e| format!("vk bind twiddle memory: {e}"))?;
        }
        let staging_upload_ptr = unsafe {
            self.ctx
                .device
                .map_memory(
                    staging_upload_memory,
                    0,
                    data_size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| format!("vk map staging upload memory: {e}"))?
                as *mut u32
        };
        let staging_readback_ptr = unsafe {
            self.ctx
                .device
                .map_memory(
                    staging_readback_memory,
                    0,
                    data_size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| format!("vk map staging readback memory: {e}"))?
                as *mut u32
        };
        let twiddle_ptr = unsafe {
            self.ctx
                .device
                .map_memory(twiddle_memory, 0, twiddle_size as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("vk map twiddle memory: {e}"))?
                as *mut u32
        };

        let data_desc = vk::DescriptorBufferInfo::default()
            .buffer(data_buffer)
            .offset(0)
            .range(data_size as u64);
        let twiddle_desc = vk::DescriptorBufferInfo::default()
            .buffer(twiddle_buffer)
            .offset(0)
            .range(twiddle_size as u64);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(core::slice::from_ref(&data_desc)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(core::slice::from_ref(&twiddle_desc)),
        ];
        unsafe {
            self.ctx.device.update_descriptor_sets(&writes, &[]);
        }

        self.io = Some(VulkanIoState {
            descriptor_pool,
            descriptor_set,
            data_buffer,
            data_memory,
            data_capacity: data_size,
            staging_upload_buffer,
            staging_upload_memory,
            staging_upload_ptr,
            staging_upload_capacity: data_size,
            staging_readback_buffer,
            staging_readback_memory,
            staging_readback_ptr,
            staging_readback_capacity: data_size,
            twiddle_buffer,
            twiddle_memory,
            twiddle_ptr,
            twiddle_uploaded_log_n: None,
            twiddle_capacity: twiddle_size,
        });
        Ok(())
    }
}

impl VulkanContext {
    fn create() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load().map_err(|e| format!("vk entry load: {e}"))? };

        let app_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"plonky3-android\0") };
        let engine_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"plonky3\0") };
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(engine_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| format!("vk create instance: {e}"))?
        };

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| format!("vk enumerate devices: {e}"))?
        };
        let physical_device = *physical_devices
            .get(0)
            .ok_or_else(|| "no vk physical devices found".to_string())?;

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find(|(_, info)| info.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| "no compute queue family found".to_string())?
        };

        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(core::slice::from_ref(&queue_info));
        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .map_err(|e| format!("vk create device: {e}"))?
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(|e| format!("vk create command pool: {e}"))?
        };

        let dev_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let queue_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_timestamp_bits = queue_props[queue_family_index as usize].timestamp_valid_bits;
        // Some mobile drivers report timestamp_compute_and_graphics=0 while still
        // supporting queue timestamps for compute. Gate on queue timestamp bits.
        let timestamp_supported = queue_timestamp_bits > 0;
        log_vulkan_timing(&format!(
            "vulkan timing init: queue_family={} timestamp_valid_bits={} timestamp_period_ns={:.3} supported={}",
            queue_family_index,
            queue_timestamp_bits,
            dev_props.limits.timestamp_period,
            timestamp_supported
        ));
        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            timestamp_period_ns: dev_props.limits.timestamp_period,
            timestamp_supported,
        })
    }
}

pub fn is_vulkan_available() -> Result<(), String> {
    with_vulkan_runtime(|runtime| {
        runtime.ensure_pipeline()?;
        Ok(())
    })
}

impl Drop for VulkanRuntime {
    fn drop(&mut self) {
        if let Some(pool) = self.timestamp_query_pool.take() {
            unsafe {
                self.ctx.device.destroy_query_pool(pool, None);
            }
        }
        if let Some(fence) = self.submit_fence.take() {
            unsafe {
                self.ctx.device.destroy_fence(fence, None);
            }
        }
        if let Some(command_buffer) = self.command_buffer.take() {
            unsafe {
                self.ctx
                    .device
                    .free_command_buffers(self.ctx.command_pool, &[command_buffer]);
            }
        }
        self.destroy_io();
        if let Some(pipeline) = self.pipeline.take() {
            unsafe {
                self.ctx.device.destroy_pipeline(pipeline.pipeline, None);
                self.ctx
                    .device
                    .destroy_pipeline_layout(pipeline.pipeline_layout, None);
                self.ctx
                    .device
                    .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
                self.ctx.device.destroy_shader_module(pipeline.shader_module, None);
            }
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FftStageParams {
    pub width: u32,
    pub height: u32,
    pub stage: u32,
    pub log_n: u32,
    pub twiddle_base: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub fn params_for_stage(width: usize, height: usize, stage: u32, log_n: u32, twiddle_base: u32) -> FftStageParams {
    FftStageParams {
        width: width as u32,
        height: height as u32,
        stage,
        log_n,
        twiddle_base,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

pub fn fft_stage_spv() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/fft_stage.spv"))
}

pub fn dispatch_dims(params: &FftStageParams) -> (u32, u32, u32) {
    // Dispatch geometry for shader main(gid):
    // - gid.x maps to `col` (batched FFT column index)
    // - gid.y maps to `j`   (butterfly lane index for this stage)
    // - gid.z is unused (kept at 1)
    //
    // Shader workgroup size is 8x8x1.
    // To cover `width` columns, dispatch ceil(width / 8) workgroups in X.
    // To cover butterfly lanes `j`, dispatch ceil((height/2) / 8) workgroups in Y.
    // Any tail lanes in final workgroups can still early-exit with:
    //   if (col >= params.width) return;
    // See shader comments for details.
    //
    // j spans up to height/2 lanes for each stage.
    let j_count = (params.height.max(1)) / 2;
    let width = params.width.max(1);
    let workgroup_x = 8u32;
    let workgroup_y = 8u32;
    let x = width.div_ceil(workgroup_x);
    let y = j_count.max(1).div_ceil(workgroup_y);
    (x, y, 1)
}

pub fn cpu_stage_u32_in_place(data: &mut [u32], params: FftStageParams, twiddles: &[u32]) {
    const PRIME: u32 = 0x7800_0001;
    const MONTY_MU: u32 = 0x8800_0001;
    const MONTY_MASK: u64 = 0xffff_ffff;

    fn add_mod(a: u32, b: u32) -> u32 {
        let sum = a.wrapping_add(b);
        if sum >= PRIME {
            sum - PRIME
        } else {
            sum
        }
    }

    fn sub_mod(a: u32, b: u32) -> u32 {
        if a >= b {
            a - b
        } else {
            a.wrapping_add(PRIME) - b
        }
    }

    fn monty_reduce(x: u64) -> u32 {
        let t = x.wrapping_mul(MONTY_MU as u64) & MONTY_MASK;
        let u = t * (PRIME as u64);
        let (x_sub_u, over) = x.overflowing_sub(u);
        let x_sub_u_hi = (x_sub_u >> 32) as u32;
        if over {
            x_sub_u_hi.wrapping_add(PRIME)
        } else {
            x_sub_u_hi
        }
    }

    fn mul_mod(a: u32, b: u32) -> u32 {
        monty_reduce((a as u64) * (b as u64))
    }

    let width = params.width as usize;
    let height = params.height as usize;
    let stage = params.stage as usize;
    let m = 1usize << (stage + 1);
    let half = m >> 1;

    for col in 0..width {
        for j in 0..(height / 2) {
            let block = j / half;
            let offset = j % half;
            let base = block * m + offset;
            if base + half >= height {
                continue;
            }
            let idx0 = base * width + col;
            let idx1 = (base + half) * width + col;
            let a = data[idx0];
            let b = data[idx1];
            let t = mul_mod(b, twiddles[offset]);
            data[idx0] = add_mod(a, t);
            data[idx1] = sub_mod(a, t);
        }
    }
}

fn monty_to_canonical(x: u32) -> u32 {
    const PRIME: u32 = 0x7800_0001;
    const MONTY_MU: u32 = 0x8800_0001;
    const MONTY_MASK: u64 = 0xffff_ffff;
    let t = (x as u64).wrapping_mul(MONTY_MU as u64) & MONTY_MASK;
    let u = t * (PRIME as u64);
    let (x_sub_u, over) = (x as u64).overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> 32) as u32;
    if over {
        x_sub_u_hi.wrapping_add(PRIME)
    } else {
        x_sub_u_hi
    }
}

#[derive(Debug)]
pub struct VulkanComputePlan {
    pub params: FftStageParams,
    pub dispatch: (u32, u32, u32),
    pub spv_len: usize,
}

pub fn prepare_compute_plan(width: usize, height: usize, stage: u32, log_n: u32) -> VulkanComputePlan {
    let params = params_for_stage(width, height, stage, log_n, 1);
    let dispatch = dispatch_dims(&params);
    let spv_len = fft_stage_spv().len();
    VulkanComputePlan {
        params,
        dispatch,
        spv_len,
    }
}

fn twiddles_for_stage(log_n: u32, stage: u32) -> Vec<u32> {
    let half = 1usize << stage;
    if half == 0 {
        return Vec::new();
    }
    let root = BabyBear::two_adic_generator(log_n as usize);
    let step = root.exp_power_of_2((log_n - stage - 1) as usize);
    step.powers()
        .take(half)
        .map(|v: BabyBear| v.to_unique_u32())
        .collect()
}

fn twiddle_table(log_n: u32) -> (Vec<u32>, Vec<u32>) {
    let mut all = Vec::new();
    let mut stage_base = Vec::with_capacity(log_n as usize);
    for stage in 0..log_n {
        stage_base.push(all.len() as u32);
        all.extend(twiddles_for_stage(log_n, stage));
    }
    (all, stage_base)
}

fn twiddle_stage_base_table(log_n: u32) -> Vec<u32> {
    // Stage s uses 2^s twiddles, so the packed base offset is:
    // base[s] = sum_{k=0..s-1} 2^k = 2^s - 1.
    (0..log_n).map(|s| (1u32 << s) - 1).collect()
}

fn reverse_bits_len_usize(mut x: usize, bits: usize) -> usize {
    let mut y = 0usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

fn write_bit_reversed_rows_u32(dst: &mut [u32], src: &[u32], width: usize) {
    if width == 0 || src.is_empty() {
        return;
    }
    let height = src.len() / width;
    if height == 0 {
        return;
    }
    if !height.is_power_of_two() {
        dst.copy_from_slice(src);
        return;
    }
    let bits = height.trailing_zeros() as usize;
    for row in 0..height {
        let src_row = reverse_bits_len_usize(row, bits);
        let dst_off = row * width;
        let src_off = src_row * width;
        dst[dst_off..dst_off + width].copy_from_slice(&src[src_off..src_off + width]);
    }
}

pub fn setup_vulkan_pipeline_plan(
    plan: &VulkanComputePlan,
    input: &[u32],
) -> Result<Vec<u32>, String> {
    // Dataflow for one Vulkan DFT call:
    // 1) CPU `input` is a row-major flattened h x w matrix (h rows, w columns).
    // 2) We bit-reverse row indices for DIT, then write that into upload staging memory.
    // 3) GPU copies upload staging -> `data_buffer` (main compute buffer).
    // 4) For each FFT stage, one dispatch updates `data_buffer` in-place.
    // 5) GPU copies `data_buffer` -> readback staging.
    // 6) CPU reads readback staging into `gpu_out`.
    //
    // Terminology:
    // - The FFT is applied down each column (transform length = h).
    // - `w` is batch size (number of independent columns transformed together).
    // - One dispatch processes butterflies for all columns for one stage.
    //
    // Memory roles:
    // - upload staging: CPU-write, GPU-read source for upload copy.
    // - data_buffer: primary GPU compute buffer used by shader dispatches.
    // - readback staging: GPU-write destination for copy, then CPU-read.
    let total_start = Instant::now();
    with_vulkan_runtime(|runtime| {
        runtime.ensure_pipeline()?;
        runtime.ensure_exec_state()?;
        runtime.ensure_timestamps()?;
        let data_size = 4usize * input.len();
        let twiddle_total_count = ((plan.params.height as usize).saturating_sub(1)).max(1);
        let twiddle_size = 4usize * twiddle_total_count;
        runtime.ensure_io(data_size, twiddle_size)?;
        let ctx = &runtime.ctx;
        let pipeline_state = runtime
            .pipeline
            .as_ref()
            .ok_or_else(|| "vulkan pipeline cache unexpectedly empty".to_string())?;
        let pipeline_layout = pipeline_state.pipeline_layout;
        let pipeline = pipeline_state.pipeline;
        let io_state = runtime
            .io
            .as_mut()
            .ok_or_else(|| "vulkan IO cache unexpectedly empty".to_string())?;
        let descriptor_set = io_state.descriptor_set;
        let command_buffer = runtime
            .command_buffer
            .ok_or_else(|| "vulkan command buffer cache unexpectedly empty".to_string())?;
        let submit_fence = runtime
            .submit_fence
            .ok_or_else(|| "vulkan submit fence cache unexpectedly empty".to_string())?;

        // Prepare input for DIT directly into upload staging (avoid temporary vec allocation).
        let mut params = plan.params;
        // Stage base offsets are needed for every call; cheap to compute from log_n.
        let twiddle_stage_base = twiddle_stage_base_table(params.log_n);

        let upload_start = Instant::now();
        let twiddles_updated = io_state.twiddle_uploaded_log_n != Some(params.log_n);
        let timestamp_pool = runtime.timestamp_query_pool;
        unsafe {
            let upload_slice = core::slice::from_raw_parts_mut(io_state.staging_upload_ptr, data_size / 4);
            write_bit_reversed_rows_u32(upload_slice, input, plan.params.width as usize);

            if twiddles_updated {
                // Build+upload packed twiddles only when log_n changes.
                let (twiddles_all, _) = twiddle_table(params.log_n);
                let twiddle_slice =
                    core::slice::from_raw_parts_mut(io_state.twiddle_ptr, twiddle_total_count);
                twiddle_slice[..twiddles_all.len()].copy_from_slice(&twiddles_all);
                if twiddles_all.len() < twiddle_total_count {
                    twiddle_slice[twiddles_all.len()..].fill(0);
                }
                io_state.twiddle_uploaded_log_n = Some(params.log_n);
            }
        }
        let upload_ms = upload_start.elapsed().as_millis();

        // Run all FFT stages. `data_buffer` is updated in-place per stage.
        let stages_start = Instant::now();
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            // Step 1: reset reusable execution objects for this call.
            ctx.device
                .reset_fences(&[submit_fence])
                .map_err(|e| format!("vk reset fence: {e}"))?;
            ctx.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("vk reset command buffer: {e}"))?;
            ctx.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| format!("vk begin command buffer: {e}"))?;
            if let Some(pool) = timestamp_pool {
                ctx.device.cmd_reset_query_pool(command_buffer, pool, 0, 4);
                ctx.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    0,
                );
            }
            // Step 2: host->transfer/compute visibility for newly written staging/twiddles.
            let upload_host_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(io_state.staging_upload_buffer)
                .offset(0)
                .size(data_size as u64);
            let mut host_barriers = Vec::with_capacity(2);
            host_barriers.push(upload_host_barrier);
            if twiddles_updated {
                host_barriers.push(
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::HOST_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .buffer(io_state.twiddle_buffer)
                        .offset(0)
                        .size(twiddle_size as u64),
                );
            }
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &host_barriers,
                &[],
            );
            // Step 3: copy CPU-uploaded input into main GPU compute buffer.
            let upload_copy = vk::BufferCopy::default().size(data_size as u64);
            ctx.device.cmd_copy_buffer(
                command_buffer,
                io_state.staging_upload_buffer,
                io_state.data_buffer,
                core::slice::from_ref(&upload_copy),
            );
            // Step 4: make transfer writes visible to compute shader stages.
            let data_upload_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(io_state.data_buffer)
                .offset(0)
                .size(data_size as u64);
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                core::slice::from_ref(&data_upload_barrier),
                &[],
            );
            // Step 5: bind compute pipeline + descriptor set (data buffer + twiddles).
            ctx.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                core::slice::from_ref(&descriptor_set),
                &[],
            );
        }

        for stage in 0..params.log_n {
            // Stage context (DIT radix-2 on each column independently):
            // - stage 0 uses pair distance 1 (size-2 butterflies)
            // - stage 1 uses pair distance 2 (size-4 butterflies)
            // - ...
            // - stage s uses pair distance 2^s and combines size-2^s blocks into size-2^(s+1)
            //
            // All columns are processed in parallel in the same dispatch:
            // each shader invocation handles one (col, j) pair.
            // col chooses which column, j chooses which butterfly lane in that stage.
            params.stage = stage;
            params.twiddle_base = twiddle_stage_base[stage as usize];
            // Packed values pushed to shader for this stage:
            // [width, height, stage, log_n, twiddle_base, pad, pad, pad]
            //
            // - width/height identify matrix shape (height x width).
            // - stage selects butterfly geometry (m, half) inside shader.
            // - twiddle_base points to this stage's twiddle segment.
            let params_words = [
                params.width,
                params.height,
                params.stage,
                params.log_n,
                params.twiddle_base,
                params._pad0,
                params._pad1,
                params._pad2,
            ];
            let params_bytes = unsafe {
                core::slice::from_raw_parts(
                    params_words.as_ptr() as *const u8,
                    core::mem::size_of_val(&params_words),
                )
            };
            unsafe {
                // Update shader-visible stage parameters for *this* dispatch only.
                // Push constants are small command-stream values, not global constants.
                // They are replaced each stage before cmd_dispatch.
                ctx.device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    params_bytes,
                );
                let dispatch = dispatch_dims(&params);
                // Launch grid:
                // - X covers columns in groups of 8 threads  (workgroup_size x = 8)
                // - Y covers butterfly lanes in groups of 8 (workgroup_size y = 8)
                // - Z is unused (=1)
                ctx.device
                    .cmd_dispatch(command_buffer, dispatch.0, dispatch.1, dispatch.2);
            }

            if stage + 1 < params.log_n {
                // Step 6: stage-to-stage dependency within data_buffer.
                // Next stage reads values produced by this stage.
                let stage_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    )
                    .buffer(io_state.data_buffer)
                    .offset(0)
                    .size(data_size as u64);
                unsafe {
                    ctx.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        core::slice::from_ref(&stage_barrier),
                        &[],
                    );
                }
            }
        }
        unsafe {
            if let Some(pool) = timestamp_pool {
                ctx.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    pool,
                    1,
                );
            }
            // Step 7: final compute->transfer visibility before readback copy.
            let pre_copy_readback_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(io_state.data_buffer)
                .offset(0)
                .size(data_size as u64);
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                core::slice::from_ref(&pre_copy_readback_barrier),
                &[],
            );
            // Step 8: copy final GPU results into CPU-readable readback staging.
            let readback_copy = vk::BufferCopy::default().size(data_size as u64);
            ctx.device.cmd_copy_buffer(
                command_buffer,
                io_state.data_buffer,
                io_state.staging_readback_buffer,
                core::slice::from_ref(&readback_copy),
            );
            if let Some(pool) = timestamp_pool {
                ctx.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    pool,
                    2,
                );
            }
        }
        unsafe {
            // Step 9: submit recorded commands and wait for completion of this DFT call.
            ctx.device
                .end_command_buffer(command_buffer)
                .map_err(|e| format!("vk end command buffer: {e}"))?;
        }
        let submit_info =
            vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
        unsafe {
            ctx.device
                .queue_submit(ctx.queue, core::slice::from_ref(&submit_info), submit_fence)
                .map_err(|e| format!("vk queue submit: {e}"))?;
            ctx.device
                .wait_for_fences(&[submit_fence], true, u64::MAX)
                .map_err(|e| format!("vk wait for fence: {e}"))?;
        }
        let mut gpu_timing_ms = None;
        if let Some(pool) = timestamp_pool {
            // Avoid blocking here; some mobile drivers can stall with WAIT despite
            // fence completion. If results are not ready, we skip GPU timing for
            // this call and keep functional behavior unchanged.
            let mut queries = [0u64; 6];
            unsafe {
                let query_res = ctx.device.get_query_pool_results(
                    pool,
                    0,
                    &mut queries,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                );
                if let Err(err) = query_res {
                    if err != vk::Result::NOT_READY {
                        return Err(format!("vk get timestamp query results: {err}"));
                    }
                }
            }
            let all_ready = queries[1] != 0 && queries[3] != 0 && queries[5] != 0;
            if all_ready {
                let period_ns = ctx.timestamp_period_ns as f64;
                let q0 = queries[0] as f64;
                let q1 = queries[2] as f64;
                let q2 = queries[4] as f64;
                gpu_timing_ms = Some((
                    ((q1 - q0) * period_ns) / 1_000_000.0,
                    ((q2 - q1) * period_ns) / 1_000_000.0,
                    ((q2 - q0) * period_ns) / 1_000_000.0,
                ));
            }
        }
        let stages_ms = stages_start.elapsed().as_millis();

        // Read back data.
        let readback_start = Instant::now();
        let mut gpu_out = vec![0u32; data_size / 4];
        unsafe {
            let readback_slice =
                core::slice::from_raw_parts(io_state.staging_readback_ptr as *const u32, data_size / 4);
            gpu_out.copy_from_slice(readback_slice);
        }
        let readback_ms = readback_start.elapsed().as_millis();

        let _ = descriptor_set;
        let total_ms = total_start.elapsed().as_millis();
        if let Some((gpu_stage_ms, gpu_readback_ms, gpu_total_ms)) = gpu_timing_ms {
            log_vulkan_timing(&format!(
                "vulkan dft: h={} w={} stages={} upload={}ms stages={}ms readback={}ms total={}ms gpu(stage={:.3}ms copy_back={:.3}ms total={:.3}ms)",
                plan.params.height,
                plan.params.width,
                plan.params.log_n,
                upload_ms,
                stages_ms,
                readback_ms,
                total_ms,
                gpu_stage_ms,
                gpu_readback_ms,
                gpu_total_ms
            ));
        } else {
            log_vulkan_timing(&format!(
                "vulkan dft: h={} w={} stages={} upload={}ms stages={}ms readback={}ms total={}ms",
                plan.params.height,
                plan.params.width,
                plan.params.log_n,
                upload_ms,
                stages_ms,
                readback_ms,
                total_ms
            ));
        }
        Ok(gpu_out)
    })
}
pub fn dft_batch<F: TwoAdicField>(
    _cpu: &Radix2DitParallel<F>,
    mat: RowMajorMatrix<F>,
) -> Result<RowMajorMatrix<F>, String> {
    let height = mat.height();
    if !height.is_power_of_two() {
        return Err(format!("vulkan backend requires power-of-two height, got {height}"));
    }
    let log_n = height.trailing_zeros();
    let plan = prepare_compute_plan(mat.width(), mat.height(), 0, log_n);
    let input = {
        if core::any::TypeId::of::<F>() != core::any::TypeId::of::<BabyBear>() {
            return Err("vulkan backend currently only supports BabyBear".to_string());
        }
        let values = &mat.values;
        let ptr = values.as_ptr() as *const BabyBear;
        let bb_slice = unsafe { core::slice::from_raw_parts(ptr, values.len()) };
        bb_slice.iter().map(|v| v.to_unique_u32()).collect::<Vec<u32>>()
    };

    // Preserve original input in BabyBear form for CPU/GPU comparison.
    let original_bb_mat = RowMajorMatrix::new(
        input
            .iter()
            .copied()
            .map(|v| BabyBear::new(monty_to_canonical(v)))
            .collect(),
        plan.params.width as usize,
    );

    let gpu_out = setup_vulkan_pipeline_plan(&plan, &input)?;

    // For now, only support BabyBear in the Vulkan path.
    if core::any::TypeId::of::<F>() != core::any::TypeId::of::<BabyBear>() {
        return Err("vulkan backend currently only supports BabyBear".to_string());
    }

    let out_vals: Vec<BabyBear> = gpu_out
        .into_iter()
        .map(|v| BabyBear::new(monty_to_canonical(v)))
        .collect();
    let mut mat = RowMajorMatrix::new(out_vals, plan.params.width as usize);

    // Compare against CPU reference in debug builds only.
    if cfg!(debug_assertions) && core::any::TypeId::of::<F>() == core::any::TypeId::of::<BabyBear>() {
        let cpu_bb = Radix2DitParallel::<BabyBear>::default();
        let cpu_out = cpu_bb.dft_batch(original_bb_mat).to_row_major_matrix();
        let gpu_vals = &mat.values;
        let cpu_vals = &cpu_out.values;
        if gpu_vals.len() != cpu_vals.len() {
            return Err(format!(
                "gpu/cpu length mismatch: gpu {} vs cpu {}",
                gpu_vals.len(),
                cpu_vals.len()
            ));
        }
        for (idx, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
            if g != c {
                // If output matches after row bit-reversal, normalize in-place and continue.
                let mut reordered = mat.clone();
                reverse_matrix_index_bits(&mut reordered);
                if reordered.values == *cpu_vals {
                    mat = reordered;
                    break;
                }
                return Err(format!(
                    "vulkan dft mismatch at idx {idx}: gpu={} cpu={}",
                    g.to_unique_u32(),
                    c.to_unique_u32()
                ));
            }
        }
    }

    // Safety: we just checked F == BabyBear.
    let ptr = Box::into_raw(Box::new(mat)) as *mut RowMajorMatrix<F>;
    let result = unsafe { *Box::from_raw(ptr) };
    Ok(result)
}
