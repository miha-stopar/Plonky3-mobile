use p3_dft::Radix2DitParallel;
use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use ash::vk;

struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
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

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
        })
    }
}

pub fn is_vulkan_available() -> Result<(), String> {
    let _ctx = VulkanContext::create()?;
    Ok(())
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
    // Each thread handles one (row, j) pair. j ranges over width/2 for each row.
    let half = 1u32 << params.stage;
    let j_count = (params.width + 1).max(1) / 2;
    let x = j_count.max(half);
    (x, params.height.max(1), 1)
}

pub fn cpu_stage_u32_in_place(data: &mut [u32], params: FftStageParams) {
    let width = params.width as usize;
    let height = params.height as usize;
    let stage = params.stage as usize;
    let m = 1usize << (stage + 1);
    let half = m >> 1;

    for row in 0..height {
        let row_start = row * width;
        for j in 0..(width / 2) {
            let block = j / half;
            let offset = j % half;
            let base = block * m + offset;
            if base + half >= width {
                continue;
            }
            let idx0 = row_start + base;
            let idx1 = idx0 + half;
            let a = data[idx0];
            let b = data[idx1];
            let t = b.wrapping_mul(params.twiddle_base);
            data[idx0] = a.wrapping_add(t);
            data[idx1] = a.wrapping_sub(t);
        }
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

pub fn setup_vulkan_pipeline_plan(_plan: &VulkanComputePlan) -> Result<(), String> {
    // Pipeline setup outline (not implemented yet):
    // 1) Create instance + select physical device.
    // 2) Create logical device + compute queue.
    // 3) Create shader module from fft_stage.spv bytes.
    // 4) Create descriptor set layout for storage buffer + uniform buffer.
    // 5) Create pipeline layout + compute pipeline.
    // 6) Allocate buffers, upload params + data.
    // 7) Record command buffer: bind pipeline, bind descriptors, dispatch.
    // 8) Submit + wait, then read back.
    let ctx = VulkanContext::create()?;

    let spv = fft_stage_spv();
    let words = unsafe {
        core::slice::from_raw_parts(spv.as_ptr() as *const u32, spv.len() / 4)
    };
    let shader_info = vk::ShaderModuleCreateInfo::default().code(words);
    let shader_module = unsafe {
        ctx.device
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
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let descriptor_set_layout = unsafe {
        ctx.device
            .create_descriptor_set_layout(&set_layout_info, None)
            .map_err(|e| format!("vk create descriptor set layout: {e}"))?
    };

    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(core::slice::from_ref(&descriptor_set_layout));
    let pipeline_layout = unsafe {
        ctx.device
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
        ctx.device
            .create_compute_pipelines(vk::PipelineCache::null(), core::slice::from_ref(&pipeline_info), None)
            .map_err(|(_, e)| format!("vk create compute pipeline: {e}"))?
            .remove(0)
    };

    let descriptor_pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        },
    ];
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&descriptor_pool_sizes)
        .max_sets(1);
    let descriptor_pool = unsafe {
        ctx.device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .map_err(|e| format!("vk create descriptor pool: {e}"))?
    };

    let set_alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(core::slice::from_ref(&descriptor_set_layout));
    let descriptor_set = unsafe {
        ctx.device
            .allocate_descriptor_sets(&set_alloc_info)
            .map_err(|e| format!("vk allocate descriptor set: {e}"))?
            .remove(0)
    };

    // Buffer creation and descriptor writes (host-visible for now).
    let data_size = 4usize * 16;
    let params_size = core::mem::size_of::<FftStageParams>();

    let buffer_info = vk::BufferCreateInfo::default()
        .size(data_size as u64)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let data_buffer = unsafe {
        ctx.device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("vk create data buffer: {e}"))?
    };

    let uniform_info = vk::BufferCreateInfo::default()
        .size(params_size as u64)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let params_buffer = unsafe {
        ctx.device
            .create_buffer(&uniform_info, None)
            .map_err(|e| format!("vk create params buffer: {e}"))?
    };

    let data_reqs = unsafe { ctx.device.get_buffer_memory_requirements(data_buffer) };
    let params_reqs = unsafe { ctx.device.get_buffer_memory_requirements(params_buffer) };
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    let host_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let find_type = |reqs: vk::MemoryRequirements| -> Result<u32, String> {
        (0..mem_props.memory_type_count)
            .find(|i| {
                let suitable = (reqs.memory_type_bits & (1 << i)) != 0;
                suitable && mem_props.memory_types[*i as usize].property_flags.contains(host_flags)
            })
            .ok_or_else(|| "no suitable host visible memory type".to_string())
    };
    let data_mem_type = find_type(data_reqs)?;
    let params_mem_type = find_type(params_reqs)?;

    let data_alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(data_reqs.size)
        .memory_type_index(data_mem_type);
    let params_alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(params_reqs.size)
        .memory_type_index(params_mem_type);
    let data_memory = unsafe {
        ctx.device
            .allocate_memory(&data_alloc, None)
            .map_err(|e| format!("vk alloc data memory: {e}"))?
    };
    let params_memory = unsafe {
        ctx.device
            .allocate_memory(&params_alloc, None)
            .map_err(|e| format!("vk alloc params memory: {e}"))?
    };
    unsafe {
        ctx.device
            .bind_buffer_memory(data_buffer, data_memory, 0)
            .map_err(|e| format!("vk bind data memory: {e}"))?;
        ctx.device
            .bind_buffer_memory(params_buffer, params_memory, 0)
            .map_err(|e| format!("vk bind params memory: {e}"))?;
    }

    let data_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(data_buffer)
        .offset(0)
        .range(data_size as u64);
    let params_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(params_buffer)
        .offset(0)
        .range(params_size as u64);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(core::slice::from_ref(&data_buffer_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(core::slice::from_ref(&params_buffer_info)),
    ];
    unsafe {
        ctx.device.update_descriptor_sets(&writes, &[]);
    }

    // Write sample data + params.
    let params = params_for_stage(16, 1, 0, 4, 1);
    unsafe {
        let data_ptr = ctx
            .device
            .map_memory(data_memory, 0, data_size as u64, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("vk map data memory: {e}"))?;
        let params_ptr = ctx
            .device
            .map_memory(params_memory, 0, params_size as u64, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("vk map params memory: {e}"))?;

        let data_slice = core::slice::from_raw_parts_mut(data_ptr as *mut u32, data_size / 4);
        for (i, value) in data_slice.iter_mut().enumerate() {
            *value = i as u32;
        }
        let params_slice = core::slice::from_raw_parts_mut(params_ptr as *mut FftStageParams, 1);
        params_slice[0] = params;

        ctx.device.unmap_memory(data_memory);
        ctx.device.unmap_memory(params_memory);
    }

    // Record and dispatch a compute command buffer (single stage).
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        ctx.device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| format!("vk allocate command buffer: {e}"))?
            .remove(0)
    };
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        ctx.device
            .begin_command_buffer(command_buffer, &begin_info)
            .map_err(|e| format!("vk begin command buffer: {e}"))?;
        ctx.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            core::slice::from_ref(&descriptor_set),
            &[],
        );
        let dispatch = dispatch_dims(&params);
        ctx.device.cmd_dispatch(command_buffer, dispatch.0, dispatch.1, dispatch.2);
        ctx.device
            .end_command_buffer(command_buffer)
            .map_err(|e| format!("vk end command buffer: {e}"))?;
    }

    let submit_info = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
    unsafe {
        ctx.device
            .queue_submit(ctx.queue, core::slice::from_ref(&submit_info), vk::Fence::null())
            .map_err(|e| format!("vk queue submit: {e}"))?;
        ctx.device
            .queue_wait_idle(ctx.queue)
            .map_err(|e| format!("vk queue wait idle: {e}"))?;
        ctx.device.free_command_buffers(ctx.command_pool, &[command_buffer]);
    }

    // Read back data and compare against CPU stage.
    let mut gpu_out = vec![0u32; data_size / 4];
    unsafe {
        let data_ptr = ctx
            .device
            .map_memory(data_memory, 0, data_size as u64, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("vk map data memory (read): {e}"))?;
        let data_slice = core::slice::from_raw_parts(data_ptr as *const u32, data_size / 4);
        gpu_out.copy_from_slice(data_slice);
        ctx.device.unmap_memory(data_memory);
    }

    let mut cpu_out: Vec<u32> = (0u32..(data_size as u32 / 4)).collect();
    cpu_stage_u32_in_place(&mut cpu_out, params);
    if gpu_out != cpu_out {
        return Err("vulkan stage mismatch vs cpu (placeholder arithmetic)".to_string());
    }

    unsafe {
        ctx.device.free_memory(data_memory, None);
        ctx.device.free_memory(params_memory, None);
        ctx.device.destroy_buffer(data_buffer, None);
        ctx.device.destroy_buffer(params_buffer, None);
        ctx.device.destroy_descriptor_pool(descriptor_pool, None);
        ctx.device.destroy_pipeline(pipeline, None);
        ctx.device.destroy_pipeline_layout(pipeline_layout, None);
        ctx.device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        ctx.device.destroy_shader_module(shader_module, None);
    }

    let _ = descriptor_set;
    Err("vulkan pipeline setup not implemented (dispatch + readback wired)".to_string())
}
pub fn dft_batch<F: TwoAdicField>(
    _cpu: &Radix2DitParallel<F>,
    mat: RowMajorMatrix<F>,
) -> Result<RowMajorMatrix<F>, String> {
    let log_n = mat.height().next_power_of_two().trailing_zeros();
    let plan = prepare_compute_plan(mat.width(), mat.height(), 0, log_n);
    let _ = setup_vulkan_pipeline_plan(&plan);
    Err("vulkan backend not implemented (shader compiled, params wired, cpu stage ready)".to_string())
}
