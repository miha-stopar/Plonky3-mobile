use ash::vk;
use ash::Entry;
use std::ffi::CString;
use std::io::Cursor;

const SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));
const WORKGROUP_SIZE: u32 = 64;

pub fn try_run(input: &[i32]) -> Result<Vec<i32>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    unsafe { run_compute(input) }
}

unsafe fn run_compute(input: &[i32]) -> Result<Vec<i32>, String> {
    let entry = Entry::load().map_err(|err| format!("load Vulkan entry: {err:?}"))?;
    let app_name = CString::new("plonky3-android").map_err(|err| err.to_string())?;
    let engine_name = CString::new("plonky3").map_err(|err| err.to_string())?;

    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .engine_name(&engine_name)
        .api_version(vk::make_api_version(0, 1, 1, 0));

    let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    let instance = entry
        .create_instance(&instance_info, None)
        .map_err(|err| format!("create instance: {err:?}"))?;

    let physical_devices = instance
        .enumerate_physical_devices()
        .map_err(|err| format!("enumerate devices: {err:?}"))?;
    let physical_device = *physical_devices
        .first()
        .ok_or_else(|| "no Vulkan physical device found".to_string())?;

    let queue_family_index = instance
        .get_physical_device_queue_family_properties(physical_device)
        .iter()
        .enumerate()
        .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|(index, _)| index as u32)
        .ok_or_else(|| "no compute-capable queue found".to_string())?;

    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&queue_info));
    let device = instance
        .create_device(physical_device, &device_info, None)
        .map_err(|err| format!("create device: {err:?}"))?;
    let queue = device.get_device_queue(queue_family_index, 0);

    let buffer_size = (input.len() * std::mem::size_of::<i32>()) as vk::DeviceSize;
    let input_buffer = create_buffer(
        &instance,
        &device,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let output_buffer = create_buffer(
        &instance,
        &device,
        physical_device,
        buffer_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    map_write(&device, input_buffer.memory, buffer_size, |ptr| {
        std::ptr::copy_nonoverlapping(input.as_ptr() as *const u8, ptr, buffer_size as usize);
    })?;

    let descriptor_set_layout = device
        .create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ]),
            None,
        )
        .map_err(|err| format!("descriptor set layout: {err:?}"))?;

    let pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[descriptor_set_layout]),
            None,
        )
        .map_err(|err| format!("pipeline layout: {err:?}"))?;

    let shader_module = create_shader_module(&device)?;

    let entry_name = CString::new("main").map_err(|err| err.to_string())?;
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_name);

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .stage(stage.build())
        .layout(pipeline_layout);

    let pipeline = device
        .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
        .map_err(|(_, err)| format!("create pipeline: {err:?}"))?
        [0];

    let descriptor_pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&[vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 2,
                }]),
            None,
        )
        .map_err(|err| format!("descriptor pool: {err:?}"))?;

    let descriptor_set = device
        .allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout]),
        )
        .map_err(|err| format!("descriptor set: {err:?}"))?
        [0];

    let input_info = vk::DescriptorBufferInfo {
        buffer: input_buffer.buffer,
        offset: 0,
        range: buffer_size,
    };
    let output_info = vk::DescriptorBufferInfo {
        buffer: output_buffer.buffer,
        offset: 0,
        range: buffer_size,
    };

    let writes = [
        vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&input_info))
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&output_info))
            .build(),
    ];
    device.update_descriptor_sets(&writes, &[]);

    let command_pool = device
        .create_command_pool(
            &vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family_index),
            None,
        )
        .map_err(|err| format!("command pool: {err:?}"))?;

    let command_buffer = device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .map_err(|err| format!("command buffer: {err:?}"))?
        [0];

    device
        .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder())
        .map_err(|err| format!("begin command buffer: {err:?}"))?;

    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[descriptor_set],
        &[],
    );

    let group_count = (input.len() as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    device.cmd_dispatch(command_buffer, group_count, 1, 1);

    device
        .end_command_buffer(command_buffer)
        .map_err(|err| format!("end command buffer: {err:?}"))?;

    let submit_info = vk::SubmitInfo::builder().command_buffers(&[command_buffer]);
    device
        .queue_submit(queue, &[submit_info.build()], vk::Fence::null())
        .map_err(|err| format!("queue submit: {err:?}"))?;
    device
        .queue_wait_idle(queue)
        .map_err(|err| format!("queue wait: {err:?}"))?;

    let mut output = vec![0i32; input.len()];
    map_read(&device, output_buffer.memory, buffer_size, |ptr| {
        std::ptr::copy_nonoverlapping(ptr as *const i32, output.as_mut_ptr(), output.len());
    })?;

    device.destroy_command_pool(command_pool, None);
    device.destroy_descriptor_pool(descriptor_pool, None);
    device.destroy_pipeline(pipeline, None);
    device.destroy_shader_module(shader_module, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
    destroy_buffer(&device, input_buffer);
    destroy_buffer(&device, output_buffer);
    device.destroy_device(None);
    instance.destroy_instance(None);

    Ok(output)
}

struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

unsafe fn create_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<BufferResource, String> {
    let buffer = device
        .create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .map_err(|err| format!("create buffer: {err:?}"))?;

    let requirements = device.get_buffer_memory_requirements(buffer);
    let memory_type_index = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index),
            None,
        )
        .map_err(|err| format!("allocate memory: {err:?}"))?;

    device
        .bind_buffer_memory(buffer, memory, 0)
        .map_err(|err| format!("bind buffer: {err:?}"))?;

    Ok(BufferResource { buffer, memory })
}

unsafe fn destroy_buffer(device: &ash::Device, buffer: BufferResource) {
    device.destroy_buffer(buffer.buffer, None);
    device.free_memory(buffer.memory, None);
}

unsafe fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    for (index, memory_type) in memory_properties.memory_types.iter().enumerate() {
        let matches = (type_filter & (1 << index)) != 0
            && memory_type.property_flags.contains(properties);
        if matches {
            return Ok(index as u32);
        }
    }

    Err("failed to find suitable memory type".to_string())
}

unsafe fn create_shader_module(device: &ash::Device) -> Result<vk::ShaderModule, String> {
    let mut cursor = Cursor::new(SHADER_SPV);
    let code = ash::util::read_spv(&mut cursor).map_err(|err| format!("read SPIR-V: {err:?}"))?;
    device
        .create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&code), None)
        .map_err(|err| format!("shader module: {err:?}"))
}

unsafe fn map_write(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    write_fn: impl FnOnce(*mut u8),
) -> Result<(), String> {
    let ptr = device
        .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        .map_err(|err| format!("map memory: {err:?}"))? as *mut u8;
    write_fn(ptr);
    device.unmap_memory(memory);
    Ok(())
}

unsafe fn map_read(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    read_fn: impl FnOnce(*const u8),
) -> Result<(), String> {
    let ptr = device
        .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        .map_err(|err| format!("map memory: {err:?}"))? as *const u8;
    read_fn(ptr);
    device.unmap_memory(memory);
    Ok(())
}
