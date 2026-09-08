//! Example: Query Codec Capabilities
//!
//! This example demonstrates how to query video codec capabilities.
//! from the Vulkan video extensions.

use ash::vk;
use ash::vk::TaggedStructure;
use pixelforge::{Codec, VideoContextBuilder};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            ),
        )
        .init();

    println!("PixelForge Codec Capabilities Example");
    println!("======================================\n");

    // Build the video context.
    let context = VideoContextBuilder::new()
        .app_name("Capabilities Example")
        .app_version(1, 0, 0)
        .enable_validation(cfg!(debug_assertions))
        .build()?;

    println!("Video Context Created\n");

    // Load video queue fn to get capabilities
    let video_queue_fn = ash::khr::video_queue::Instance::load(context.entry(), context.instance());

    // Query codec support.
    println!("Codec Support:");
    println!("--------------");

    let codecs = [Codec::H264, Codec::H265, Codec::AV1];

    for codec in codecs {
        println!("\n{:?}:", codec);

        // Check encode support.
        let encode_supported = context.supports_encode(codec);
        println!(
            "  Encode: {}",
            if encode_supported {
                "✓ Supported"
            } else {
                "✗ Not supported"
            }
        );

        if encode_supported {
            query_detailed_capabilities(&context, codec, &video_queue_fn)?;
        }
    }

    Ok(())
}

fn query_detailed_capabilities(
    context: &pixelforge::VideoContext,
    codec: Codec,
    video_queue_fn: &ash::khr::video_queue::Instance,
) -> Result<(), Box<dyn std::error::Error>> {
    let physical_device = context.physical_device();

    let combinations = [
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            "4:2:0 8-bit",
        ),
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            "4:4:4 8-bit",
        ),
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            "4:2:0 10-bit",
        ),
        (
            vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            "4:4:4 10-bit",
        ),
    ];

    for (subsampling, bit_depth, desc) in combinations {
        println!("    Checking {}: ", desc);

        // Construct profile info
        let (mut profile_info, mut h264_profile, mut h265_profile, mut av1_profile) = match codec {
            Codec::H264 => {
                let profile_idc = if subsampling == vk::VideoChromaSubsamplingFlagsKHR::TYPE_444 {
                    ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
                } else {
                    ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH
                };

                let h264 =
                    vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(profile_idc);
                let info = vk::VideoProfileInfoKHR::default()
                    .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
                    .chroma_subsampling(subsampling)
                    .luma_bit_depth(bit_depth)
                    .chroma_bit_depth(bit_depth);
                (info, Some(h264), None, None)
            }
            Codec::H265 => {
                let profile_idc = if subsampling == vk::VideoChromaSubsamplingFlagsKHR::TYPE_444 {
                    ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
                } else if bit_depth == vk::VideoComponentBitDepthFlagsKHR::TYPE_10 {
                    ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
                } else {
                    ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
                };

                let h265 =
                    vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);
                let info = vk::VideoProfileInfoKHR::default()
                    .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
                    .chroma_subsampling(subsampling)
                    .luma_bit_depth(bit_depth)
                    .chroma_bit_depth(bit_depth);
                (info, None, Some(h265), None)
            }
            Codec::AV1 => {
                let profile = if subsampling == vk::VideoChromaSubsamplingFlagsKHR::TYPE_444 {
                    ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH
                } else {
                    ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN
                };

                let av1 = vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(profile);
                let info = vk::VideoProfileInfoKHR::default()
                    .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
                    .chroma_subsampling(subsampling)
                    .luma_bit_depth(bit_depth)
                    .chroma_bit_depth(bit_depth);
                (info, None, None, Some(av1))
            }
        };

        if let Some(h264) = &mut h264_profile {
            profile_info = profile_info.push(h264);
        }
        if let Some(h265) = &mut h265_profile {
            profile_info = profile_info.push(h265);
        }
        if let Some(av1) = &mut av1_profile {
            profile_info = profile_info.push(av1);
        }

        // 1. Query Video Capabilities
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default().push(&mut encode_caps);

        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut h265_caps = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut av1_caps = vk::VideoEncodeAV1CapabilitiesKHR::default();
        match codec {
            Codec::H264 => caps = caps.push(&mut h264_caps),
            Codec::H265 => caps = caps.push(&mut h265_caps),
            Codec::AV1 => caps = caps.push(&mut av1_caps),
        }

        let result = unsafe {
            (video_queue_fn
                .fp()
                .get_physical_device_video_capabilities_khr)(
                physical_device,
                &profile_info,
                &mut caps,
            )
        };

        if result != vk::Result::SUCCESS {
            println!("      Not Supported ({:?})", result);
            continue;
        }
        println!("      Supported");
        println!(
            "      Max Dimenstions: {}x{}",
            caps.max_coded_extent.width, caps.max_coded_extent.height
        );
        println!(
            "      Max Reference Pictures: {}",
            caps.max_active_reference_pictures
        );
        println!("      Max DPB Slots: {}", caps.max_dpb_slots);

        // 2. Query Supported Formats
        let mut format_props_count = 0;
        let mut format_props_list =
            vk::VideoProfileListInfoKHR::default().profiles(std::slice::from_ref(&profile_info));

        // Check for Input Image support (VIDEO_ENCODE_SRC_KHR)
        let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR)
            .push(&mut format_props_list);

        let result = unsafe {
            (video_queue_fn
                .fp()
                .get_physical_device_video_format_properties_khr)(
                physical_device,
                &format_info,
                &mut format_props_count,
                std::ptr::null_mut(),
            )
        };

        if result == vk::Result::SUCCESS {
            let mut format_props =
                vec![vk::VideoFormatPropertiesKHR::default(); format_props_count as usize];
            unsafe {
                let _ = (video_queue_fn
                    .fp()
                    .get_physical_device_video_format_properties_khr)(
                    physical_device,
                    &format_info,
                    &mut format_props_count,
                    format_props.as_mut_ptr(),
                );
            };
            println!("      Supported Input Formats (SRC):");
            for prop in format_props {
                println!("        Format: {:?}", prop.format);
            }
        }

        // Check for DPB Image support (VIDEO_ENCODE_DPB_KHR)
        let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR)
            .push(&mut format_props_list);

        let mut format_props_count = 0;
        let result = unsafe {
            (video_queue_fn
                .fp()
                .get_physical_device_video_format_properties_khr)(
                physical_device,
                &format_info,
                &mut format_props_count,
                std::ptr::null_mut(),
            )
        };

        if result == vk::Result::SUCCESS {
            let mut format_props =
                vec![vk::VideoFormatPropertiesKHR::default(); format_props_count as usize];
            unsafe {
                let _ = (video_queue_fn
                    .fp()
                    .get_physical_device_video_format_properties_khr)(
                    physical_device,
                    &format_info,
                    &mut format_props_count,
                    format_props.as_mut_ptr(),
                );
            };
            println!("      Supported DPB Formats (DPB):");
            for prop in format_props {
                println!("        Format: {:?}", prop.format);
            }
        }
    }

    Ok(())
}
