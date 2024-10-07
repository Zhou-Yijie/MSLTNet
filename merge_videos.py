from moviepy.editor import VideoFileClip, clips_array, vfx

def merge_videos_side_by_side(input_video, output_video, result_video):
    # Load the video clips
    clip1 = VideoFileClip(input_video)
    clip2 = VideoFileClip(output_video)

    # Ensure both clips have the same duration
    min_duration = min(clip1.duration, clip2.duration)
    clip1 = clip1.subclip(0, min_duration)
    clip2 = clip2.subclip(0, min_duration)

    # Resize the clips to fit side by side in a 1920x1080 frame
    clip1 = clip1.resize(width=960)
    clip2 = clip2.resize(width=960)

    # Create the side-by-side video
    final_clip = clips_array([[clip1, clip2]])

    # Write the result to a file
    final_clip.write_videofile(result_video, codec="libx264", audio_codec="aac")

    # Close the clips
    clip1.close()
    clip2.close()
    final_clip.close()

# Usage
input_video = "sample_video/Nokia 3.4 low-light video recording sample.mp4"
output_video = "sample_video/output_Nokia 3.4 low-light video recording sample.mp4/output_video.mp4"
result_video = "merged_video_1_720p.mp4"

merge_videos_side_by_side(input_video, output_video, result_video)