import ffmpeg
import os

# Handy script just to crop or trim movies etc for demonstration purposes.

h264_tune = 'film'
# input_file = "J:/Data/s2p/20210823_16_59_50_1114353/movies/reg.mp4"
# output_file_name = "reg_20210823_16_59_50_1114353.mp4"
# input_file = "J:/Data/s2p/20210823_16_59_50_1114353/movies/raw.mp4"
# output_file_name = "raw_20210823_16_59_50_1114353.mp4"


# input_file = "J:/Data/s2p/20220411_16_45_08_1116663/movies/reg.mp4"
# output_file_name = "reg_20220411_16_45_08_1116663.mp4"



# input_file = "J:/Data/s2p/20210920_11_09_37_1114356/movies/reg.mp4"
# output_file_name = "reg_20210920_11_09_37_1114356.mp4"
# input_file = "J:/Data/s2p/20210920_11_09_37_1114356/movies/raw.mp4"
# output_file_name = "raw_20210920_11_09_37_1114356.mp4"

# input_file = "L:/Data/m2p/video/20210823_16_59_50_1114353/20210823_17_00_04_1114353_maze-rose_overhead.camera-halfres_lowfps.mp4"
# output_file_name = "short2_20210823_17_00_04_1114353_maze-rose_overhead.camera-halfres_lowfps.mp4"


# input_file = "L:/Data/m2p/s2p/20211216_14_36_39_1115816/movies/raw.mp4"
# output_file_name = "Benoit_1115816.mp4"
#
# start_frame = 8000
# end_frame = start_frame + 300
#
#
# output_file_path = "L:/Data/movie-edits"
# output_file = os.path.join(output_file_path, output_file_name)
#
# # https://trac.ffmpeg.org/wiki/Encode/H.264
# # Controls compression/speed trade off.
# h264_preset = 'fast'
# # H264 quality, apparently 17 is indistinguishable from lossless.
# h264_crf = '17'
#
#

# input_file = "stitched-2.mp4"
# output_file = "stitched-2-trim.mp4"
# print("Trim movie")
# (
#     ffmpeg
#     .input(input_file)
#     .trim(start_frame=0, end_frame=1200)
#     .setpts('0.25*PTS')
#     .filter('pp', 'al')
#     .output(output_file, vcodec='h264', format='mp4')
#     .overwrite_output()
#     .run()
# )
#
# print("Trim movie speed4x")
# (
#     ffmpeg
#     .input(input_file)
#     .trim(start_frame=start_frame, end_frame=end_frame)
#     .setpts('0.25*PTS')
#     .filter('pp', 'al')
#     .output(output_file, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
#     .overwrite_output()
#     .run()
# )
#
#
#
# # print("Cropping")
# # (
# #     ffmpeg
# #     .input(input_file)
# #     #.filter('crop', '{out_w}:{out_h}:{x}:{y}'.format(out_w=width, out_h=height, x=x, y=y))
# #     .crop(x, y, width, height)
# #     .output(output_file, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
# #     .overwrite_output()
# #     .run()
# # )
# #
# # print("Downsample res by half")
# # (
# #     ffmpeg
# #     .input(input_file)
# #     .filter('scale', 'in_w*0.5', 'in_h*0.5')
# #     .output(output_file, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
# #     .overwrite_output()
# #     .run()
# # )
#
#
# # print("Downsample fps to something more normal (30)")
# # (
# #     ffmpeg
# #     .input(input_file)
# #     .filter('fps', '30')
# #     .output(output_file, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
# #     .overwrite_output()
# #     .run()
# # )\


# input1 = "/Users/tristan/Neuro/hm2p/video/20221115_13_27_42_1118213/20221115_13_27_52_1118213_maze-rose_overhead.camera-cropped.mp4"
# input2 = "/Users/tristan/Neuro/hm2p/s2p/20221115_13_27_42_1118213/movies/raw.mp4"
# output = "stitched-2.mp4"

# input1 = "/Users/tristan/Neuro/hm2p/video/20220411_16_45_08_1116663/20220411_16_45_16_1116663_maze-rose_overhead.camera-cropped.mp4"
# input2 = "/Users/tristan/Neuro/hm2p/s2p/20220411_16_45_08_1116663/movies/reg.mp4"
# output = "stitched-20220411_16_45_08_1116663.mp4"


# input1 = "/Users/tristan/Neuro/hm2p/video/20221115_13_27_42_1118213/20221115_13_27_52_1118213_maze-rose_overhead.camera-cropped.mp4"
# input2 = "/Users/tristan/Neuro/hm2p/s2p/20221115_13_27_42_1118213/movies/reg.mp4"
# output = "stitched-20221115_13_27_42_1118213.mp4"
#
# # Get resolution and frame rate of input2
# probe = ffmpeg.probe(input2)
# video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
# height = video_stream['height']
# fps = eval(video_stream['avg_frame_rate'])
#
# # Adjust input1
# stream1 = (
#     ffmpeg
#     .input(input1)
#     .filter('scale', '-2', height)  # Adjust the height and keep the aspect ratio. '-2' ensures even width.
#     .filter('fps', fps=fps)         # Adjust the frame rate
# )
#
# stream2 = ffmpeg.input(input2)
#
# # Use the hstack filter to stitch the videos side by side
# merged_video = ffmpeg.filter([stream1, stream2], 'hstack', inputs=2)
#
# # Ensure the merged video's width is even
# merged_video = merged_video.filter('crop', 'floor(iw/2)*2', 'ih')
#
# (
#     ffmpeg
#     .output(merged_video, output)
#     .run()
# )

input_file = "stitched-20221115_13_27_42_1118213.mp4"
output_file = "stitched-20221115_13_27_42_1118213-4x.mp4"
print("Trim movie speed4x")
(
    ffmpeg
    .input(input_file)
    .trim(start_frame=0, end_frame=1200)
    .setpts('0.25*PTS')
    .filter('pp', 'al')
    .output(output_file, vcodec='h264', format='mp4')
    .overwrite_output()
    .run()
)

print("Done")